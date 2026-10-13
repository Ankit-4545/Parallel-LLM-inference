"""Layer partitioning of GPT-2 into pipeline stages (objective O2).

`StageModule` is a stateless re-implementation of the GPT-2 forward pass for a
contiguous range of transformer blocks, built from the pretrained tensors only.
The arithmetic mirrors Hugging Face's eager GPT-2 implementation op for op
(Conv1D = addmm, gelu_new, eager attention with a finfo.min causal mask), so a
split model reproduces the single-process logits.

Batching uses *selective batching*: the rows of all items in a micro-batch are
concatenated into one [sum(n_i), hidden] matrix for every row-wise operation
(layer norms, QKV/out projections, MLP), while attention is computed per item
against that item's own KV cache. Requests with different prompt lengths and
cache lengths can therefore share a micro-batch without padding or masks, and
prefill and decode items can be mixed.

`PipelineStage` wraps a `StageModule` and a `KVCacheManager` and exposes the
`forward(hidden_states, request_id, step)` interface required by O2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pllm.config import parse_split
from pllm.kv_cache import KV, KVCacheManager, RequestCache
from pllm.model_loader import GPT2Config, load_tensors, resolve_model_dir


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    # Identical expression to transformers.activations.NewGELUActivation.
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def stage_ranges(cfg: GPT2Config, num_stages: int, split: str = "6") -> list[tuple[int, int]]:
    embed = cfg.vocab_size * cfg.n_embd + cfg.n_positions * cfg.n_embd
    head = cfg.vocab_size * cfg.n_embd + 2 * cfg.n_embd  # tied LM head (a copy of wte) + ln_f
    return parse_split(split, cfg.n_layer, num_stages, [cfg.params_per_layer()] * cfg.n_layer, embed, head)


@dataclass
class SegmentInput:
    """One micro-batch item as seen by `StageModule.forward_segments`."""
    n_tokens: int
    past: list[KV | None] | None  # per local layer; None = no cache (fresh prefill or cache disabled)


class StageModule(torch.nn.Module):
    def __init__(self, cfg: GPT2Config, layer_start: int, layer_end: int,
                 tensors: dict[str, torch.Tensor]):
        super().__init__()
        self.cfg = cfg
        self.layer_start, self.layer_end = layer_start, layer_end
        self.num_local_layers = layer_end - layer_start
        self.has_embeddings = layer_start == 0
        self.has_head = layer_end == cfg.n_layer
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.eps = cfg.layer_norm_epsilon

        def p(name: str) -> torch.nn.Parameter:
            return torch.nn.Parameter(tensors[name], requires_grad=False)

        if self.has_embeddings:
            self.wte = p("wte.weight")
            self.wpe = p("wpe.weight")
        self.blocks = torch.nn.ModuleList()
        for i in range(layer_start, layer_end):
            b = torch.nn.Module()
            for name in ("ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias",
                         "attn.c_proj.weight", "attn.c_proj.bias", "ln_2.weight", "ln_2.bias",
                         "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias"):
                b.register_parameter(name.replace(".", "_"), p(f"h.{i}.{name}"))
            self.blocks.append(b)
        if self.has_head:
            self.ln_f_weight = p("ln_f.weight")
            self.ln_f_bias = p("ln_f.bias")
            # GPT-2 ties the LM head to the token embedding; the last stage needs its own copy.
            self.lm_head = self.wte if self.has_embeddings else p("wte.weight")
        self.eval()

    # -- construction -------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_dir, layer_start: int, layer_end: int) -> "StageModule":
        cfg = GPT2Config.from_json(model_dir / "config.json")
        first, last = layer_start == 0, layer_end == cfg.n_layer

        def wanted(name: str) -> bool:
            if name.startswith("h."):
                idx = int(name.split(".")[1])
                return layer_start <= idx < layer_end and not name.endswith((".attn.bias", ".attn.masked_bias"))
            if name == "wte.weight":
                return first or last
            if name == "wpe.weight":
                return first
            if name.startswith("ln_f."):
                return last
            return False

        return cls(cfg, layer_start, layer_end, load_tensors(model_dir, wanted))

    def param_bytes(self) -> int:
        seen, total = set(), 0
        for prm in self.parameters():
            if id(prm) not in seen:
                seen.add(id(prm))
                total += prm.numel() * prm.element_size()
        return total

    def num_params(self) -> int:
        return self.param_bytes() // 4

    # -- forward ------------------------------------------------------------
    def embed(self, token_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """token_ids, positions: [n] -> hidden [n, d]."""
        return F.embedding(token_ids, self.wte) + F.embedding(positions, self.wpe)

    @staticmethod
    def _conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.addmm(bias, x, weight)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Eager causal attention for one item. q: [1,H,n,D], k/v: [1,H,T,D] with T >= n."""
        w = torch.matmul(q, k.transpose(-1, -2))
        w = w / torch.full([], v.size(-1) ** 0.5, dtype=w.dtype, device=w.device)
        n, t = q.size(-2), k.size(-2)
        if n > 1:
            # query i (absolute position t-n+i) may attend to keys 0..t-n+i
            mask = torch.ones(n, t, dtype=torch.bool).tril(diagonal=t - n)
            w = torch.where(mask, w, torch.full([], torch.finfo(w.dtype).min, dtype=w.dtype))
        w = F.softmax(w, dim=-1).type(v.dtype)
        return torch.matmul(w, v)

    def forward_segments(self, x: torch.Tensor, segments: list[SegmentInput]
                         ) -> tuple[torch.Tensor, list[list[KV]]]:
        """Run this stage's blocks over a micro-batch.

        x: [sum(n_i), d] concatenated hidden states of all items.
        Returns the output hidden states (same layout) and, for every item, the
        *new* (key, value) pairs per local layer (only the n_i new positions).
        """
        d, H, D = self.cfg.n_embd, self.n_head, self.head_dim
        new_kv: list[list[KV]] = [[] for _ in segments]
        for li, b in enumerate(self.blocks):
            residual = x
            h = F.layer_norm(x, (d,), b.ln_1_weight, b.ln_1_bias, self.eps)
            qkv = self._conv1d(h, b.attn_c_attn_weight, b.attn_c_attn_bias)
            q_all, k_all, v_all = qkv.split(d, dim=1)
            outs, offset = [], 0
            for si, seg in enumerate(segments):
                n = seg.n_tokens
                q = q_all[offset:offset + n].view(1, n, H, D).transpose(1, 2)
                k = k_all[offset:offset + n].view(1, n, H, D).transpose(1, 2)
                v = v_all[offset:offset + n].view(1, n, H, D).transpose(1, 2)
                new_kv[si].append((k, v))
                past = seg.past[li] if seg.past is not None else None
                if past is not None:
                    k = torch.cat((past[0], k), dim=-2)
                    v = torch.cat((past[1], v), dim=-2)
                a = self._attention(q, k, v)  # [1,H,n,D]
                outs.append(a.transpose(1, 2).reshape(n, d))
                offset += n
            a = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
            a = self._conv1d(a, b.attn_c_proj_weight, b.attn_c_proj_bias)
            x = a + residual
            residual = x
            h = F.layer_norm(x, (d,), b.ln_2_weight, b.ln_2_bias, self.eps)
            h = gelu_new(self._conv1d(h, b.mlp_c_fc_weight, b.mlp_c_fc_bias))
            h = self._conv1d(h, b.mlp_c_proj_weight, b.mlp_c_proj_bias)
            x = residual + h
        return x, new_kv

    def head(self, x: torch.Tensor) -> torch.Tensor:
        """Final layer norm + LM head. x: [m, d] -> logits [m, vocab]."""
        h = F.layer_norm(x, (self.cfg.n_embd,), self.ln_f_weight, self.ln_f_bias, self.eps)
        return F.linear(h, self.lm_head)


def load_stages(model_dir=None, num_stages: int = 2, split: str = "6") -> list[StageModule]:
    """Load every stage into the current process (used by tests and in-process experiments)."""
    model_dir = resolve_model_dir() if model_dir is None else model_dir
    cfg = GPT2Config.from_json(model_dir / "config.json")
    return [StageModule.from_pretrained(model_dir, s, e) for s, e in stage_ranges(cfg, num_stages, split)]


class PipelineStage:
    """A stage plus its per-request KV caches: the `forward(hidden_states, request_id, step)` API.

    For stage 0 `hidden_states` is a 1-D LongTensor of token ids (the embedding
    lives in stage 0); for later stages it is a [n, d] float tensor. The last
    stage returns logits for the final position; other stages return hidden
    states for the new positions.
    """

    def __init__(self, module: StageModule, max_cached_requests: int = 128, cache_ttl_s: float = 120.0):
        self.module = module
        self.cache = KVCacheManager(module.num_local_layers, max_cached_requests, cache_ttl_s)

    def forward(self, hidden_states: torch.Tensor, request_id: str, step: int,
                reset: bool = False, use_cache: bool = True) -> torch.Tensor:
        return self.forward_batch([(hidden_states, request_id, step, reset, use_cache)])[0]

    def forward_batch(self, items) -> list[torch.Tensor]:
        """items: list of (hidden_states, request_id, step, reset, use_cache)."""
        caches: list[RequestCache | None] = []
        segments, xs = [], []
        for hs, rid, step, reset, use_cache in items:
            c = None
            if use_cache:
                c = self.cache.get(rid)
                if reset or c is None:
                    c = self.cache.create(rid, attempt=0)
                elif step != c.last_step + 1:
                    raise ValueError(f"out-of-order step {step} for {rid} (expected {c.last_step + 1})")
            caches.append(c)
            n = hs.shape[0]
            past_len = c.length if c is not None else 0
            if self.module.has_embeddings:
                xs.append(self.module.embed(hs, torch.arange(past_len, past_len + n)))
            else:
                xs.append(hs)
            segments.append(SegmentInput(n, c.layers if (c is not None and c.length) else None))
        x = xs[0] if len(xs) == 1 else torch.cat(xs, dim=0)
        y, new_kv = self.module.forward_segments(x, segments)
        outs, offset = [], 0
        for (hs, rid, step, _, _), seg, c, kvs in zip(items, segments, caches, new_kv):
            rows = y[offset:offset + seg.n_tokens]
            offset += seg.n_tokens
            if c is not None:
                for li, (k, v) in enumerate(kvs):
                    KVCacheManager.append(c, li, k, v)
                c.length += seg.n_tokens
                c.last_step = step
            outs.append(self.module.head(rows[-1:])[0] if self.module.has_head else rows)
        return outs
