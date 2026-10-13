"""O1: single-process greedy-decoding baseline and reference outputs.

    python baseline.py              # (re)generate reference_outputs.json
    python baseline.py --check      # re-run and verify the token sequences are byte-identical

The baseline uses the stock Hugging Face GPT-2 model (eager attention, fp32,
KV cache) in one process. Every later configuration of the distributed system
is checked token-for-token against the file this script writes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time

import torch

from pllm.config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED, DEFAULT_THREADS, PROMPTS_FILE, REFERENCE_FILE
from pllm.model_loader import resolve_model_dir, set_determinism, snapshot_commit


def load_model(model_dir=None):
    from transformers import AutoTokenizer, GPT2LMHeadModel

    model_dir = resolve_model_dir() if model_dir is None else model_dir
    model = GPT2LMHeadModel.from_pretrained(model_dir, attn_implementation="eager").float()
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    return model, tok, model_dir


@torch.inference_mode()
def greedy_generate(model, prompt_ids: list[int], max_new_tokens: int, use_cache: bool = True):
    """Plain greedy loop. Returns (generated token ids, per-token latency in seconds).

    Token 0's latency is the prefill (time to first token); the rest are decode steps.
    Generation always runs for exactly `max_new_tokens` (no early stop at EOS).
    """
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    past, out, lat = None, [], []
    inp = ids
    for _ in range(max_new_tokens):
        t0 = time.perf_counter()
        if use_cache:
            o = model(input_ids=inp, past_key_values=past, use_cache=True)
            past = o.past_key_values
        else:
            o = model(input_ids=torch.cat([ids, torch.tensor([out], dtype=torch.long)], dim=1)
                      if out else ids, use_cache=False)
        tok = int(torch.argmax(o.logits[0, -1]))
        lat.append(time.perf_counter() - t0)
        out.append(tok)
        inp = torch.tensor([[tok]], dtype=torch.long)
    return out, lat


def tokens_digest(entries) -> str:
    h = hashlib.sha256()
    for e in entries:
        h.update(json.dumps([e["prompt_token_ids"], e["output_token_ids"]]).encode())
    return h.hexdigest()


def run(prompts: list[str], max_new_tokens: int, threads: int):
    set_determinism(DEFAULT_SEED, threads)
    model, tok, model_dir = load_model()
    # Warm-up so the first prompt's latency is not dominated by lazy initialisation.
    greedy_generate(model, tok.encode("warm up"), 2)
    entries = []
    t_all = time.perf_counter()
    for i, p in enumerate(prompts):
        ids = tok.encode(p)
        t0 = time.perf_counter()
        out, lat = greedy_generate(model, ids, max_new_tokens)
        total = time.perf_counter() - t0
        entries.append({
            "id": i, "prompt": p, "prompt_token_ids": ids, "prompt_len": len(ids),
            "output_token_ids": out, "output_text": tok.decode(out),
            "per_token_latency_ms": [round(x * 1e3, 4) for x in lat],
            "total_ms": round(total * 1e3, 3),
        })
        print(f"[{i:2d}] len={len(ids):3d} total={total * 1e3:7.1f}ms  {p[:40]!r} -> {tok.decode(out)[:40]!r}")
    total_runtime = time.perf_counter() - t_all
    import transformers

    meta = {
        "model": "gpt2", "model_commit": snapshot_commit(model_dir), "decoding": "greedy",
        "attn_implementation": "eager", "dtype": "float32", "max_new_tokens": max_new_tokens,
        "seed": DEFAULT_SEED, "torch_threads": threads, "deterministic_algorithms": True,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "total_runtime_s": round(total_runtime, 4),
        "total_generated_tokens": max_new_tokens * len(prompts),
    }
    meta["tokens_sha256"] = tokens_digest(entries)
    return {"meta": meta, "prompts": entries}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=str(PROMPTS_FILE))
    ap.add_argument("--out", default=str(REFERENCE_FILE))
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--check", action="store_true",
                    help="re-run and compare token sequences with the existing reference file")
    a = ap.parse_args(argv)
    prompts = json.load(open(a.prompts))
    if a.check:
        ref = json.load(open(a.out))
        result = run([e["prompt"] for e in ref["prompts"]], ref["meta"]["max_new_tokens"], a.threads)
        same = tokens_digest(result["prompts"]) == ref["meta"]["tokens_sha256"]
        mism = [e["id"] for e, r in zip(result["prompts"], ref["prompts"])
                if e["output_token_ids"] != r["output_token_ids"]]
        print(f"re-run digest {'MATCHES' if same else 'DIFFERS'}; mismatching prompts: {mism}")
        return 0 if same and not mism else 1
    result = run(prompts, a.max_new_tokens, a.threads)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    m = result["meta"]
    print(f"wrote {a.out}: {len(prompts)} prompts x {a.max_new_tokens} tokens in {m['total_runtime_s']:.2f}s "
          f"({m['total_generated_tokens'] / m['total_runtime_s']:.1f} tok/s), digest {m['tokens_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
