"""The stateful part of a worker: validates and executes one micro-batch on one stage.

This module has no gRPC dependency so its ordering / idempotency / epoch rules
can be unit-tested directly. `pllm.worker` runs `StageExecutor.process` on a
single dedicated compute thread, so all cache mutation is serialised.

Validation rules for an item (request_id, attempt, step), in order:

1. Epoch: the message's epoch for this stage must equal the worker's epoch.
   Lower  -> STALE_EPOCH  (a message from before a crash/restart)
   Higher -> FUTURE_EPOCH (meant for a newer incarnation of this stage)
2. use_cache = False: stateless full recompute, always executed.
3. reset = True (prefill or recovery replay):
   - no cache, or cache.attempt < attempt  -> rebuild from scratch
   - same attempt and step == last_step    -> duplicate: replay stored output
   - same attempt and step  < last_step    -> STALE_STEP
   - cache.attempt > attempt               -> STALE_ATTEMPT
4. decode (reset = False): requires a cache with the same attempt
   - no cache                              -> UNKNOWN_REQUEST (or REQUEST_CLOSED if tombstoned)
   - step == last_step + 1                 -> execute
   - step == last_step                     -> duplicate: replay stored output (idempotent retry)
   - step  < last_step                     -> STALE_STEP
   - step  > last_step + 1                 -> OUT_OF_ORDER
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from pllm.kv_cache import CacheFullError, KVCacheManager, RequestCache
from pllm.partition import SegmentInput, StageModule
from pllm.proto import pipeline_pb2 as pb


@dataclass
class ExecItem:
    """A validated item that will be computed."""
    index: int                    # position in the incoming batch
    request_id: str
    step: int
    attempt: int
    release_after: bool
    cache: RequestCache | None    # None when the KV cache is disabled for this item
    tokens: torch.Tensor | None   # stage 0 input
    hidden: torch.Tensor | None   # stage >0 input
    n_tokens: int


@dataclass
class ItemOutcome:
    request_id: str
    step: int
    attempt: int
    status: int = pb.OK
    token_id: int = -1
    hidden: torch.Tensor | None = None
    replayed: bool = False
    detail: str = ""


@dataclass
class ExecutorStats:
    batches: int = 0
    items: int = 0
    tokens_processed: int = 0
    busy_s: float = 0.0
    rejected: dict = field(default_factory=dict)
    replayed: int = 0

    def reject(self, status: int) -> None:
        name = pb.ItemStatus.Name(status)
        self.rejected[name] = self.rejected.get(name, 0) + 1


class StageExecutor:
    def __init__(self, module: StageModule, stage: int, epoch: int,
                 max_cached_requests: int = 128, cache_ttl_s: float = 120.0):
        self.module = module
        self.stage = stage
        self.epoch = epoch
        self.cache = KVCacheManager(module.num_local_layers, max_cached_requests, cache_ttl_s)
        self.stats = ExecutorStats()

    # ------------------------------------------------------------------
    def check_epoch(self, stage_epochs) -> int:
        if self.stage >= len(stage_epochs):
            return pb.STALE_EPOCH
        e = stage_epochs[self.stage]
        if e < self.epoch:
            return pb.STALE_EPOCH
        if e > self.epoch:
            return pb.FUTURE_EPOCH
        return pb.OK

    def process(self, stage_epochs, items, decoded_hidden: list[torch.Tensor | None]) -> list[ItemOutcome]:
        """Validate and execute a micro-batch. `decoded_hidden[i]` is item i's input
        activations (already deserialised) for stages > 0."""
        t0 = time.perf_counter()
        outcomes = [ItemOutcome(it.request_id, it.step, it.attempt) for it in items]
        epoch_status = self.check_epoch(stage_epochs)
        if epoch_status != pb.OK:
            for o in outcomes:
                o.status = epoch_status
                o.detail = f"stage {self.stage} epoch {self.epoch}, message epoch {list(stage_epochs)}"
                self.stats.reject(epoch_status)
            return outcomes

        self.cache.evict_expired()
        to_run: list[ExecItem] = []
        seen: set[str] = set()
        for i, it in enumerate(items):
            o = outcomes[i]
            if it.request_id in seen:
                o.status, o.detail = pb.STALE_STEP, "request appears twice in one batch"
                self.stats.reject(o.status)
                continue
            seen.add(it.request_id)
            status, cache, replay = self._validate(it)
            if replay is not None:
                o.replayed = True
                self._set_output(o, replay)
                self.stats.replayed += 1
                continue
            if status != pb.OK:
                o.status = status
                self.stats.reject(status)
                continue
            if self.module.has_embeddings:
                tokens = torch.tensor(list(it.token_ids), dtype=torch.long)
                hidden, n = None, tokens.numel()
            else:
                tokens, hidden = None, decoded_hidden[i]
                n = hidden.shape[0]
            if n == 0:
                o.status, o.detail = pb.INTERNAL_ERROR, "empty input"
                continue
            to_run.append(ExecItem(i, it.request_id, it.step, it.attempt, it.release_after,
                                   cache, tokens, hidden, n))

        if to_run:
            try:
                self._execute(to_run, outcomes)
            except Exception as e:  # never leave a half-updated cache behind
                for ex in to_run:
                    if ex.cache is not None:
                        self.cache.release(ex.request_id, reason="error")
                    outcomes[ex.index].status = pb.INTERNAL_ERROR
                    outcomes[ex.index].detail = repr(e)
        self.stats.batches += 1
        self.stats.items += len(items)
        self.stats.busy_s += time.perf_counter() - t0
        return outcomes

    # ------------------------------------------------------------------
    def _validate(self, it) -> tuple[int, RequestCache | None, object]:
        """Returns (status, cache, replay_output). replay_output is not None for duplicates."""
        if not it.use_cache:
            return pb.OK, None, None
        c = self.cache.get(it.request_id)
        if c is None:
            tomb = self.cache.tombstone(it.request_id)
            if tomb is not None:
                if tomb.attempt == it.attempt and tomb.last_step == it.step and tomb.last_output is not None:
                    return pb.OK, None, tomb.last_output  # retry of the final step
                if not it.reset or tomb.reason in ("completed", "released"):
                    return pb.REQUEST_CLOSED, None, None
                # evicted / errored caches may be rebuilt by a replay
            if not it.reset:
                return pb.UNKNOWN_REQUEST, None, None
            try:
                return pb.OK, self.cache.create(it.request_id, it.attempt), None
            except CacheFullError:
                return pb.CACHE_FULL, None, None

        if it.attempt < c.attempt:
            return pb.STALE_ATTEMPT, None, None
        if it.attempt > c.attempt:
            if not it.reset:
                # The coordinator moved on to a newer replay attempt this worker never saw.
                return pb.UNKNOWN_REQUEST, None, None
            return pb.OK, self.cache.create(it.request_id, it.attempt), None
        # same attempt
        if it.step == c.last_step:
            return pb.OK, c, c.last_output
        if it.step < c.last_step:
            return pb.STALE_STEP, None, None
        if it.reset:
            # A reset for a later step within the same attempt: the coordinator only
            # does this if it lost track of state, so rebuilding is the safe answer.
            return pb.OK, self.cache.create(it.request_id, it.attempt), None
        if it.step != c.last_step + 1:
            return pb.OUT_OF_ORDER, None, None
        return pb.OK, c, None

    def _set_output(self, o: ItemOutcome, out) -> None:
        if self.module.has_head:
            o.token_id = int(out)
        else:
            o.hidden = out

    def _execute(self, to_run: list[ExecItem], outcomes: list[ItemOutcome]) -> None:
        m = self.module
        xs, segments = [], []
        for ex in to_run:
            past_len = ex.cache.length if ex.cache is not None else 0
            if m.has_embeddings:
                xs.append(m.embed(ex.tokens, torch.arange(past_len, past_len + ex.n_tokens)))
            else:
                xs.append(ex.hidden)
            past = ex.cache.layers if (ex.cache is not None and ex.cache.length) else None
            segments.append(SegmentInput(ex.n_tokens, past))
        x = xs[0] if len(xs) == 1 else torch.cat(xs, dim=0)
        y, new_kv = m.forward_segments(x, segments)

        if m.has_head:
            last_rows, offset = [], 0
            for ex in to_run:
                offset += ex.n_tokens
                last_rows.append(offset - 1)
            logits = m.head(y[last_rows])
            tokens = torch.argmax(logits, dim=-1).tolist()

        offset = 0
        for j, ex in enumerate(to_run):
            o = outcomes[ex.index]
            rows = y[offset:offset + ex.n_tokens]
            offset += ex.n_tokens
            out = int(tokens[j]) if m.has_head else rows.clone()
            self._set_output(o, out)
            self.stats.tokens_processed += ex.n_tokens
            if ex.cache is not None:
                for li, (k, v) in enumerate(new_kv[j]):
                    KVCacheManager.append(ex.cache, li, k, v)
                ex.cache.length += ex.n_tokens
                ex.cache.last_step = ex.step
                ex.cache.last_output = out
                if ex.release_after:
                    self.cache.release(ex.request_id, reason="completed")

    def release(self, request_ids) -> int:
        return sum(self.cache.release(r) for r in request_ids)
