"""O4 (+ the per-item rules used by O5-O7): per-request KV caches, ordering, idempotency, epochs."""

import time

import pytest
import torch

from pllm.proto import pipeline_pb2 as pb
from pllm.stage_executor import StageExecutor

EPOCHS = [1, 1]


def item(rid, step, tokens=(), attempt=0, reset=False, use_cache=True, release=False):
    return pb.StageItem(request_id=rid, step=step, attempt=attempt, reset=reset, use_cache=use_cache,
                        release_after=release, token_ids=list(tokens))


class Pipe:
    """Two StageExecutors wired together in-process (no gRPC)."""

    def __init__(self, stages, **kw):
        self.s0 = StageExecutor(stages[0], 0, epoch=1, **kw)
        self.s1 = StageExecutor(stages[1], 1, epoch=1, **kw)

    def run(self, items, epochs=EPOCHS):
        o0 = self.s0.process(epochs, items, [None] * len(items))
        fwd = [(it, o) for it, o in zip(items, o0) if o.status == pb.OK]
        o1 = self.s1.process(epochs, [it for it, _ in fwd], [o.hidden for _, o in fwd])
        res = {o.request_id: o for o in o0}
        for o in o1:
            res[o.request_id] = o
        return [res[it.request_id] for it in items]

    def generate(self, rid, prompt, n, use_cache=True):
        out = []
        for step in range(n):
            full = step == 0 or not use_cache
            toks = (prompt + out) if full else [out[-1]]
            (o,) = self.run([item(rid, step, toks, reset=use_cache and step == 0, use_cache=use_cache)])
            assert o.status == pb.OK, pb.ItemStatus.Name(o.status)
            out.append(o.token_id)
        return out


@pytest.fixture
def pipe(stages):
    return Pipe(stages)


def test_cache_on_equals_cache_off_and_reference(stages, reference):
    n = reference["meta"]["max_new_tokens"]
    for e in reference["prompts"][::3]:
        on = Pipe(stages).generate("a", e["prompt_token_ids"], n, use_cache=True)
        off = Pipe(stages).generate("a", e["prompt_token_ids"], n, use_cache=False)
        assert on == off == e["output_token_ids"]


def test_cache_reduces_decode_latency(stages, reference):
    e = max(reference["prompts"], key=lambda e: e["prompt_len"])
    timings = {}
    for use_cache in (True, False):
        p = Pipe(stages)
        t0 = time.perf_counter()
        p.generate("x", e["prompt_token_ids"], 16, use_cache=use_cache)
        timings[use_cache] = time.perf_counter() - t0
    print(f"\n16 tokens, prompt {e['prompt_len']}: cache on {timings[True] * 1e3:.0f} ms, "
          f"off {timings[False] * 1e3:.0f} ms, speed-up {timings[False] / timings[True]:.2f}x")
    assert timings[True] < timings[False]


def test_interleaved_requests_never_mix(stages, reference):
    a, b = reference["prompts"][1], reference["prompts"][-1]
    n = 12
    p = Pipe(stages)
    out = {"A": [], "B": []}
    prompts = {"A": a["prompt_token_ids"], "B": b["prompt_token_ids"]}
    for step in range(n):
        # alternate the order and mix single-item and two-item batches
        order = ["A", "B"] if step % 2 == 0 else ["B", "A"]
        its = [item(r, step, prompts[r] if step == 0 else [out[r][-1]], reset=step == 0) for r in order]
        batches = [its] if step % 3 == 0 else [[x] for x in its]
        for batch in batches:
            for o in p.run(batch):
                assert o.status == pb.OK
                out[o.request_id].append(o.token_id)
    assert out["A"] == a["output_token_ids"][:n]
    assert out["B"] == b["output_token_ids"][:n]
    assert len(p.s0.cache) == len(p.s1.cache) == 2


def test_mixed_prefill_and_decode_batch_matches_reference(stages, reference):
    """Selective batching: requests of different lengths/phases share one micro-batch."""
    entries = reference["prompts"][:8]
    n = 10
    p = Pipe(stages)
    out = {e["id"]: [] for e in entries}
    started = []
    for step_round in range(n + len(entries)):
        # admit one new request per round, so every batch mixes one prefill with decodes
        if len(started) < len(entries):
            started.append(entries[len(started)])
        its = []
        for e in started:
            k = len(out[e["id"]])
            if k >= n:
                continue
            toks = e["prompt_token_ids"] if k == 0 else [out[e["id"]][-1]]
            its.append(item(str(e["id"]), k, toks, reset=k == 0))
        if not its:
            break
        for o in p.run(its):
            assert o.status == pb.OK
            out[int(o.request_id)].append(o.token_id)
    for e in entries:
        assert out[e["id"]] == e["output_token_ids"][:n], f"prompt {e['id']}"


def test_out_of_order_and_stale_steps_are_rejected(pipe, reference):
    ids = reference["prompts"][2]["prompt_token_ids"]
    (o,) = pipe.run([item("r", 0, ids, reset=True)])
    t0 = o.token_id
    length = pipe.s0.cache.get("r").length
    (o,) = pipe.run([item("r", 2, [t0])])                 # skips step 1
    assert o.status == pb.OUT_OF_ORDER
    assert pipe.s0.cache.get("r").length == length       # nothing appended
    (o,) = pipe.run([item("r", 1, [t0])])
    assert o.status == pb.OK
    (o2,) = pipe.run([item("r", 2, [o.token_id])])
    assert o2.status == pb.OK
    (o,) = pipe.run([item("r", 1, [t0])])                 # already applied, older than last step
    assert o.status == pb.STALE_STEP


def test_duplicate_step_is_idempotent(pipe, reference):
    e = reference["prompts"][4]
    (o,) = pipe.run([item("r", 0, e["prompt_token_ids"], reset=True)])
    (a,) = pipe.run([item("r", 1, [o.token_id])])
    len0, len1 = pipe.s0.cache.get("r").length, pipe.s1.cache.get("r").length
    (b,) = pipe.run([item("r", 1, [o.token_id])])         # retry / duplicate delivery of step 1
    assert b.status == pb.OK and b.replayed and b.token_id == a.token_id
    assert (pipe.s0.cache.get("r").length, pipe.s1.cache.get("r").length) == (len0, len1)
    # generation continues correctly after the duplicate
    (c,) = pipe.run([item("r", 2, [a.token_id])])
    assert [o.token_id, a.token_id, c.token_id] == e["output_token_ids"][:3]


def test_partial_duplicate_only_on_stage1(pipe, reference):
    """Stage 0 applied the step, stage 1 did not (e.g. lost message): the retry recomputes only stage 1."""
    e = reference["prompts"][5]
    (o,) = pipe.run([item("r", 0, e["prompt_token_ids"], reset=True)])
    it = item("r", 1, [o.token_id])
    pipe.s0.process(EPOCHS, [it], [None])                  # only stage 0 sees step 1
    (a,) = pipe.run([it])                                  # retry through both stages
    assert a.status == pb.OK and a.token_id == e["output_token_ids"][1]


def test_replay_attempt_rebuilds_and_fences_old_attempt(pipe, reference):
    e = reference["prompts"][6]
    ids, ref = e["prompt_token_ids"], e["output_token_ids"]
    (o,) = pipe.run([item("r", 0, ids, reset=True)])
    (o,) = pipe.run([item("r", 1, [o.token_id])])
    # simulate a crash of stage 1: its cache is gone
    pipe.s1.cache.clear()
    (x,) = pipe.run([item("r", 2, [o.token_id])])
    assert x.status == pb.UNKNOWN_REQUEST
    # replay prompt + generated tokens under attempt 1
    (y,) = pipe.run([item("r", 2, ids + ref[:2], attempt=1, reset=True)])
    assert y.status == pb.OK and y.token_id == ref[2]
    (z,) = pipe.run([item("r", 3, [ref[2]], attempt=0)])   # late message from the old attempt
    assert z.status == pb.STALE_ATTEMPT
    (w,) = pipe.run([item("r", 3, [ref[2]], attempt=1)])
    assert w.status == pb.OK and w.token_id == ref[3]


def test_epoch_fencing(pipe, reference):
    ids = reference["prompts"][0]["prompt_token_ids"]
    (o,) = pipe.run([item("r", 0, ids, reset=True)], epochs=[0, 1])
    assert o.status == pb.STALE_EPOCH and len(pipe.s0.cache) == 0
    (o,) = pipe.run([item("r", 0, ids, reset=True)], epochs=[1, 0])
    assert o.status == pb.STALE_EPOCH and len(pipe.s1.cache) == 0
    (o,) = pipe.run([item("r", 0, ids, reset=True)], epochs=[2, 1])
    assert o.status == pb.FUTURE_EPOCH


def test_unknown_request_and_release(pipe, reference):
    ids = reference["prompts"][0]["prompt_token_ids"]
    (o,) = pipe.run([item("r", 1, [5])])
    assert o.status == pb.UNKNOWN_REQUEST
    (o,) = pipe.run([item("r", 0, ids, reset=True)])
    assert len(pipe.s0.cache) == 1 and pipe.s0.cache.total_bytes() > 0
    pipe.s0.release(["r"])
    pipe.s1.release(["r"])
    assert len(pipe.s0.cache) == len(pipe.s1.cache) == 0
    (o,) = pipe.run([item("r", 1, [o.token_id])])
    assert o.status == pb.REQUEST_CLOSED


def test_release_after_last_step_frees_cache_and_final_retry_is_idempotent(pipe, reference):
    ids = reference["prompts"][0]["prompt_token_ids"]
    (o,) = pipe.run([item("r", 0, ids, reset=True)])
    (a,) = pipe.run([item("r", 1, [o.token_id], release=True)])
    assert len(pipe.s0.cache) == len(pipe.s1.cache) == 0
    (b,) = pipe.run([item("r", 1, [o.token_id], release=True)])  # retry of the final step
    assert b.status == pb.OK and b.replayed and b.token_id == a.token_id


def test_cache_capacity_and_ttl(stages, reference):
    p = Pipe(stages, max_cached_requests=2, cache_ttl_s=3600)
    ids = reference["prompts"][0]["prompt_token_ids"]
    for rid in ("a", "b"):
        assert p.run([item(rid, 0, ids, reset=True)])[0].status == pb.OK
    assert p.run([item("c", 0, ids, reset=True)])[0].status == pb.CACHE_FULL
    p.s0.cache.ttl_s = p.s1.cache.ttl_s = 0.0
    time.sleep(0.01)
    assert p.run([item("c", 0, ids, reset=True)])[0].status == pb.OK   # idle caches were evicted
    assert len(p.s0.cache) == 1
