"""End-to-end tests over real processes and gRPC (O3, O5, O7).

Each cluster = 1 coordinator process + 2 worker processes on localhost.
Run only these with:  pytest -m slow
"""

import asyncio
import json
import os
import signal
import time

import grpc
import pytest

from pllm.client import channel, generate
from pllm.cluster import LocalCluster
from pllm.proto import pipeline_pb2 as pb
from pllm.proto import pipeline_pb2_grpc as pb_grpc

pytestmark = pytest.mark.slow


async def run_requests(addr, entries, n, concurrency=None, **kw):
    async with channel(addr) as ch:
        stub = pb_grpc.CoordinatorServiceStub(ch)
        sem = asyncio.Semaphore(concurrency or len(entries))

        async def one(i, e):
            async with sem:
                return await generate(stub, prompt_ids=e["prompt_token_ids"], max_new_tokens=n,
                                      request_id=f"t{i}", **kw)

        return await asyncio.wait_for(asyncio.gather(*[one(i, e) for i, e in enumerate(entries)]), 300)


def assert_matches(results, entries, n):
    for r, e in zip(results, entries):
        assert r.status == "COMPLETED", (r.request_id, r.status, r.detail)
        assert r.tokens == e["output_token_ids"][:n], f"prompt {e['id']} mismatch"
        assert r.final_token_ids == r.tokens


def wait_until_healthy(cluster, timeout=60.0):
    """Block until no stage is recovering and every worker answers a health check."""
    async def check():
        s = await cluster.astats(workers=False)
        if s["recovering"] or min(s["epochs"]) <= 0:
            return False
        for stage in range(len(s["epochs"])):
            async with grpc.aio.insecure_channel(cluster.worker_addr(stage)) as ch:
                try:
                    h = await pb_grpc.StageServiceStub(ch).Health(pb.HealthRequest(), timeout=1)
                except grpc.aio.AioRpcError:
                    return False
                if h.epoch != s["epochs"][stage]:
                    return False
        return True

    deadline = time.time() + timeout
    while time.time() < deadline:
        if asyncio.run(check()):
            return
        time.sleep(0.1)
    raise AssertionError("cluster did not become healthy")


@pytest.fixture(scope="module")
def cluster():
    with LocalCluster(port=50250, log_name="test_cluster", extra=["--max-batch-size", "8"]) as c:
        yield c


def test_sequential_requests_match_reference(cluster, reference):
    n = reference["meta"]["max_new_tokens"]
    entries = reference["prompts"]
    results = asyncio.run(run_requests(cluster.addr, entries, n, concurrency=1))
    assert_matches(results, entries, n)
    s = cluster.stats()
    assert s["counters"].get("activation_bytes", 0) > 0


def test_50_concurrent_requests_match_reference(cluster, reference):
    n = reference["meta"]["max_new_tokens"]
    entries = [reference["prompts"][i % len(reference["prompts"])] for i in range(50)]
    cluster.stats(reset=True)
    results = asyncio.run(run_requests(cluster.addr, entries, n))
    assert_matches(results, entries, n)
    s = cluster.stats()
    assert s["max_admission_depth"] <= s["admission_capacity"]
    assert s["max_active_seen"] <= s["active_capacity"]
    for w in s["workers"]:
        assert w["max_queue_depth"] <= w["queue_size"]
        assert w["cached_requests"] == 0  # every cache freed on completion
    assert s["mean_batch_size"] > 1.5     # micro-batching actually happened


def test_cancellation_frees_worker_caches(cluster, reference):
    e = reference["prompts"][-1]
    res = asyncio.run(run_requests(cluster.addr, [e], 30, cancel_after=3))[0]
    assert res.status == "CANCELLED" and len(res.tokens) == 3
    deadline = time.time() + 5
    while time.time() < deadline:
        if all(w["cached_requests"] == 0 for w in cluster.stats()["workers"]):
            break
        time.sleep(0.1)
    assert all(w["cached_requests"] == 0 for w in cluster.stats()["workers"])


def test_request_timeout_is_explicit(cluster, reference):
    e = reference["prompts"][-1]
    res = asyncio.run(run_requests(cluster.addr, [e], 400, timeout_s=0.05))[0]
    assert res.status == "TIMEOUT"


@pytest.mark.parametrize("stage", [1, 0])
def test_worker_kill_mid_generation_recovers_exactly(cluster, reference, stage):
    n = reference["meta"]["max_new_tokens"]
    entries = reference["prompts"][:12]
    wait_until_healthy(cluster)
    before = cluster.stats(workers=False)
    old_epoch = before["epochs"][stage]
    pid = cluster.worker_pid(stage, before)

    async def scenario():
        task = asyncio.create_task(run_requests(cluster.addr, entries, n))
        # wait until generation is under way, then crash-stop the worker
        while True:
            s = await cluster.astats(workers=False)
            if s["counters"].get("tokens_generated", 0) - before["counters"].get("tokens_generated", 0) >= 40:
                break
            await asyncio.sleep(0.01)
        os.kill(pid, signal.SIGKILL)
        return await task

    results = asyncio.run(scenario())
    assert_matches(results, entries, n)   # no duplicate, missing or wrong tokens
    after = cluster.stats(workers=False)
    assert after["epochs"][stage] > old_epoch
    ev = [e for e in after["recovery_events"] if e["stage"] == stage][-1]
    assert ev["result"] == "recovered"
    print(f"\nstage {stage} recovery: detection {ev['detection_ms']:.0f} ms, restart {ev['restart_ms']:.0f} ms, "
          f"replay {ev['replay_ms']:.0f} ms, total {ev['recovery_ms']:.0f} ms")


def test_stale_epoch_message_is_rejected(cluster, reference):
    """Inject a message carrying a pre-crash epoch directly into a restarted worker."""
    wait_until_healthy(cluster)
    s = cluster.stats(workers=False)
    epochs = s["epochs"]
    stale = [max(0, e - 1) for e in epochs]
    e = reference["prompts"][0]

    async def inject():
        async with grpc.aio.insecure_channel(cluster.worker_addr(0)) as ch:
            stub = pb_grpc.StageServiceStub(ch)
            msg = pb.ForwardRequest(stage_epochs=stale, batch_id=999, items=[pb.StageItem(
                request_id="stale", step=0, reset=True, use_cache=True, token_ids=e["prompt_token_ids"])])
            return await stub.Forward(msg, timeout=5)

    resp = asyncio.run(inject())
    assert resp.results[0].status in (pb.STALE_EPOCH,), pb.ItemStatus.Name(resp.results[0].status)


def test_duplicate_dispatch_emits_no_duplicate_tokens(reference):
    n = reference["meta"]["max_new_tokens"]
    entries = reference["prompts"][:10]
    with LocalCluster(port=50270, log_name="test_dup", extra=["--duplicate-dispatch"]) as c:
        results = asyncio.run(run_requests(c.addr, entries, n))
        s = c.stats()
    assert_matches(results, entries, n)
    ctr = s["counters"]
    assert ctr.get("duplicates_sent", 0) > 0
    assert ctr.get("discarded_duplicate_results", 0) > 0
    print("\n" + json.dumps({k: v for k, v in ctr.items() if "dup" in k or "replay" in k}))


def test_overload_is_rejected_explicitly_never_hangs(reference):
    n = 16
    entries = [reference["prompts"][i % 24] for i in range(40)]
    with LocalCluster(port=50290, log_name="test_overload",
                      extra=["--admission-queue", "4", "--max-active", "4"]) as c:
        results = asyncio.run(run_requests(c.addr, entries, n))
        s = c.stats()
    statuses = [r.status for r in results]
    assert set(statuses) <= {"COMPLETED", "OVERLOADED"}
    assert statuses.count("OVERLOADED") > 0 and statuses.count("COMPLETED") >= 4
    for r, e in zip(results, entries):
        if r.status == "COMPLETED":
            assert r.tokens == e["output_token_ids"][:n]
        else:
            assert r.tokens == [] and "admission queue full" in r.detail
    assert s["max_admission_depth"] <= 4 and s["max_active_seen"] <= 4


def test_kv_cache_off_cluster_matches_reference(reference):
    n = 12
    entries = reference["prompts"][::4]
    with LocalCluster(port=50310, log_name="test_nocache", extra=["--no-kv-cache"]) as c:
        results = asyncio.run(run_requests(c.addr, entries, n))
    assert_matches(results, entries, n)
