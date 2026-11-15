"""Coordinator crash recovery end to end (E10): request log + worker adoption + Resume + fencing.

Run only these with:  pytest -m slow tests/test_coordinator_recovery.py -s
"""

import asyncio
import os
import signal
import time

import pytest

from pllm.client import channel, generate, get_stats
from pllm.cluster import LocalCluster
from pllm.proto import pipeline_pb2 as pb
from pllm.proto import pipeline_pb2_grpc as pb_grpc

pytestmark = pytest.mark.slow


def kill_mid_generation(c, entries, n, after_tokens=60):
    """Stream `entries` with reconnecting clients, SIGKILL the coordinator once `after_tokens` tokens have
    been produced, restart it, and return (results, stats before the kill, stats after)."""
    async def scenario():
        before = await get_stats(c.addr, workers=False)
        async with channel(c.addr) as ch:
            stub = pb_grpc.CoordinatorServiceStub(ch)
            tasks = [asyncio.create_task(generate(stub, prompt_ids=e["prompt_token_ids"], max_new_tokens=n,
                                                  request_id=f"k{i}-{time.monotonic_ns()}", reconnect_s=60))
                     for i, e in enumerate(entries)]
            while (await get_stats(c.addr, workers=False))["counters"].get("tokens_generated", 0) < after_tokens:
                await asyncio.sleep(0.01)
            await asyncio.to_thread(c.kill)
            await asyncio.to_thread(c.restart)
            results = await asyncio.wait_for(asyncio.gather(*tasks), 120)
        return results, before, await get_stats(c.addr, workers=True)

    return asyncio.run(scenario())


def assert_exact(results, entries, n):
    for r, e in zip(results, entries):
        assert r.status == "COMPLETED", (r.request_id, r.status, r.detail)
        assert r.tokens == e["output_token_ids"][:n], "duplicate, missing or wrong tokens"
        assert r.reconnects == 1


@pytest.mark.parametrize("detach", [True, False], ids=["adopt-workers", "respawn-workers"])
def test_coordinator_kill_resumes_every_request_exactly(reference, tmp_path, detach):
    n = reference["meta"]["max_new_tokens"]
    entries = reference["prompts"][:10]
    extra = ["--request-log", str(tmp_path / "log.db")] + (["--detach-workers"] if detach else [])
    with LocalCluster(port=50700, log_name=f"test_coord_kill_{detach}", extra=extra) as c:
        results, before, after = kill_mid_generation(c, entries, n)
    assert_exact(results, entries, n)
    st = after["startup"]
    assert st["restored"] == len(entries) and st["restored_tokens"] >= 60
    assert after["coordinator_epoch"] > before["coordinator_epoch"]
    assert all(w["coordinator_epoch"] == after["coordinator_epoch"] for w in after["workers"])
    assert all(w["cached_requests"] == 0 for w in after["workers"])  # old-id caches released, new ones freed
    if detach:
        assert st["adopted_stages"] == [0, 1]
        assert after["worker_pids"] == before["worker_pids"]
    else:
        assert st["spawned_stages"] == [0, 1]
        assert set(after["worker_pids"].values()).isdisjoint(before["worker_pids"].values())
        assert all(a > b for a, b in zip(after["epochs"], before["epochs"]))
    print(f"\ndetach={detach}: kill -> serving {(st['serving'] - st['process_start']) * 1e3:.0f} ms after process "
          f"start, all restored requests resumed {(st['replayed'] - st['process_start']) * 1e3:.0f} ms after; "
          f"client gap {max(r.resumed_at[0] - r.broken_at[0] for r in results) * 1e3:.0f} ms")


def test_finished_request_can_be_resumed_after_restart(reference, tmp_path):
    """The final chunk can be lost with the old coordinator; the log still has the result."""
    e = reference["prompts"][3]
    n = 12
    with LocalCluster(port=50710, log_name="test_coord_resume_done",
                      extra=["--request-log", str(tmp_path / "log.db"), "--detach-workers"]) as c:
        async def first():
            async with channel(c.addr) as ch:
                return await generate(pb_grpc.CoordinatorServiceStub(ch), prompt_ids=e["prompt_token_ids"],
                                      max_new_tokens=n, request_id="done-1")

        assert asyncio.run(first()).status == "COMPLETED"
        c.kill()
        c.restart()

        async def resume(rid, from_step):
            async with channel(c.addr) as ch:
                stub = pb_grpc.CoordinatorServiceStub(ch)
                return [ch async for ch in stub.Resume(pb.ResumeRequest(request_id=rid, from_step=from_step))]

        chunks = asyncio.run(resume("done-1", 5))
        assert [x.token_id for x in chunks[:-1]] == e["output_token_ids"][5:n]
        assert chunks[-1].final and chunks[-1].status == pb.COMPLETED
        assert list(chunks[-1].token_ids) == e["output_token_ids"][:n]
        (missing,) = asyncio.run(resume("never-seen", 0))
        assert missing.status == pb.NOT_FOUND


def test_replaced_coordinator_is_fenced_out(reference, tmp_path):
    """A coordinator that was frozen (not dead) while a new one took over must not act when it wakes up:
    workers reject it, it cannot write the log, and it shuts itself down."""
    db = str(tmp_path / "log.db")
    common = ["--request-log", db, "--detach-workers"]
    old = LocalCluster(port=50720, worker_base_port=50721, log_name="test_fence_old", extra=common).start()
    new = None
    try:
        old_stats = old.stats(workers=False)
        os.kill(old.proc.pid, signal.SIGSTOP)                    # "frozen": GC pause, VM stall, partition
        new = LocalCluster(port=50730, worker_base_port=50721, log_name="test_fence_new", extra=common).start()
        s = new.stats()
        assert s["startup"]["adopted_stages"] == [0, 1]
        assert s["coordinator_epoch"] > old_stats["coordinator_epoch"]
        os.kill(old.proc.pid, signal.SIGCONT)
        deadline = time.time() + 10
        while old.proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        assert old.proc.poll() is not None, "the replaced coordinator kept running"
        # The new coordinator and the adopted workers are unaffected.
        e = reference["prompts"][0]

        async def one():
            async with channel(new.addr) as ch:
                return await generate(pb_grpc.CoordinatorServiceStub(ch), prompt_ids=e["prompt_token_ids"],
                                      max_new_tokens=8, request_id="after-fence")

        r = asyncio.run(one())
        assert r.status == "COMPLETED" and r.tokens == e["output_token_ids"][:8]
        assert all(w["coordinator_epoch"] == s["coordinator_epoch"] for w in new.stats()["workers"])
    finally:
        if old.proc is not None and old.proc.poll() is None:
            os.kill(old.proc.pid, signal.SIGCONT)
        old.proc = None  # it must not take the shared workers down with it
        if new is not None:
            new.stop()
        old.reap_workers()
