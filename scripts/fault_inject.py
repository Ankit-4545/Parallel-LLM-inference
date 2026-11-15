"""O7 / E6-E8: fault injection.

    python scripts/fault_inject.py kill --stage 0 --reps 3   # E6: SIGKILL worker 1 (stage 0) mid-generation
    python scripts/fault_inject.py kill --stage 1 --reps 3   # E7: SIGKILL worker 2 (stage 1) mid-generation
    python scripts/fault_inject.py stale                     # E8: inject pre-crash-epoch + duplicate messages
    python scripts/fault_inject.py coordinator --reps 3      # E10: SIGKILL the coordinator, restart it, resume
    python scripts/fault_inject.py all                       # everything above

Each kill run streams several concurrent requests, SIGKILLs the worker process
once a fraction of the tokens has been produced, and then checks that every
response is token-for-token identical to reference_outputs.json (no duplicate,
missing or wrong tokens). Recovery time is split into detection (kill -> worker
declared dead), restart (-> new incarnation healthy) and replay (-> every
affected request has rebuilt its KV cache and produced its next token).

E10 SIGKILLs the *coordinator* instead, then plays the role of systemd and
starts a new incarnation with the same --request-log. Clients reconnect with
Resume. It runs twice: workers adopted (--detach-workers) and workers respawned
(attached workers die with their coordinator), and also measures what the
request log costs during normal operation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grpc  # noqa: E402

from benchmarks.common import RESULTS_DIR, cluster, correctness, entries_for, ensure_dirs, write_csv  # noqa: E402
from pllm.client import get_stats  # noqa: E402
from pllm.proto import pipeline_pb2 as pb  # noqa: E402
from pllm.proto import pipeline_pb2_grpc as pb_grpc  # noqa: E402

from benchmarks.common import run_workload  # noqa: E402


async def kill_run(c, stage: int, rep: int, n_requests: int, n: int, kill_frac: float):
    entries = entries_for(n_requests)
    before = await get_stats(c.addr, workers=False)
    pid = int(before["worker_pids"][str(stage)])
    base_tokens = before["counters"].get("tokens_generated", 0)
    pre_crash_epochs = list(before["epochs"])
    target = int(kill_frac * n_requests * n)

    task = asyncio.create_task(run_workload(c.addr, entries, n))
    while True:
        s = await get_stats(c.addr, workers=False)
        if s["counters"].get("tokens_generated", 0) - base_tokens >= target:
            break
        await asyncio.sleep(0.005)
    t_kill = time.monotonic()
    os.kill(pid, signal.SIGKILL)
    results, wall = await task
    after = await get_stats(c.addr, workers=False)

    ev = [e for e in after["recovery_events"] if e["stage"] == stage][-1]
    t0c = after["t0_monotonic"]
    to_kill_ms = lambda t_coord: (t0c + t_coord - t_kill) * 1e3  # noqa: E731
    ok, done = correctness(results, entries, n)
    dup = sum(max(0, len(r.tokens) - n) for r in results)
    missing = sum(max(0, n - len(r.tokens)) for r in results if r.status == "COMPLETED")
    row = {
        "stage": stage, "rep": rep, "requests": n_requests, "max_new_tokens": n,
        "tokens_before_kill": target, "old_epoch": ev["old_epoch"], "new_epoch": ev.get("new_epoch"),
        "result": ev.get("result"), "completed": done, "correct": ok, "duplicate_tokens": dup,
        "missing_tokens": missing,
        "detect_ms": to_kill_ms(ev["detected_t"]),
        "restart_ms": ev.get("restart_ms"),
        "replay_ms": ev.get("replay_ms"),
        "total_recovery_ms": to_kill_ms(ev["replayed_t"]) if "replayed_t" in ev else None,
        "heartbeat_gap_ms": ev.get("detection_ms"),
        "affected_requests": ev.get("affected_requests"),
        "discarded_stale_epoch_batches": after["counters"].get("discarded_stale_epoch_batches", 0)
        - before["counters"].get("discarded_stale_epoch_batches", 0),
        "wall_s": wall,
    }
    timeline = []
    for i, r in enumerate(results):
        for step, t in enumerate(r.token_times):
            timeline.append({"stage": stage, "rep": rep, "request": i, "step": step,
                             "t_ms": (t - t_kill) * 1e3})
    marks = {"stage": stage, "rep": rep, "kill_ms": 0.0, "detect_ms": row["detect_ms"],
             "restarted_ms": to_kill_ms(ev["restarted_t"]),
             "replayed_ms": row["total_recovery_ms"]}
    return row, timeline, marks, pre_crash_epochs, after["epochs"]


async def inject_stale(c, pre_crash_epochs, current_epochs) -> dict:
    """Send a message stamped with the pre-crash epochs to stage 0 and expect a rejection."""
    entries = entries_for(1)
    msg = pb.ForwardRequest(stage_epochs=pre_crash_epochs, batch_id=10**9, items=[pb.StageItem(
        request_id=f"stale-{time.monotonic_ns()}", step=0, attempt=0, reset=True, use_cache=True,
        token_ids=entries[0]["prompt_token_ids"])])
    async with grpc.aio.insecure_channel(c.worker_addr(0)) as ch:
        resp = await pb_grpc.StageServiceStub(ch).Forward(msg, timeout=5)
    r = resp.results[0]
    return {"injected_epochs": pre_crash_epochs, "current_epochs": current_epochs,
            "status": pb.ItemStatus.Name(r.status), "detail": r.detail,
            "rejected": r.status in (pb.STALE_EPOCH,)}


async def inject_duplicates(c, n: int = 32) -> dict:
    """Duplicate (request_id, attempt, step) deliveries straight to the workers:
    a duplicate of the last applied step is answered from the idempotency cache,
    an older one is rejected, and neither changes the KV cache."""
    e = entries_for(1)[0]
    s = await get_stats(c.addr, workers=False)
    epochs = s["epochs"]
    rid = f"dup-{time.monotonic_ns()}"
    out = []
    async with grpc.aio.insecure_channel(c.worker_addr(0)) as ch:
        stub = pb_grpc.StageServiceStub(ch)

        async def send(step, tokens, reset=False, release=False):
            m = pb.ForwardRequest(stage_epochs=epochs, batch_id=1, items=[pb.StageItem(
                request_id=rid, step=step, attempt=0, reset=reset, use_cache=True, release_after=release,
                token_ids=tokens)])
            return (await stub.Forward(m, timeout=5)).results[0]

        r0 = await send(0, e["prompt_token_ids"], reset=True)
        out.append(r0.token_id)
        log = []
        for step in range(1, 6):
            r = await send(step, [out[-1]])
            d = await send(step, [out[-1]])          # immediate duplicate
            log.append({"step": step, "first": r.token_id, "dup_status": pb.ItemStatus.Name(d.status),
                        "dup_replayed": d.replayed, "dup_token": d.token_id})
            out.append(r.token_id)
        old = await send(2, [out[1]])                 # an old step arriving late
        await c_release(c, rid)
    return {"tokens": out, "reference": e["output_token_ids"][:6], "match": out == e["output_token_ids"][:6],
            "duplicates": log, "late_old_step_status": pb.ItemStatus.Name(old.status),
            "all_duplicates_idempotent": all(x["dup_replayed"] and x["dup_token"] == x["first"] for x in log)}


async def c_release(c, rid):
    for stage in range(2):
        async with grpc.aio.insecure_channel(c.worker_addr(stage)) as ch:
            await pb_grpc.StageServiceStub(ch).Release(pb.ReleaseRequest(request_ids=[rid]), timeout=5)


async def duplicate_dispatch_run(n_requests=16, n=32) -> dict:
    """Coordinator-level E8: every micro-batch is delivered twice."""
    entries = entries_for(n_requests)
    with cluster("duplicate_dispatch", ["--duplicate-dispatch"], port_offset=60) as c:
        results, wall = await run_workload(c.addr, entries, n)
        s = await get_stats(c.addr, workers=True)
    ok, done = correctness(results, entries, n)
    ctr = s["counters"]
    return {"requests": n_requests, "completed": done, "correct": ok,
            "duplicate_tokens": sum(max(0, len(r.tokens) - n) for r in results),
            "duplicates_sent": ctr.get("duplicates_sent", 0),
            "discarded_duplicate_results": ctr.get("discarded_duplicate_results", 0),
            "worker_idempotent_replays": [w.get("replayed") for w in s["workers"]],
            "worker_rejections": [w.get("rejected") for w in s["workers"]]}


async def coordinator_kill_run(c, variant: str, rep: int, n_requests: int, n: int, kill_frac: float):
    """E10: SIGKILL the coordinator mid-generation, restart it, let clients Resume."""
    from pllm.client import channel, generate
    from pllm.proto import pipeline_pb2_grpc as pb_grpc

    entries = entries_for(n_requests)
    before = await get_stats(c.addr, workers=False)
    base_tokens = before["counters"].get("tokens_generated", 0)
    target = int(kill_frac * n_requests * n)
    async with channel(c.addr) as ch:
        stub = pb_grpc.CoordinatorServiceStub(ch)
        tasks = [asyncio.create_task(generate(stub, prompt_ids=e["prompt_token_ids"], max_new_tokens=n,
                                              request_id=f"e10-{variant}-{rep}-{i}", reconnect_s=120))
                 for i, e in enumerate(entries)]
        while True:
            s = await get_stats(c.addr, workers=False)
            if s["counters"].get("tokens_generated", 0) - base_tokens >= target:
                break
            await asyncio.sleep(0.005)
        t_kill_mono = time.monotonic()
        t_kill = await asyncio.to_thread(c.kill)
        await asyncio.to_thread(c.restart)          # what systemd / launchd would do
        results = await asyncio.gather(*tasks)
    after = await get_stats(c.addr, workers=True)
    st = after["startup"]
    ms = lambda t: (t - t_kill) * 1e3 if t else None  # noqa: E731
    ok, done = correctness(results, entries, n)
    gaps = [(r.resumed_at[0] - r.broken_at[0]) * 1e3 for r in results if r.resumed_at and r.broken_at]
    row = {
        "variant": variant, "rep": rep, "requests": n_requests, "max_new_tokens": n,
        "tokens_before_kill": st.get("restored_tokens"), "restored_requests": st.get("restored"),
        "completed": done, "correct": ok,
        "duplicate_tokens": sum(max(0, len(r.tokens) - n) for r in results),
        "missing_tokens": sum(max(0, n - len(r.tokens)) for r in results if r.status == "COMPLETED"),
        "reconnects": sum(r.reconnects for r in results),
        "adopted_stages": len(st.get("adopted_stages", [])), "spawned_stages": len(st.get("spawned_stages", [])),
        "process_start_ms": ms(st.get("process_start")),
        "init_ms": (st["start_called"] - st["process_start"]) * 1e3,          # interpreter + imports + tokenizer
        "workers_ms": (st["workers_ready"] - st["start_called"]) * 1e3,       # adopt (ping) or spawn + load weights
        "log_load_ms": st.get("log_load_ms"),
        "serving_ms": ms(st.get("serving")),
        "replay_ms": (st["replayed"] - st["serving"]) * 1e3 if st.get("replayed") else None,
        "total_recovery_ms": ms(st.get("replayed")),
        "client_gap_ms_mean": statistics.fmean(gaps) if gaps else None,
        "client_gap_ms_max": max(gaps) if gaps else None,
        "pids_kept": after["worker_pids"] == before["worker_pids"],
    }
    timeline = [{"variant": variant, "rep": rep, "request": i, "step": step, "t_ms": (t - t_kill_mono) * 1e3}
                for i, r in enumerate(results) for step, t in enumerate(r.token_times)]
    marks = {"variant": variant, "rep": rep, "kill_ms": 0.0, "process_start_ms": row["process_start_ms"],
             "workers_ready_ms": ms(st.get("workers_ready")), "serving_ms": row["serving_ms"],
             "replayed_ms": row["total_recovery_ms"]}
    return row, timeline, marks


async def run_coordinator_kills(reps, n_requests, n, kill_frac):
    rows, timeline, marks = [], [], []
    tmp = RESULTS_DIR / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for variant, flags in (("adopt", ["--detach-workers"]), ("respawn", [])):
        db = tmp / f"e10_{variant}.db"
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
        with cluster(f"e10_{variant}", ["--max-batch-size", "8", "--request-log", str(db), *flags],
                     port_offset=80) as c:
            for rep in range(reps):
                row, tl, mk = await coordinator_kill_run(c, variant, rep, n_requests, n, kill_frac)
                rows.append(row)
                timeline += tl
                marks.append(mk)
                print(json.dumps(row))
                await asyncio.sleep(0.5)
    return rows, timeline, marks


async def request_log_overhead(reps, n=32):
    """What durability costs while nothing fails: no log vs WAL/NORMAL vs WAL/FULL (+F_FULLFSYNC)."""
    from benchmarks.common import result_metrics

    rows = []
    tmp = RESULTS_DIR / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for mode in ("none", "normal", "full"):
        db = tmp / f"e10_overhead_{mode}.db"
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
        extra = ["--max-batch-size", "8"] + ([] if mode == "none" else
                                             ["--request-log", str(db), "--request-log-sync", mode])
        with cluster(f"e10_overhead_{mode}", extra, port_offset=90) as c:
            await run_workload(c.addr, entries_for(4), 8)          # warm-up
            for rep in range(reps):
                single, w1 = await run_workload(c.addr, entries_for(8), n, concurrency=1)
                await get_stats(c.addr, reset=True, workers=False)
                conc, w32 = await run_workload(c.addr, entries_for(32), n)
                s = await get_stats(c.addr, workers=False)
                m1 = result_metrics(single, entries_for(8), n, w1)
                m32 = result_metrics(conc, entries_for(32), n, w32)
                lg = s.get("request_log") or {}
                row = {"log_mode": mode, "rep": rep, "single_itl_ms_mean": m1["itl_ms_mean"],
                       "conc32_throughput_tok_s": m32["throughput_tok_s"], "conc32_itl_ms_p50": m32["itl_ms_p50"],
                       "correct_rate": (m1["correct"] + m32["correct"]) / 40,
                       "commit_ms_mean": lg.get("mean_commit_ms"), "commit_ms_max": lg.get("max_commit_ms")}
                rows.append(row)
                print(json.dumps(row))
    return rows


async def run_kills(stages, reps, n_requests, n, kill_frac):
    rows, timeline, marks, stale = [], [], [], []
    for stage in stages:
        with cluster(f"fault_stage{stage}", ["--max-batch-size", "8"], port_offset=50) as c:
            for rep in range(reps):
                row, tl, mk, pre, cur = await kill_run(c, stage, rep, n_requests, n, kill_frac)
                rows.append(row)
                timeline += tl
                marks.append(mk)
                print(json.dumps(row))
                if rep == 0:
                    st = await inject_stale(c, pre, cur)
                    st["killed_stage"] = stage
                    stale.append(st)
                    print("stale-epoch injection:", json.dumps(st))
                await asyncio.sleep(0.5)
    return rows, timeline, marks, stale


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["kill", "stale", "coordinator", "all"])
    ap.add_argument("--stage", type=int, choices=[0, 1], default=None, help="stage to kill (default: both)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--kill-at", type=float, default=0.3, help="fraction of tokens generated before the kill")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    ensure_dirs()
    ok = True
    summary = {}
    if a.mode in ("kill", "all"):
        stages = [a.stage] if a.stage is not None else [0, 1]
        rows, timeline, marks, stale = asyncio.run(run_kills(stages, a.reps, a.requests, a.max_new_tokens,
                                                             a.kill_at))
        write_csv("e6_e7_recovery.csv", rows)
        write_csv("e6_e7_timeline.csv", timeline)
        write_csv("e6_e7_marks.csv", marks)
        summary["kills"] = rows
        summary["stale_epoch_injection"] = stale
        ok &= all(r["correct"] == r["requests"] and r["duplicate_tokens"] == 0 and r["missing_tokens"] == 0
                  for r in rows)
        ok &= all(s["rejected"] for s in stale)
    if a.mode in ("stale", "all"):
        async def e8():
            with cluster("e8_stale", [], port_offset=70) as c:
                s0 = await get_stats(c.addr, workers=False)
                pid = int(s0["worker_pids"]["1"])
                pre = list(s0["epochs"])
                os.kill(pid, signal.SIGKILL)  # crash stage 1 while idle, wait for the new incarnation
                while True:
                    s = await get_stats(c.addr, workers=False)
                    if s["epochs"][1] > pre[1]:
                        break
                    await asyncio.sleep(0.05)
                stale = await inject_stale(c, pre, s["epochs"])
                dups = await inject_duplicates(c)
            dd = await duplicate_dispatch_run()
            return stale, dups, dd

        stale, dups, dd = asyncio.run(e8())
        summary["e8_stale_epoch"] = stale
        summary["e8_direct_duplicates"] = dups
        summary["e8_duplicate_dispatch"] = dd
        print(json.dumps({"stale": stale, "duplicates": dups, "duplicate_dispatch": dd}, indent=2))
        ok &= stale["rejected"] and dups["match"] and dups["all_duplicates_idempotent"]
        ok &= dd["correct"] == dd["requests"] and dd["duplicate_tokens"] == 0
    if a.mode in ("coordinator", "all"):
        rows, timeline, marks = asyncio.run(run_coordinator_kills(a.reps, a.requests, a.max_new_tokens, a.kill_at))
        write_csv("e10_coordinator_recovery.csv", rows)
        write_csv("e10_timeline.csv", timeline)
        write_csv("e10_marks.csv", marks)
        overhead = asyncio.run(request_log_overhead(a.reps))
        write_csv("e10_log_overhead.csv", overhead)
        ok &= all(r["correct"] == r["requests"] and r["duplicate_tokens"] == 0 and r["missing_tokens"] == 0
                  for r in rows)
        ok &= all(r["correct_rate"] == 1.0 for r in overhead)
        if a.plot:
            from benchmarks.plots import plot_e10

            plot_e10()
    prev = {}
    path = RESULTS_DIR / "e6_e8_fault_summary.json"
    if path.exists():
        prev = json.loads(path.read_text())
    prev.update(summary)
    path.write_text(json.dumps(prev, indent=2))
    if a.plot and a.mode in ("kill", "all"):
        from benchmarks.plots import plot_recovery_timeline

        plot_recovery_timeline()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
