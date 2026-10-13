"""O5 / E5: load test with sustained overload.

Fires an open-loop stream of requests faster than the system can serve them and
samples the coordinator / worker queue depths, rejections and memory every
100 ms. Writes a time-series CSV, a summary JSON and (with --plot) a chart.

    python scripts/load_test.py                       # starts its own cluster
    python scripts/load_test.py --addr 127.0.0.1:50050 # use an already running coordinator
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import RESULTS_DIR, cluster, correctness, entries_for, ensure_dirs, write_csv  # noqa: E402
from pllm.client import channel, generate, get_stats  # noqa: E402
from pllm.proto import pipeline_pb2_grpc as pb_grpc  # noqa: E402


async def load_test(addr: str, n_requests: int, rate: float, max_new_tokens: int, sample_s: float = 0.1,
                    tail_s: float = 2.0) -> tuple[list[dict], dict]:
    entries = entries_for(n_requests)
    samples: list[dict] = []
    stop = asyncio.Event()
    t0 = time.monotonic()
    rejected_seen = {"n": 0}

    async def sampler():
        while not stop.is_set():
            try:
                s = await get_stats(addr, workers=True)
            except Exception:
                await asyncio.sleep(sample_s)
                continue
            row = {"t_s": time.monotonic() - t0, "admission_depth": s["admission_depth"],
                   "admission_capacity": s["admission_capacity"], "active": s["active"],
                   "active_capacity": s["active_capacity"], "inflight_batches": s["inflight_batches"],
                   "rejected_total": s["counters"].get("rejected_overloaded", 0),
                   "completed_total": s["counters"].get("finished_completed", 0),
                   "coordinator_rss_mb": s["rss_bytes"] / 2**20,
                   "client_rejections": rejected_seen["n"]}
            for w in s.get("workers", []):
                st = w.get("stage")
                row[f"s{st}_queue_depth"] = w.get("queue_depth", 0)
                row[f"s{st}_queue_size"] = w.get("queue_size", 0)
                row[f"s{st}_rss_mb"] = w.get("rss_bytes", 0) / 2**20
                row[f"s{st}_cached_requests"] = w.get("cached_requests", 0)
            samples.append(row)
            await asyncio.sleep(sample_s)

    async with channel(addr) as ch:
        stub = pb_grpc.CoordinatorServiceStub(ch)

        async def one(i, e):
            await asyncio.sleep(i / rate)
            r = await generate(stub, prompt_ids=e["prompt_token_ids"], max_new_tokens=max_new_tokens,
                               request_id=f"load{i}-{time.monotonic_ns()}", rpc_timeout=600)
            r.t_submit_rel = r.t_start - t0
            r.t_done_rel = r.t_end - t0
            if r.status == "OVERLOADED":
                rejected_seen["n"] += 1
            return r

        await get_stats(addr, reset=True, workers=True)
        st = asyncio.create_task(sampler())
        results = await asyncio.gather(*[one(i, e) for i, e in enumerate(entries)])
        await asyncio.sleep(tail_s)
        stop.set()
        await st
        final = await get_stats(addr, workers=True)

    statuses: dict[str, int] = {}
    for r in results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    ok, done = correctness(results, entries, max_new_tokens)
    rej_latency = [r.latency_ms for r in results if r.status == "OVERLOADED"]
    rss = {k: [s[k] for s in samples if k in s] for k in ("coordinator_rss_mb", "s0_rss_mb", "s1_rss_mb")}
    summary = {
        "requests": n_requests, "offered_rate_rps": rate, "max_new_tokens": max_new_tokens,
        "statuses": statuses, "rejection_rate": statuses.get("OVERLOADED", 0) / n_requests,
        "completed": done, "completed_correct": ok,
        "hung": sum(1 for r in results if r.status not in ("COMPLETED", "OVERLOADED", "TIMEOUT", "FAILED")),
        "max_rejection_latency_ms": max(rej_latency) if rej_latency else 0.0,
        "mean_rejection_latency_ms": (sum(rej_latency) / len(rej_latency)) if rej_latency else 0.0,
        "max_admission_depth": final["max_admission_depth"], "admission_capacity": final["admission_capacity"],
        "max_active": final["max_active_seen"], "active_capacity": final["active_capacity"],
        "worker_max_queue_depth": [w.get("max_queue_depth") for w in final.get("workers", [])],
        "worker_queue_size": [w.get("queue_size") for w in final.get("workers", [])],
        "rss_mb_min_max": {k: [min(v), max(v)] if v else None for k, v in rss.items()},
        "goodput_tok_s": sum(len(r.tokens) for r in results if r.status == "COMPLETED")
        / max(1e-9, max(r.t_done_rel for r in results)),
    }
    return samples, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--addr", default=None, help="existing coordinator (default: start a dedicated cluster)")
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--rate", type=float, default=40.0, help="offered load, requests/second")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--admission-queue", type=int, default=16)
    ap.add_argument("--max-active", type=int, default=16)
    ap.add_argument("--tag", default="e5_overload")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    ensure_dirs()

    async def go(addr):
        return await load_test(addr, a.requests, a.rate, a.max_new_tokens)

    if a.addr:
        samples, summary = asyncio.run(go(a.addr))
    else:
        extra = ["--admission-queue", str(a.admission_queue), "--max-active", str(a.max_active),
                 "--max-batch-size", "8", "--max-inflight", "2"]
        with cluster("load_test", extra, port_offset=40) as c:
            samples, summary = asyncio.run(go(c.addr))
    write_csv(f"{a.tag}_timeseries.csv", samples)
    (RESULTS_DIR / f"{a.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if a.plot:
        from benchmarks.plots import plot_overload

        plot_overload(a.tag)
    bounded = (summary["max_admission_depth"] <= summary["admission_capacity"]
               and summary["max_active"] <= summary["active_capacity"]
               and all(d <= q for d, q in zip(summary["worker_max_queue_depth"], summary["worker_queue_size"])))
    ok = bounded and summary["hung"] == 0 and summary["completed_correct"] == summary["completed"]
    print("PASS" if ok else "FAIL", "- queues bounded, no hangs, all completed requests correct")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
