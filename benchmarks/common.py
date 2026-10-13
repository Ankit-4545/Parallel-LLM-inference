"""Shared helpers for the benchmark suite: workloads, metrics, CSV output."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path

from pllm.client import channel, generate
from pllm.cluster import LocalCluster, load_reference
from pllm.metrics import percentile, summarize
from pllm.proto import pipeline_pb2_grpc as pb_grpc

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
PLOTS_DIR = BENCH_DIR / "plots"
BENCH_PORT = 50400  # benchmark clusters use 50400+ so they never clash with a manual cluster on 50050


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(name: str, rows: list[dict]) -> Path:
    ensure_dirs()
    path = RESULTS_DIR / name
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in keys})
    return path


def read_csv(name: str) -> list[dict]:
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", None) else math.nan
                if r[k].is_integer() and "." not in v and "e" not in v.lower():
                    r[k] = int(r[k])
            except (ValueError, AttributeError):
                pass
    return rows


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return v


def reference():
    return load_reference()


def entries_for(n_requests: int, ref=None) -> list[dict]:
    ref = ref or reference()
    p = ref["prompts"]
    return [p[i % len(p)] for i in range(n_requests)]


def cluster(name: str, extra: list[str], port_offset: int = 0) -> LocalCluster:
    return LocalCluster(port=BENCH_PORT + port_offset, worker_base_port=BENCH_PORT + port_offset + 1,
                        extra=extra, log_name=f"bench_{name}")


async def run_workload(addr: str, entries: list[dict], max_new_tokens: int, concurrency: int | None = None,
                       arrival_interval_s: float = 0.0, timeout_s: float = 0.0) -> tuple[list, float]:
    """Submit `entries` (closed loop with at most `concurrency` outstanding, or open loop with a fixed
    inter-arrival time). Returns (results, wall seconds)."""
    async with channel(addr) as ch:
        stub = pb_grpc.CoordinatorServiceStub(ch)
        sem = asyncio.Semaphore(concurrency or len(entries))

        async def one(i, e):
            if arrival_interval_s:
                await asyncio.sleep(i * arrival_interval_s)
            async with sem:
                return await generate(stub, prompt_ids=e["prompt_token_ids"], max_new_tokens=max_new_tokens,
                                      request_id=f"b{i}-{time.monotonic_ns()}", timeout_s=timeout_s)

        t0 = time.monotonic()
        results = await asyncio.gather(*[one(i, e) for i, e in enumerate(entries)])
        return results, time.monotonic() - t0


def correctness(results, entries, n) -> tuple[int, int]:
    """(# completed requests matching the reference exactly, # completed)."""
    done = [(r, e) for r, e in zip(results, entries) if r.status == "COMPLETED"]
    ok = sum(1 for r, e in done if r.tokens == e["output_token_ids"][:n])
    return ok, len(done)


def result_metrics(results, entries, n, wall_s: float) -> dict:
    done = [r for r in results if r.status == "COMPLETED"]
    tokens = sum(len(r.tokens) for r in done)
    ok, n_done = correctness(results, entries, n)
    gaps = [g for r in done for g in r.itl_ms]
    out = {
        "requests": len(results), "completed": n_done,
        "rejected": sum(1 for r in results if r.status == "OVERLOADED"),
        "correct": ok, "correct_rate": ok / n_done if n_done else math.nan,
        "wall_s": wall_s, "tokens": tokens, "throughput_tok_s": tokens / wall_s if wall_s > 0 else math.nan,
    }
    out.update(summarize([r.ttft_ms for r in done], "ttft_ms_"))
    out.update(summarize(gaps, "itl_ms_"))
    out.update(summarize([r.latency_ms for r in done], "latency_ms_"))
    return out


def cluster_metrics(stats: dict) -> dict:
    """Utilisation, bytes and a time breakdown from a coordinator Stats(reset=True) snapshot."""
    c = stats.get("counters", {})
    out = {"mean_batch_size": stats.get("mean_batch_size", math.nan), "batches": stats.get("batches", 0)}
    tokens = c.get("tokens_generated", 0)
    out["activation_bytes"] = c.get("activation_bytes", 0)
    out["wire_bytes"] = c.get("wire_bytes", 0)
    out["activation_bytes_per_token"] = out["activation_bytes"] / tokens if tokens else math.nan
    out["wire_bytes_per_token"] = out["wire_bytes"] / tokens if tokens else math.nan
    for w in stats.get("workers", []):
        s = w.get("stage")
        out[f"s{s}_utilisation"] = w.get("utilisation", math.nan)
        out[f"s{s}_rss_mb"] = w.get("rss_bytes", 0) / 2**20
        out[f"s{s}_max_queue"] = w.get("max_queue_depth", 0)
    out.update(breakdown(stats.get("batch_log", [])))
    out["coordinator_rss_mb"] = stats.get("rss_bytes", 0) / 2**20
    return out


def breakdown(batch_log: list[dict]) -> dict:
    """Mean per-batch time split into compute, (de)serialisation, queueing and RPC/network overhead."""
    logs = [b for b in batch_log if b.get("ok") and "s0_compute_ms" in b]
    if not logs:
        return {}
    def mean(k):
        vals = [b.get(k, 0.0) or 0.0 for b in logs]
        return statistics.fmean(vals)
    rpc = mean("rpc_ms")
    compute = mean("s0_compute_ms") + mean("s1_compute_ms")
    serde = mean("s0_ser_ms") + mean("s0_deser_ms") + mean("s1_ser_ms") + mean("s1_deser_ms")
    queue = mean("s0_queue_ms") + mean("s1_queue_ms")
    delay = mean("s0_link_delay_ms")
    overhead = max(0.0, rpc - compute - serde - queue - delay)
    return {"bd_rpc_ms": rpc, "bd_compute_ms": compute, "bd_s0_compute_ms": mean("s0_compute_ms"),
            "bd_s1_compute_ms": mean("s1_compute_ms"), "bd_serde_ms": serde, "bd_queue_ms": queue,
            "bd_link_delay_ms": delay, "bd_rpc_overhead_ms": overhead}


def aggregate(rows: list[dict], keys: list[str], group_by: list[str]) -> list[dict]:
    """Mean and stdev across repetitions for each group."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[g] for g in group_by), []).append(r)
    out = []
    for gk, rs in groups.items():
        row = dict(zip(group_by, gk))
        row["reps"] = len(rs)
        for k in keys:
            vals = [r[k] for r in rs if isinstance(r.get(k), (int, float)) and not math.isnan(r[k])]
            row[k] = statistics.fmean(vals) if vals else math.nan
            row[k + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def baseline_metrics(ref: dict, entries: list[dict] | None = None) -> dict:
    """Metrics of the single-process baseline from per-token latencies recorded in a run."""
    entries = entries or ref["prompts"]
    lat = [e["per_token_latency_ms"] for e in entries]
    total_s = sum(sum(l) for l in lat) / 1e3
    tokens = sum(len(l) for l in lat)
    out = {"tokens": tokens, "wall_s": total_s, "throughput_tok_s": tokens / total_s}
    out.update(summarize([l[0] for l in lat], "ttft_ms_"))
    out.update(summarize([x for l in lat for x in l[1:]], "itl_ms_"))
    out.update(summarize([sum(l) for l in lat], "latency_ms_"))
    return out


def p(values, q):
    return percentile(values, q)


def env_info() -> dict:
    import platform

    import torch

    return {"python": platform.python_version(), "torch": torch.__version__, "machine": platform.machine(),
            "platform": platform.platform(), "cpus": os.cpu_count(),
            "threads_per_process": int(os.environ.get("PLLM_THREADS", "4"))}
