"""O8: one command that reproduces every benchmark and regenerates every plot.

    python -m benchmarks.run_all                 # everything, 3 repetitions each (~25-35 min on a laptop CPU)
    python -m benchmarks.run_all --quick         # 1 repetition, fewer sweep points (~8 min)
    python -m benchmarks.run_all --only e1,e4    # a subset
    python -m benchmarks.plots                   # only regenerate plots/summary from existing CSVs

Experiments (see README): e1 e2 e3 e4 e5 e6 (=E6+E7) e8 e9 plen (prompt length) fp16 (activation compression)
Outputs: benchmarks/results/*.csv|json, benchmarks/results/summary.md, benchmarks/plots/*.png
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import (RESULTS_DIR, aggregate, baseline_metrics, cluster, cluster_metrics,  # noqa: E402
                               ensure_dirs, entries_for, env_info, reference, result_metrics, run_workload,
                               write_csv)
from pllm.client import get_stats  # noqa: E402
from pllm.proto import pipeline_pb2 as pb  # noqa: E402
from pllm.proto import pipeline_pb2_grpc as pb_grpc  # noqa: E402

N = 32  # tokens per request (matches reference_outputs.json)

# Configurations compared throughout (O8: baseline / pipeline without batching / with micro-batching)
CONFIGS = {
    "pipeline_unbatched": ["--max-batch-size", "1", "--max-inflight", "1"],
    "pipeline_microbatch": ["--max-batch-size", "8", "--max-inflight", "2"],
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


async def measured(c, entries, n, concurrency=None):
    await get_stats(c.addr, reset=True, workers=True)
    results, wall = await run_workload(c.addr, entries, n, concurrency=concurrency)
    stats = await get_stats(c.addr, reset=True, workers=True)
    m = result_metrics(results, entries, n, wall)
    m.update(cluster_metrics(stats))
    return m, results


# --------------------------------------------------------------------------- baseline
def run_baseline(reps: int, entries=None, n=N, tag="baseline"):
    """Single-process HF baseline, measured in this process (no cluster running)."""
    from baseline import greedy_generate, load_model
    from pllm.config import DEFAULT_THREADS
    from pllm.model_loader import set_determinism

    set_determinism(threads=DEFAULT_THREADS)
    model, _, _ = load_model()
    greedy_generate(model, [464, 3290], 4)  # warm-up
    ref = reference()
    entries = entries or ref["prompts"]
    rows = []
    for rep in range(reps):
        runs = []
        for e in entries:
            out, lat = greedy_generate(model, e["prompt_token_ids"], n)
            runs.append({"per_token_latency_ms": [x * 1e3 for x in lat],
                         "match": out == e["output_token_ids"][:n] if "output_token_ids" in e else True,
                         "tokens": out})
        m = baseline_metrics(ref, runs)
        m.update({"system": tag, "rep": rep, "correct_rate": sum(r["match"] for r in runs) / len(runs)})
        rows.append(m)
    return rows, model


def run_split_inprocess(reps: int, n=N):
    """Both stages in one process, no RPC: isolates the cost of the network hop."""
    from pllm.partition import PipelineStage, load_stages
    from pllm.config import DEFAULT_THREADS
    from pllm.model_loader import set_determinism
    import torch

    set_determinism(threads=DEFAULT_THREADS)
    stages = load_stages()
    ref = reference()
    rows = []
    with torch.inference_mode():
        for rep in range(reps):
            runs = []
            s0, s1 = PipelineStage(stages[0]), PipelineStage(stages[1])
            for e in ref["prompts"]:
                lat, out = [], []
                inp = torch.tensor(e["prompt_token_ids"])
                for step in range(n):
                    t0 = time.perf_counter()
                    tok = int(s1.forward(s0.forward(inp, "r", step, reset=step == 0), "r", step,
                                         reset=step == 0).argmax())
                    lat.append((time.perf_counter() - t0) * 1e3)
                    out.append(tok)
                    inp = torch.tensor([tok])
                runs.append({"per_token_latency_ms": lat, "match": out == e["output_token_ids"][:n]})
            m = baseline_metrics(ref, runs)
            m.update({"system": "split_inprocess", "rep": rep,
                      "correct_rate": sum(r["match"] for r in runs) / len(runs)})
            rows.append(m)
    return rows


# --------------------------------------------------------------------------- experiments
async def e1(reps):
    """E1: single request at a time, distributed vs baseline."""
    ref = reference()
    rows = []
    with cluster("e1", CONFIGS["pipeline_microbatch"]) as c:
        await run_workload(c.addr, ref["prompts"][:2], 4, concurrency=1)  # warm-up
        for rep in range(reps):
            m, _ = await measured(c, ref["prompts"], N, concurrency=1)
            m.update({"system": "distributed", "rep": rep})
            rows.append(m)
            log("e1 rep", rep, f"itl={m['itl_ms_mean']:.2f}ms correct={m['correct_rate']:.0%}")
    return rows


async def e2(reps):
    """E2: KV cache on vs off (sequential requests)."""
    ref = reference()
    rows = []
    for mode, extra in (("cache_on", []), ("cache_off", ["--no-kv-cache"])):
        with cluster(f"e2_{mode}", CONFIGS["pipeline_microbatch"] + extra) as c:
            await run_workload(c.addr, ref["prompts"][:2], 4, concurrency=1)
            for rep in range(reps):
                m, results = await measured(c, ref["prompts"], N, concurrency=1)
                m.update({"mode": mode, "rep": rep})
                rows.append(m)
                # per-prompt rows for the ITL-vs-length plot
                for r, e in zip(results, ref["prompts"]):
                    rows.append({"mode": mode, "rep": rep, "per_prompt": 1, "prompt_len": e["prompt_len"],
                                 "itl_ms_mean": sum(r.itl_ms) / len(r.itl_ms),
                                 "match": int(r.tokens == e["output_token_ids"])})
                log("e2", mode, rep, f"itl={m['itl_ms_mean']:.2f}ms correct={m['correct_rate']:.0%}")
    return rows


async def e3(reps, quick):
    """E3: 1..64 concurrent requests, unbatched vs micro-batched pipeline."""
    sweeps = {"pipeline_microbatch": [1, 2, 4, 8, 16, 32, 50, 64] if not quick else [1, 8, 50],
              "pipeline_unbatched": [1, 4, 16, 50] if not quick else [1, 50]}
    rows = []
    for name, concs in sweeps.items():
        with cluster(f"e3_{name}", CONFIGS[name] + ["--admission-queue", "128", "--max-active", "64"]) as c:
            await run_workload(c.addr, entries_for(4), 4)
            for conc in concs:
                for rep in range(reps):
                    entries = entries_for(conc)
                    m, _ = await measured(c, entries, N)
                    m.update({"system": name, "concurrency": conc, "rep": rep})
                    rows.append(m)
                    log("e3", name, conc, rep, f"{m['throughput_tok_s']:.1f} tok/s correct={m['correct_rate']:.0%}")
    return rows


async def e4(reps, quick):
    """E4: batch size sweep 1..16, with and without pipelining (2 batches in flight)."""
    rows = []
    conc = 32
    for inflight in (1, 2):
        for bs in [1, 2, 4, 8, 16]:
            extra = ["--max-batch-size", str(bs), "--max-inflight", str(inflight), "--max-wait-ms", "2"]
            with cluster(f"e4_b{bs}_i{inflight}", extra) as c:
                await run_workload(c.addr, entries_for(4), 4)
                for rep in range(reps):
                    m, _ = await measured(c, entries_for(conc), N)
                    m.update({"batch_size": bs, "pipelined": int(inflight > 1), "inflight": inflight,
                              "concurrency": conc, "rep": rep})
                    rows.append(m)
                    log("e4", bs, inflight, rep, f"{m['throughput_tok_s']:.1f} tok/s "
                        f"util s0={m.get('s0_utilisation', 0):.2f} s1={m.get('s1_utilisation', 0):.2f}")
    return rows


async def e9(reps, quick):
    """E9: artificial network delay on the stage0 -> stage1 link."""
    delays = [0, 1, 2, 5, 10, 20] if not quick else [0, 5, 20]
    rows = []
    for inflight in (1, 4):
        extra = ["--max-batch-size", "8", "--max-inflight", str(inflight)]
        with cluster(f"e9_i{inflight}", extra) as c:
            await run_workload(c.addr, entries_for(4), 4)
            async with __import__("grpc").aio.insecure_channel(c.worker_addr(0)) as ch:
                stub = pb_grpc.StageServiceStub(ch)
                for d in delays:
                    await stub.Configure(pb.JsonRequest(json=json.dumps({"link_delay_ms": d})), timeout=5)
                    for rep in range(reps):
                        single, _ = await measured(c, reference()["prompts"][:6], N, concurrency=1)
                        conc, _ = await measured(c, entries_for(32), N)
                        rows.append({"delay_ms": d, "inflight": inflight, "rep": rep,
                                     "single_itl_ms_mean": single["itl_ms_mean"],
                                     "single_itl_ms_p95": single["itl_ms_p95"],
                                     "single_ttft_ms_mean": single["ttft_ms_mean"],
                                     "conc32_throughput_tok_s": conc["throughput_tok_s"],
                                     "conc32_itl_ms_p50": conc["itl_ms_p50"],
                                     "correct_rate": min(single["correct_rate"], conc["correct_rate"])})
                        log("e9", inflight, d, rep, f"itl={single['itl_ms_mean']:.1f}ms "
                            f"thr={conc['throughput_tok_s']:.1f}")
    return rows


async def plen(reps, model):
    """Vary prompt length: TTFT / ITL / bytes for baseline vs distributed (synthetic prompts)."""
    from baseline import greedy_generate

    base_text_ids = []
    for e in reference()["prompts"]:
        base_text_ids += e["prompt_token_ids"]
    lengths = [8, 64, 256, 512]
    entries = []
    for L in lengths:
        ids = (base_text_ids * (L // len(base_text_ids) + 1))[:L]
        out, _ = greedy_generate(model, ids, N)
        entries.append({"id": L, "prompt_token_ids": ids, "prompt_len": L, "output_token_ids": out})
    rows = []
    for e in entries:
        for rep in range(reps):
            out, lat = greedy_generate(model, e["prompt_token_ids"], N)
            rows.append({"system": "baseline", "prompt_len": e["prompt_len"], "rep": rep,
                         "ttft_ms": lat[0] * 1e3, "itl_ms": sum(lat[1:]) / (N - 1) * 1e3,
                         "correct": int(out == e["output_token_ids"])})
    with cluster("plen", CONFIGS["pipeline_microbatch"]) as c:
        await run_workload(c.addr, entries[:1], 4)
        for e in entries:
            for rep in range(reps):
                m, results = await measured(c, [e], N, concurrency=1)
                r = results[0]
                rows.append({"system": "distributed", "prompt_len": e["prompt_len"], "rep": rep,
                             "ttft_ms": r.ttft_ms, "itl_ms": sum(r.itl_ms) / len(r.itl_ms),
                             "correct": int(r.tokens == e["output_token_ids"]),
                             "wire_bytes_per_request": m["wire_bytes"],
                             "activation_bytes_per_request": m["activation_bytes"]})
        log("plen done")
    return rows


async def fp16(reps):
    """Stretch: fp16 activation compression on the stage0 -> stage1 link."""
    ref = reference()
    rows = []
    for dt in ("float32", "float16"):
        with cluster(f"fp16_{dt}", CONFIGS["pipeline_microbatch"] + ["--activation-dtype", dt]) as c:
            for rep in range(reps):
                m, results = await measured(c, ref["prompts"], N, concurrency=1)
                first_div = []
                for r, e in zip(results, ref["prompts"]):
                    k = next((i for i, (a, b) in enumerate(zip(r.tokens, e["output_token_ids"])) if a != b), N)
                    first_div.append(k)
                token_acc = sum(sum(a == b for a, b in zip(r.tokens, e["output_token_ids"]))
                                for r, e in zip(results, ref["prompts"])) / (N * len(results))
                rows.append({"activation_dtype": dt, "rep": rep, "correct_rate": m["correct_rate"],
                             "token_accuracy": token_acc, "mean_first_divergence": sum(first_div) / len(first_div),
                             "activation_bytes_per_token": m["activation_bytes_per_token"],
                             "wire_bytes_per_token": m["wire_bytes_per_token"],
                             "itl_ms_mean": m["itl_ms_mean"], "ttft_ms_mean": m["ttft_ms_mean"]})
                log("fp16", dt, rep, f"exact={m['correct_rate']:.0%} bytes/token={m['activation_bytes_per_token']:.0f}")
    return rows


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="1 repetition and fewer sweep points")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--only", default="", help="comma separated subset, e.g. e1,e3,e5")
    a = ap.parse_args(argv)
    reps = a.reps or (1 if a.quick else 3)
    only = set(x.strip() for x in a.only.split(",") if x.strip())
    want = lambda k: not only or k in only  # noqa: E731
    ensure_dirs()
    t_start = time.time()
    (RESULTS_DIR / "env.json").write_text(json.dumps(env_info() | {"reps": reps, "quick": a.quick}, indent=2))

    model = None
    if want("baseline") or want("e1") or want("e3") or want("e4") or want("plen"):
        log("baseline (single process) ...")
        rows, model = run_baseline(reps)
        rows += run_split_inprocess(reps)
        write_csv("baseline.csv", rows)
    if want("e1"):
        write_csv("e1_single_request.csv", asyncio.run(e1(reps)))
    if want("e2"):
        write_csv("e2_kv_cache.csv", asyncio.run(e2(reps)))
    if want("e3"):
        write_csv("e3_concurrency.csv", asyncio.run(e3(reps, a.quick)))
    if want("e4"):
        write_csv("e4_batch_sweep.csv", asyncio.run(e4(reps, a.quick)))
    if want("e5"):
        from scripts.load_test import main as load_main

        load_main(["--requests", "250" if a.quick else "400", "--rate", "40"])
    if want("e6"):
        from scripts.fault_inject import main as fault_main

        fault_main(["kill", "--reps", str(reps)])
    if want("e8"):
        from scripts.fault_inject import main as fault_main

        fault_main(["stale"])
    if want("e9"):
        write_csv("e9_network_delay.csv", asyncio.run(e9(reps, a.quick)))
    if want("plen"):
        if model is None:
            _, model = run_baseline(0)
        write_csv("prompt_length.csv", asyncio.run(plen(reps, model)))
    if want("fp16"):
        write_csv("fp16_compression.csv", asyncio.run(fp16(reps)))

    from benchmarks.plots import main as plots_main

    plots_main([])
    log(f"all benchmarks finished in {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
