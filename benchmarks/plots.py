"""Regenerate every plot and benchmarks/results/summary.md from the raw CSV/JSON results.

    python -m benchmarks.plots
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks.common import PLOTS_DIR, RESULTS_DIR, aggregate, ensure_dirs, read_csv  # noqa: E402

# Reference categorical palette (fixed order) and neutral tokens, light mode.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 5,
    "font.family": "DejaVu Sans",
})


def _save(fig, name):
    fig.tight_layout()
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def _mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not math.isnan(r[key])]
    return statistics.fmean(vals) if vals else math.nan


def _std(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not math.isnan(r[key])]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _bar_labels(ax, bars, fmt="{:.1f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK2)


# --------------------------------------------------------------------------- plots
def plot_e1():
    base, e1 = read_csv("baseline.csv"), read_csv("e1_single_request.csv")
    if not base or not e1:
        return None
    systems = [("baseline (1 process)", [r for r in base if r["system"] == "baseline"]),
               ("split, no RPC", [r for r in base if r["system"] == "split_inprocess"]),
               ("distributed (gRPC)", e1)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, title in ((axes[0], "ttft_ms_mean", "Time to first token"),
                           (axes[1], "itl_ms_mean", "Inter-token latency")):
        vals = [_mean(rs, key) for _, rs in systems]
        errs = [_std(rs, key) for _, rs in systems]
        bars = ax.bar(range(3), vals, yerr=errs, color=[S1, S2, S3], width=0.6, capsize=3,
                      edgecolor=SURFACE, linewidth=2)
        _bar_labels(ax, bars)
        ax.set_xticks(range(3), [s for s, _ in systems])
        ax.set_ylabel("ms (mean over 24 prompts)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    _save(fig, "e1_latency.png")


def plot_e2():
    rows = [r for r in read_csv("e2_kv_cache.csv") if r.get("per_prompt") == 1]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for mode, color, label in (("cache_on", S1, "KV cache on"), ("cache_off", S2, "KV cache off")):
        pts = {}
        for r in rows:
            if r["mode"] == mode:
                pts.setdefault(r["prompt_len"], []).append(r["itl_ms_mean"])
        xs = sorted(pts)
        ax.plot(xs, [statistics.fmean(pts[x]) for x in xs], "o-", color=color, label=label)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("mean inter-token latency (ms)")
    ax.set_title("E2 - decode latency with and without KV cache", loc="left")
    ax.legend()
    _save(fig, "e2_kv_cache.png")


def plot_e3():
    rows, base = read_csv("e3_concurrency.csv"), read_csv("baseline.csv")
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, color, label in (("pipeline_microbatch", S1, "pipeline + micro-batching"),
                               ("pipeline_unbatched", S2, "pipeline, unbatched")):
        agg = sorted(aggregate([r for r in rows if r["system"] == name],
                               ["throughput_tok_s", "itl_ms_p95"], ["concurrency"]), key=lambda r: r["concurrency"])
        xs = [r["concurrency"] for r in agg]
        axes[0].errorbar(xs, [r["throughput_tok_s"] for r in agg], yerr=[r["throughput_tok_s_std"] for r in agg],
                         fmt="o-", color=color, label=label, capsize=3)
        axes[1].plot(xs, [r["itl_ms_p95"] for r in agg], "o-", color=color, label=label)
    b = [r for r in base if r["system"] == "baseline"]
    if b:
        axes[0].axhline(_mean(b, "throughput_tok_s"), color=INK2, lw=1.2, ls="--", label="single-process baseline")
    axes[0].set_title("E3 - throughput vs concurrency", loc="left")
    axes[0].set_ylabel("generated tokens / s")
    axes[1].set_title("E3 - p95 inter-token latency", loc="left")
    axes[1].set_ylabel("ms")
    for ax in axes:
        ax.set_xlabel("concurrent requests")
        ax.set_xscale("log", base=2)
        ax.legend()
    _save(fig, "e3_throughput_vs_concurrency.png")


def plot_e4():
    rows = read_csv("e4_batch_sweep.csv")
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for pipelined, color, label in ((0, S2, "1 batch in flight"), (1, S1, "2 batches in flight (pipelined)")):
        agg = sorted(aggregate([r for r in rows if r["pipelined"] == pipelined],
                               ["throughput_tok_s", "itl_ms_p50", "itl_ms_p95", "s0_utilisation", "s1_utilisation"],
                               ["batch_size"]), key=lambda r: r["batch_size"])
        xs = [r["batch_size"] for r in agg]
        axes[0].errorbar(xs, [r["throughput_tok_s"] for r in agg], yerr=[r["throughput_tok_s_std"] for r in agg],
                         fmt="o-", color=color, label=label, capsize=3)
        axes[1].plot(xs, [r["itl_ms_p50"] for r in agg], "o-", color=color, label=label + ", p50")
        axes[1].plot(xs, [r["itl_ms_p95"] for r in agg], "o--", color=color, lw=1.2, label=label + ", p95")
        util = [(r["s0_utilisation"] + r["s1_utilisation"]) / 2 * 100 for r in agg]
        axes[2].plot(xs, util, "o-", color=color, label=label)
    axes[0].set_title("E4 - throughput vs batch size", loc="left")
    axes[0].set_ylabel("generated tokens / s")
    axes[1].set_title("E4 - inter-token latency", loc="left")
    axes[1].set_ylabel("ms")
    axes[2].set_title("E4 - mean stage utilisation", loc="left")
    axes[2].set_ylabel("% of wall time computing")
    axes[2].set_ylim(0, 100)
    for ax in axes:
        ax.set_xlabel("max micro-batch size (32 concurrent requests)")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.legend()
    _save(fig, "e4_batch_sweep.png")

    # Bottleneck breakdown per batch size (non-pipelined runs: no overlap muddles the split)
    agg = sorted(aggregate([r for r in rows if r["pipelined"] == 0],
                           ["bd_compute_ms", "bd_serde_ms", "bd_queue_ms", "bd_rpc_overhead_ms"], ["batch_size"]),
                 key=lambda r: r["batch_size"])
    if agg and not math.isnan(agg[0]["bd_compute_ms"]):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        xs = range(len(agg))
        bottom = [0.0] * len(agg)
        for key, color, label in (("bd_compute_ms", S1, "compute (both stages)"),
                                  ("bd_serde_ms", S2, "(de)serialisation"),
                                  ("bd_queue_ms", S3, "worker queueing"),
                                  ("bd_rpc_overhead_ms", S4, "RPC + network + coordinator")):
            vals = [r[key] for r in agg]
            ax.bar(xs, vals, bottom=bottom, color=color, label=label, width=0.6, edgecolor=SURFACE, linewidth=2)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(list(xs), [str(r["batch_size"]) for r in agg])
        ax.set_xlabel("micro-batch size")
        ax.set_ylabel("ms per pipeline step (mean)")
        ax.set_title("Where a decode step's time goes", loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend()
        _save(fig, "bottleneck_breakdown.png")


def plot_overload(tag="e5_overload"):
    rows = read_csv(f"{tag}_timeseries.csv")
    if not rows:
        return
    t = [r["t_s"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axes[0]
    ax.plot(t, [r["admission_depth"] for r in rows], color=S1, label="admission queue depth")
    ax.plot(t, [r["active"] for r in rows], color=S2, label="active requests")
    ax.axhline(rows[0]["admission_capacity"], color=S1, lw=1, ls="--", label="admission capacity")
    ax.axhline(rows[0]["active_capacity"], color=S2, lw=1, ls=":", label="active capacity")
    ax.set_title("E5 - coordinator queues", loc="left")
    ax.set_ylabel("requests")
    ax = axes[1]
    ax.plot(t, [r["rejected_total"] for r in rows], color=S1, label="rejected (OVERLOADED)")
    ax.plot(t, [r["completed_total"] for r in rows], color=S2, label="completed")
    ax.set_title("E5 - cumulative outcomes", loc="left")
    ax.set_ylabel("requests")
    ax = axes[2]
    for key, color, label in (("coordinator_rss_mb", S1, "coordinator"), ("s0_rss_mb", S2, "worker 1 (stage 0)"),
                              ("s1_rss_mb", S3, "worker 2 (stage 1)")):
        if key in rows[0]:
            ax.plot(t, [r[key] for r in rows], color=color, label=label)
    ax.set_ylim(bottom=0)
    ax.set_title("E5 - resident memory", loc="left")
    ax.set_ylabel("MiB")
    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.legend()
    _save(fig, f"{tag}.png")


def plot_recovery_timeline():
    tl, marks = read_csv("e6_e7_timeline.csv"), read_csv("e6_e7_marks.csv")
    if not tl:
        return
    stages = sorted({r["stage"] for r in tl})
    fig, axes = plt.subplots(1, len(stages), figsize=(6.5 * len(stages), 3.8), squeeze=False)
    for ax, st in zip(axes[0], stages):
        rows = [r for r in tl if r["stage"] == st and r["rep"] == 0]
        for req in sorted({r["request"] for r in rows}):
            pts = sorted((r["t_ms"], r["step"]) for r in rows if r["request"] == req)
            ax.plot([p[0] for p in pts], [p[1] + 1 for p in pts], color=S1, lw=1, alpha=0.55)
        mk = next((m for m in marks if m["stage"] == st and m["rep"] == 0), None)
        if mk:
            for key, color, label in (("kill_ms", S2, "SIGKILL"), ("detect_ms", S4, "declared dead"),
                                      ("restarted_ms", S3, "new epoch healthy"),
                                      ("replayed_ms", INK2, "caches rebuilt")):
                ax.axvline(mk[key], color=color, lw=1.5, ls="--", label=f"{label} ({mk[key]:.0f} ms)")
        ax.set_title(f"Kill worker {st + 1} (stage {st}) mid-generation", loc="left")
        ax.set_xlabel("time relative to SIGKILL (ms)")
        ax.set_ylabel("tokens received per request")
        ax.legend(loc="lower right")
    _save(fig, "e6_e7_recovery_timeline.png")


def plot_e9():
    rows = read_csv("e9_network_delay.csv")
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for inflight, color in ((1, S2), (4, S1)):
        agg = sorted(aggregate([r for r in rows if r["inflight"] == inflight],
                               ["single_itl_ms_mean", "conc32_throughput_tok_s"], ["delay_ms"]),
                     key=lambda r: r["delay_ms"])
        xs = [r["delay_ms"] for r in agg]
        if inflight == 1:
            axes[0].plot(xs, [r["single_itl_ms_mean"] for r in agg], "o-", color=S1, label="measured")
            axes[0].plot(xs, [agg[0]["single_itl_ms_mean"] + x for x in xs], ":", color=INK2, lw=1.2,
                         label="no-delay latency + delay")
        axes[1].errorbar(xs, [r["conc32_throughput_tok_s"] for r in agg],
                         yerr=[r["conc32_throughput_tok_s_std"] for r in agg], fmt="o-", color=color, capsize=3,
                         label=f"{inflight} micro-batch{'es' if inflight > 1 else ''} in flight")
    axes[0].set_title("E9 - single request inter-token latency", loc="left")
    axes[0].set_ylabel("ms")
    axes[1].set_title("E9 - throughput, 32 concurrent requests", loc="left")
    axes[1].set_ylabel("generated tokens / s")
    for ax in axes:
        ax.set_xlabel("one-way delay on stage0 -> stage1 link (ms)")
        ax.legend()
    _save(fig, "e9_network_delay.png")


def plot_plen():
    rows = read_csv("prompt_length.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for system, color in (("baseline", S1), ("distributed", S2)):
        agg = sorted(aggregate([r for r in rows if r["system"] == system], ["ttft_ms"], ["prompt_len"]),
                     key=lambda r: r["prompt_len"])
        ax.errorbar([r["prompt_len"] for r in agg], [r["ttft_ms"] for r in agg],
                    yerr=[r["ttft_ms_std"] for r in agg], fmt="o-", color=color, capsize=3, label=system)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("time to first token (ms)")
    ax.set_title("Prefill cost vs prompt length", loc="left")
    ax.legend()
    _save(fig, "prompt_length_ttft.png")


# --------------------------------------------------------------------------- summary
def _f(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def summary_md() -> str:
    out = ["# Benchmark summary", "",
           "Generated by `python -m benchmarks.plots` from the CSV/JSON files in this directory. "
           "Each configuration was run `reps` times (see env.json); values are means across repetitions "
           "(± standard deviation where shown). ITL = inter-token latency, TTFT = time to first token.", ""]
    env = RESULTS_DIR / "env.json"
    if env.exists():
        e = json.loads(env.read_text())
        out += [f"Environment: {e.get('machine')} / {e.get('platform')}, {e.get('cpus')} CPUs, "
                f"torch {e.get('torch')}, {e.get('threads_per_process')} torch threads per process, "
                f"reps={e.get('reps')}.", ""]

    base, e1 = read_csv("baseline.csv"), read_csv("e1_single_request.csv")
    if base and e1:
        out += ["## E1 - single request: distributed vs baseline", "",
                "| system | TTFT mean (ms) | ITL mean (ms) | ITL p50 | ITL p95 | ITL p99 | throughput (tok/s) | exact match |",
                "|---|---|---|---|---|---|---|---|"]
        for name, rs in (("baseline (HF, 1 process)", [r for r in base if r["system"] == "baseline"]),
                         ("split stages, 1 process, no RPC", [r for r in base if r["system"] == "split_inprocess"]),
                         ("distributed (2 workers, gRPC)", e1)):
            out.append(f"| {name} | {_f(_mean(rs, 'ttft_ms_mean'))} ± {_f(_std(rs, 'ttft_ms_mean'))} | "
                       f"{_f(_mean(rs, 'itl_ms_mean'), 2)} ± {_f(_std(rs, 'itl_ms_mean'), 2)} | "
                       f"{_f(_mean(rs, 'itl_ms_p50'), 2)} | {_f(_mean(rs, 'itl_ms_p95'), 2)} | "
                       f"{_f(_mean(rs, 'itl_ms_p99'), 2)} | {_f(_mean(rs, 'throughput_tok_s'))} | "
                       f"{_f(100 * _mean(rs, 'correct_rate'), 0)}% |")
        b = [r for r in base if r["system"] == "baseline"]
        ov = _mean(e1, "itl_ms_mean") - _mean(b, "itl_ms_mean")
        out += ["", f"Distributed overhead per decoded token: **{ov:+.2f} ms** "
                    f"({_mean(e1, 'itl_ms_mean') / _mean(b, 'itl_ms_mean'):.2f}x the baseline ITL). "
                    f"Activation bytes per decoded token on the stage0->stage1 link: "
                    f"{_f(_mean(e1, 'activation_bytes_per_token'), 0)} B; all wire bytes per token "
                    f"(every hop, incl. protobuf framing): {_f(_mean(e1, 'wire_bytes_per_token'), 0)} B.", ""]
        bd = {k: _mean(e1, k) for k in ("bd_rpc_ms", "bd_compute_ms", "bd_serde_ms", "bd_queue_ms",
                                         "bd_rpc_overhead_ms")}
        if not math.isnan(bd["bd_rpc_ms"]):
            out += [f"Mean pipeline step as seen by the coordinator: {bd['bd_rpc_ms']:.2f} ms = compute "
                    f"{bd['bd_compute_ms']:.2f} ms + (de)serialisation {bd['bd_serde_ms']:.3f} ms + worker "
                    f"queueing {bd['bd_queue_ms']:.3f} ms + RPC/network/event-loop overhead "
                    f"{bd['bd_rpc_overhead_ms']:.2f} ms.", ""]

    e2 = [r for r in read_csv("e2_kv_cache.csv") if r.get("per_prompt") != 1]
    if e2:
        on, off = [r for r in e2 if r["mode"] == "cache_on"], [r for r in e2 if r["mode"] == "cache_off"]
        out += ["## E2 - KV cache on vs off", "",
                "| mode | ITL mean (ms) | ITL p95 (ms) | TTFT mean (ms) | exact match |", "|---|---|---|---|---|"]
        for name, rs in (("cache on", on), ("cache off", off)):
            out.append(f"| {name} | {_f(_mean(rs, 'itl_ms_mean'), 2)} ± {_f(_std(rs, 'itl_ms_mean'), 2)} | "
                       f"{_f(_mean(rs, 'itl_ms_p95'), 2)} | {_f(_mean(rs, 'ttft_ms_mean'))} | "
                       f"{_f(100 * _mean(rs, 'correct_rate'), 0)}% |")
        out += ["", f"Speed-up of the KV cache on per-token decode latency: "
                    f"**{_mean(off, 'itl_ms_mean') / _mean(on, 'itl_ms_mean'):.2f}x**.", ""]

    e3 = read_csv("e3_concurrency.csv")
    if e3:
        out += ["## E3 - concurrency sweep", "",
                "| system | concurrent | throughput (tok/s) | TTFT p50 (ms) | ITL p50 (ms) | ITL p95 (ms) | ITL p99 (ms) | mean batch | exact match |",
                "|---|---|---|---|---|---|---|---|---|"]
        for name in ("pipeline_unbatched", "pipeline_microbatch"):
            agg = sorted(aggregate([r for r in e3 if r["system"] == name],
                                   ["throughput_tok_s", "ttft_ms_p50", "itl_ms_p50", "itl_ms_p95", "itl_ms_p99",
                                    "mean_batch_size", "correct_rate"], ["concurrency"]),
                         key=lambda r: r["concurrency"])
            for r in agg:
                out.append(f"| {name} | {r['concurrency']} | {_f(r['throughput_tok_s'])} ± "
                           f"{_f(r['throughput_tok_s_std'])} | {_f(r['ttft_ms_p50'])} | {_f(r['itl_ms_p50'], 2)} | "
                           f"{_f(r['itl_ms_p95'], 2)} | {_f(r['itl_ms_p99'], 2)} | {_f(r['mean_batch_size'], 2)} | "
                           f"{_f(100 * r['correct_rate'], 0)}% |")
        b = [r for r in base if r["system"] == "baseline"] if base else []
        if b:
            out += ["", f"Single-process baseline throughput (sequential): {_f(_mean(b, 'throughput_tok_s'))} tok/s.", ""]

    e4 = read_csv("e4_batch_sweep.csv")
    if e4:
        out += ["## E4 - micro-batch size sweep (32 concurrent requests)", "",
                "| batch | pipelined | throughput (tok/s) | ITL p50 (ms) | ITL p95 (ms) | stage 0 util | stage 1 util | exact match |",
                "|---|---|---|---|---|---|---|---|"]
        agg = sorted(aggregate(e4, ["throughput_tok_s", "itl_ms_p50", "itl_ms_p95", "s0_utilisation",
                                    "s1_utilisation", "correct_rate"], ["batch_size", "pipelined"]),
                     key=lambda r: (r["pipelined"], r["batch_size"]))
        for r in agg:
            out.append(f"| {r['batch_size']} | {'yes' if r['pipelined'] else 'no'} | {_f(r['throughput_tok_s'])} ± "
                       f"{_f(r['throughput_tok_s_std'])} | {_f(r['itl_ms_p50'], 2)} | {_f(r['itl_ms_p95'], 2)} | "
                       f"{_f(100 * r['s0_utilisation'], 0)}% | {_f(100 * r['s1_utilisation'], 0)}% | "
                       f"{_f(100 * r['correct_rate'], 0)}% |")
        unb = [r for r in agg if r["batch_size"] == 1 and not r["pipelined"]]
        best = max(agg, key=lambda r: r["throughput_tok_s"])
        if unb:
            out += ["", f"Best configuration: batch {best['batch_size']}, pipelined={bool(best['pipelined'])}: "
                        f"**{best['throughput_tok_s'] / unb[0]['throughput_tok_s']:.2f}x** the throughput of the "
                        f"unbatched, unpipelined pipeline.", ""]
        nb = sorted(aggregate([r for r in e4 if r["pipelined"] == 0],
                              ["bd_rpc_ms", "bd_compute_ms", "bd_serde_ms", "bd_queue_ms", "bd_rpc_overhead_ms"],
                              ["batch_size"]), key=lambda r: r["batch_size"])
        if nb and not math.isnan(nb[0]["bd_rpc_ms"]):
            out += ["### Bottleneck breakdown (non-pipelined runs, mean per pipeline step)", "",
                    "| batch | step (ms) | compute | (de)serialisation | queueing | RPC/network/other |",
                    "|---|---|---|---|---|---|"]
            for r in nb:
                tot = r["bd_rpc_ms"]
                out.append(f"| {r['batch_size']} | {_f(tot, 2)} | {_f(r['bd_compute_ms'], 2)} "
                           f"({_f(100 * r['bd_compute_ms'] / tot, 0)}%) | {_f(r['bd_serde_ms'], 3)} "
                           f"({_f(100 * r['bd_serde_ms'] / tot, 1)}%) | {_f(r['bd_queue_ms'], 3)} | "
                           f"{_f(r['bd_rpc_overhead_ms'], 2)} ({_f(100 * r['bd_rpc_overhead_ms'] / tot, 0)}%) |")
            out.append("")

    p5 = RESULTS_DIR / "e5_overload_summary.json"
    if p5.exists():
        s = json.loads(p5.read_text())
        out += ["## E5 - sustained overload", "",
                f"Offered load {s['offered_rate_rps']:.0f} req/s for {s['requests']} requests "
                f"({s['max_new_tokens']} tokens each). Outcomes: {s['statuses']}; rejection rate "
                f"**{100 * s['rejection_rate']:.1f}%**; completed requests matching the reference: "
                f"{s['completed_correct']}/{s['completed']}; hung requests: {s['hung']}. "
                f"Rejections were answered in {s['mean_rejection_latency_ms']:.1f} ms on average "
                f"(max {s['max_rejection_latency_ms']:.1f} ms).", "",
                f"Queue bounds held: admission depth max {s['max_admission_depth']} / capacity "
                f"{s['admission_capacity']}, active max {s['max_active']} / {s['active_capacity']}, worker "
                f"input queue max {s['worker_max_queue_depth']} / {s['worker_queue_size']}. Resident memory "
                f"(min-max MiB during the run): " + ", ".join(
                    f"{k.replace('_rss_mb', '')} {v[0]:.0f}-{v[1]:.0f}" for k, v in s["rss_mb_min_max"].items() if v)
                + ".", ""]

    p6 = RESULTS_DIR / "e6_e8_fault_summary.json"
    if p6.exists():
        s = json.loads(p6.read_text())
        if s.get("kills"):
            out += ["## E6 / E7 - worker crash (SIGKILL) mid-generation", "",
                    "| killed | rep | exact match | dup tokens | missing tokens | kill->detect (ms) | restart (ms) | replay (ms) | total recovery (ms) |",
                    "|---|---|---|---|---|---|---|---|---|"]
            for r in s["kills"]:
                out.append(f"| worker {r['stage'] + 1} (stage {r['stage']}) | {r['rep']} | {r['correct']}/{r['requests']} | "
                           f"{r['duplicate_tokens']} | {r['missing_tokens']} | {_f(r['detect_ms'], 0)} | "
                           f"{_f(r['restart_ms'], 0)} | {_f(r['replay_ms'], 0)} | {_f(r['total_recovery_ms'], 0)} |")
            for st in (0, 1):
                rs = [r for r in s["kills"] if r["stage"] == st and r.get("total_recovery_ms")]
                if rs:
                    out.append(f"\nWorker {st + 1}: mean total recovery {statistics.fmean(r['total_recovery_ms'] for r in rs):.0f} ms "
                               f"(detection {statistics.fmean(r['detect_ms'] for r in rs):.0f} + restart "
                               f"{statistics.fmean(r['restart_ms'] for r in rs):.0f} + replay "
                               f"{statistics.fmean(r['replay_ms'] for r in rs):.0f}).")
            out.append("")
        if s.get("e8_stale_epoch") or s.get("stale_epoch_injection"):
            out += ["## E8 - stale-epoch and duplicate messages", ""]
            for st in s.get("stale_epoch_injection", []) + ([s["e8_stale_epoch"]] if s.get("e8_stale_epoch") else []):
                out.append(f"- Message stamped with pre-crash epochs {st['injected_epochs']} sent after recovery "
                           f"(current {st['current_epochs']}): **{st['status']}** - {'rejected' if st['rejected'] else 'ACCEPTED (bug)'}.")
            d = s.get("e8_direct_duplicates")
            if d:
                out.append(f"- Every decode step delivered twice straight to the workers: duplicates answered from the "
                           f"idempotency cache = {d['all_duplicates_idempotent']}; token stream matches reference = "
                           f"{d['match']}; a late older step was answered with **{d['late_old_step_status']}**.")
            dd = s.get("e8_duplicate_dispatch")
            if dd:
                out.append(f"- Coordinator re-sent every micro-batch ({dd['duplicates_sent']} duplicates): "
                           f"{dd['discarded_duplicate_results']} duplicate results discarded, "
                           f"{dd['duplicate_tokens']} duplicate tokens emitted, {dd['correct']}/{dd['requests']} "
                           f"responses exact; worker idempotent replays {dd['worker_idempotent_replays']}.")
            out.append("")

    e9 = read_csv("e9_network_delay.csv")
    if e9:
        out += ["## E9 - artificial delay between workers", "",
                "| delay (ms) | in flight | single-request ITL (ms) | throughput @32 (tok/s) | exact match |",
                "|---|---|---|---|---|"]
        for r in sorted(aggregate(e9, ["single_itl_ms_mean", "conc32_throughput_tok_s", "correct_rate"],
                                  ["delay_ms", "inflight"]), key=lambda r: (r["inflight"], r["delay_ms"])):
            out.append(f"| {r['delay_ms']} | {r['inflight']} | {_f(r['single_itl_ms_mean'], 2)} | "
                       f"{_f(r['conc32_throughput_tok_s'])} | {_f(100 * r['correct_rate'], 0)}% |")
        out.append("")

    pl = read_csv("prompt_length.csv")
    if pl:
        out += ["## Prompt length", "", "| prompt tokens | system | TTFT (ms) | ITL (ms) | wire bytes / request | exact match |",
                "|---|---|---|---|---|---|"]
        for r in sorted(aggregate(pl, ["ttft_ms", "itl_ms", "wire_bytes_per_request", "correct"],
                                  ["prompt_len", "system"]), key=lambda r: (r["prompt_len"], r["system"])):
            out.append(f"| {r['prompt_len']} | {r['system']} | {_f(r['ttft_ms'])} | {_f(r['itl_ms'], 2)} | "
                       f"{_f(r['wire_bytes_per_request'], 0)} | {_f(100 * r['correct'], 0)}% |")
        out.append("")

    fp = read_csv("fp16_compression.csv")
    if fp:
        out += ["## Stretch: fp16 activation compression", "",
                "| wire dtype | activation bytes / token | ITL (ms) | exact-match responses | token accuracy | mean first divergence (token idx) |",
                "|---|---|---|---|---|---|"]
        for r in aggregate(fp, ["activation_bytes_per_token", "itl_ms_mean", "correct_rate", "token_accuracy",
                                "mean_first_divergence"], ["activation_dtype"]):
            out.append(f"| {r['activation_dtype']} | {_f(r['activation_bytes_per_token'], 0)} | {_f(r['itl_ms_mean'], 2)} | "
                       f"{_f(100 * r['correct_rate'], 0)}% | {_f(100 * r['token_accuracy'], 1)}% | "
                       f"{_f(r['mean_first_divergence'], 1)} |")
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ensure_dirs()
    for fn in (plot_e1, plot_e2, plot_e3, plot_e4, plot_overload, plot_recovery_timeline, plot_e9, plot_plen):
        try:
            fn()
        except Exception as e:  # keep going: one missing result should not block the rest
            print(f"plot {fn.__name__} failed: {e!r}")
    md = summary_md()
    (RESULTS_DIR / "summary.md").write_text(md)
    print("wrote", RESULTS_DIR / "summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
