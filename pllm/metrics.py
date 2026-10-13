"""Small metric helpers shared by the coordinator, benchmarks and tests."""

from __future__ import annotations

import collections
import math
import statistics


class Counter(collections.Counter):
    """A Counter whose missing keys read as 0 (collections.Counter already does) - kept as a named type."""


def percentile(values, q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]); NaN for an empty input."""
    xs = sorted(values)
    if not xs:
        return math.nan
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def summarize(values, prefix: str = "") -> dict:
    xs = list(values)
    if not xs:
        return {f"{prefix}mean": math.nan, f"{prefix}p50": math.nan, f"{prefix}p95": math.nan,
                f"{prefix}p99": math.nan, f"{prefix}n": 0}
    return {
        f"{prefix}mean": statistics.fmean(xs),
        f"{prefix}p50": percentile(xs, 50),
        f"{prefix}p95": percentile(xs, 95),
        f"{prefix}p99": percentile(xs, 99),
        f"{prefix}n": len(xs),
    }


def inter_token_gaps(times) -> list[float]:
    return [b - a for a, b in zip(times, times[1:])]
