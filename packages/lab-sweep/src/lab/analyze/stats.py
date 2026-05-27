"""Pass@k / pass^k / bootstrap CIs."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def pass_at_1(passes: Sequence[bool]) -> float:
    """Mean pass rate over the seeds."""
    if not passes:
        return 0.0
    return float(sum(1 for p in passes if p)) / len(passes)


def pass_caret_k(passes: Sequence[bool], k: int) -> float:
    """pass^k — empirical probability that ALL of k random draws (without replacement) pass.

    For a (model, task) cell with N seeds where M pass:
        pass^k = C(M, k) / C(N, k)   when k <= N, else 0.
    """
    n = len(passes)
    if k <= 0 or k > n:
        return 0.0
    m = sum(1 for p in passes if p)
    if m < k:
        return 0.0
    return math.comb(m, k) / math.comb(n, k)


def bootstrap_ci(
    passes: Sequence[bool],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile-bootstrap CI of the pass rate."""
    if not passes:
        return 0.0, 0.0, 0.0
    arr = np.array([1 if p else 0 for p in passes], dtype=np.int8)
    rng = np.random.default_rng(seed)
    means = []
    n = len(arr)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means.append(arr[idx].mean())
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return float(arr.mean()), lo, hi


@dataclass(frozen=True)
class CellStats:
    model: str
    task: str
    n: int
    pass_at_1: float
    pass_caret_4: float
    pass_caret_8: float
    ci_lo: float
    ci_hi: float


def per_cell_stats(rows: list[dict[str, object]]) -> list[CellStats]:
    """Aggregate per-(model, task) pass^k + bootstrap CI from per-seed rows."""
    grouped: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in rows:
        key = (str(r["model"]), str(r["task"]))
        grouped[key].append(bool(r["passed"]))

    out: list[CellStats] = []
    for (model, task), passes in sorted(grouped.items()):
        _mean, lo, hi = bootstrap_ci(passes)
        out.append(
            CellStats(
                model=model,
                task=task,
                n=len(passes),
                pass_at_1=pass_at_1(passes),
                pass_caret_4=pass_caret_k(passes, 4),
                pass_caret_8=pass_caret_k(passes, 8),
                ci_lo=lo,
                ci_hi=hi,
            )
        )
    return out


def per_model_reliability(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Per-model aggregate: how reliable is each model across tasks?

    pass_at_1_mean  — averaged across (model, task) cells
    pass_caret_8_mean — averaged across (model, task) cells
    reliability_ratio — pass^8 / pass@1, in [0, 1]; closer to 1 = more reliable
    """
    cells = per_cell_stats(rows)
    grouped: dict[str, list[CellStats]] = defaultdict(list)
    for c in cells:
        grouped[c.model].append(c)
    summary = []
    for model, cs in sorted(grouped.items()):
        if not cs:
            continue
        p1 = float(np.mean([c.pass_at_1 for c in cs]))
        p8 = float(np.mean([c.pass_caret_8 for c in cs]))
        ratio = (p8 / p1) if p1 > 0 else 0.0
        summary.append(
            {
                "model": model,
                "n_tasks": len(cs),
                "pass_at_1_mean": round(p1, 3),
                "pass_caret_8_mean": round(p8, 3),
                "reliability_ratio": round(ratio, 3),
            }
        )
    return summary
