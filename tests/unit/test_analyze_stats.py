"""pass^k + bootstrap CI tests."""

from __future__ import annotations

from lab.analyze.stats import (
    bootstrap_ci,
    pass_at_1,
    pass_caret_k,
    per_cell_stats,
    per_model_reliability,
)


def test_pass_at_1() -> None:
    assert pass_at_1([True, True, True, True]) == 1.0
    assert pass_at_1([False, False, False, False]) == 0.0
    assert pass_at_1([True, False, True, False]) == 0.5
    assert pass_at_1([]) == 0.0


def test_pass_caret_k_all_pass() -> None:
    # 8/8 → pass^8 = 1
    assert pass_caret_k([True] * 8, 8) == 1.0
    # 7/8 → pass^8 = 0
    assert pass_caret_k([True] * 7 + [False], 8) == 0.0


def test_pass_caret_k_partial() -> None:
    # 6/8 → pass^4 = C(6,4)/C(8,4) = 15/70 ≈ 0.214
    assert abs(pass_caret_k([True] * 6 + [False, False], 4) - 15 / 70) < 1e-9


def test_pass_caret_k_invalid() -> None:
    assert pass_caret_k([True], 0) == 0.0
    assert pass_caret_k([True], 5) == 0.0


def test_bootstrap_ci_all_pass() -> None:
    mean, lo, hi = bootstrap_ci([True] * 10, seed=42)
    assert mean == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_bootstrap_ci_split() -> None:
    mean, lo, hi = bootstrap_ci([True] * 5 + [False] * 5, seed=42, n_resamples=5000)
    assert abs(mean - 0.5) < 0.01
    assert lo < hi
    assert lo > 0.0
    assert hi < 1.0


def test_per_cell_stats() -> None:
    rows = [
        {"model": "A", "task": "t1", "passed": True},
        {"model": "A", "task": "t1", "passed": True},
        {"model": "A", "task": "t1", "passed": False},
        {"model": "A", "task": "t1", "passed": True},
    ]
    stats = per_cell_stats(rows)
    assert len(stats) == 1
    s = stats[0]
    assert s.model == "A"
    assert s.task == "t1"
    assert s.n == 4
    assert s.pass_at_1 == 0.75


def test_per_model_reliability() -> None:
    # Model A: pass@1=1.0, pass^8 N/A (only 4 seeds → 0); two tasks averaged
    rows = [{"model": "A", "task": "t1", "passed": True}] * 4 + [
        {"model": "A", "task": "t2", "passed": True}
    ] * 4
    summary = per_model_reliability(rows)
    assert summary[0]["model"] == "A"
    assert summary[0]["pass_at_1_mean"] == 1.0
