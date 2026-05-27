"""Phase 13.3 — tests for the benchmarks/ runner.

Covers the parts of ``benchmarks.runner`` that don't require real
services: bench dispatch, history I/O, regression math, and the
end-to-end ``run_all`` orchestrator with mocked bench modules.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from benchmarks import BenchmarkSkipped
from benchmarks import runner as runner_mod
from benchmarks.runner import (
    BenchOutcome,
    RegressionVerdict,
    append_history,
    compute_verdict,
    format_summary,
    grade_outcomes,
    read_history,
    run_one_bench,
)

# ----------------------------------------------------------------------
# History I/O
# ----------------------------------------------------------------------


def test_append_history_writes_header_then_rows(tmp_path: Path) -> None:
    """First write creates the header; subsequent writes only append rows."""
    history = tmp_path / "h.csv"
    outcomes = [
        BenchOutcome("bench_demo", "ok", {"p50_sec": 0.12, "n": 20.0}),
        BenchOutcome("bench_skipped", "skipped", {}, "service down"),  # not recorded
    ]

    rows_written = append_history(
        outcomes,
        path=history,
        now=lambda: "2026-05-26T22:00:00Z",
        commit_sha=lambda: "abc1234",
    )
    assert rows_written == 2

    rows = read_history(history)
    assert [r["bench_name"] for r in rows] == ["bench_demo", "bench_demo"]
    assert [r["metric"] for r in rows] == ["p50_sec", "n"]
    assert rows[0]["commit_sha"] == "abc1234"
    assert rows[0]["timestamp"] == "2026-05-26T22:00:00Z"

    # Second append — header NOT duplicated; new rows just tacked on.
    append_history(
        [BenchOutcome("bench_demo", "ok", {"p50_sec": 0.15})],
        path=history,
        now=lambda: "2026-05-26T23:00:00Z",
        commit_sha=lambda: "def5678",
    )
    raw = history.read_text(encoding="utf-8").splitlines()
    assert raw[0] == "timestamp,bench_name,metric,value,commit_sha"
    assert sum(1 for line in raw if line.startswith("timestamp,")) == 1
    assert len(raw) == 1 + 2 + 1  # header + 2 from first call + 1 from second


# ----------------------------------------------------------------------
# Regression math
# ----------------------------------------------------------------------


def _hist_row(
    bench: str = "b",
    metric: str = "m",
    value: float = 1.0,
    ts: str = "2026-05-26T12:00:00Z",
    sha: str = "old",
) -> dict[str, str]:
    return {
        "bench_name": bench,
        "metric": metric,
        "value": str(value),
        "timestamp": ts,
        "commit_sha": sha,
    }


def test_compute_verdict_warn_fail_ok_thresholds() -> None:
    """Exercises the 1.20x / 1.50x cliffs."""
    now = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    history = [
        _hist_row(value=1.0, ts=yesterday),
        _hist_row(value=1.0, ts=yesterday),
        _hist_row(value=1.0, ts=yesterday),  # median = 1.0
    ]

    # latest == median → ok
    v = compute_verdict("b", "m", 1.0, history, now=now)
    assert v.severity == "ok"
    assert v.ratio == pytest.approx(1.0)

    # latest = 1.30 → warn (>1.20, <1.50)
    v = compute_verdict("b", "m", 1.30, history, now=now)
    assert v.severity == "warn"

    # latest = 1.60 → fail (>1.50)
    v = compute_verdict("b", "m", 1.60, history, now=now)
    assert v.severity == "fail"


def test_compute_verdict_excludes_same_commit_sha() -> None:
    """Re-running the same commit shouldn't dominate the rolling median."""
    now = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    history = [
        # three rows on the latest sha — should ALL be excluded
        _hist_row(value=10.0, ts=yesterday, sha="latest"),
        _hist_row(value=10.0, ts=yesterday, sha="latest"),
        _hist_row(value=10.0, ts=yesterday, sha="latest"),
        # three older-sha rows that DO count
        _hist_row(value=1.0, ts=yesterday, sha="oldA"),
        _hist_row(value=1.0, ts=yesterday, sha="oldB"),
        _hist_row(value=1.0, ts=yesterday, sha="oldC"),
    ]
    v = compute_verdict("b", "m", 2.0, history, latest_sha="latest", now=now)
    assert v.median == pytest.approx(1.0)
    assert v.severity == "fail"  # 2.0 / 1.0 = 2.0x


def test_compute_verdict_no_history_under_three_samples() -> None:
    now = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    history = [_hist_row(value=1.0, ts=yesterday), _hist_row(value=1.0, ts=yesterday)]
    v = compute_verdict("b", "m", 99.0, history, now=now)
    assert v.severity == "no-history"
    assert v.ratio is None
    assert v.median is None


def test_compute_verdict_window_excludes_old_samples() -> None:
    """Rows outside the 7-day window must not count toward the median."""
    now = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)
    long_ago = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    inside = (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    history = [
        _hist_row(value=100.0, ts=long_ago),  # too old
        _hist_row(value=100.0, ts=long_ago),
        _hist_row(value=1.0, ts=inside),
        _hist_row(value=1.0, ts=inside),
    ]
    v = compute_verdict("b", "m", 1.0, history, now=now)
    # Only 2 in-window samples — under the 3-sample minimum → no-history.
    assert v.severity == "no-history"


# ----------------------------------------------------------------------
# Bench dispatch
# ----------------------------------------------------------------------


def test_run_one_bench_handles_skip_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bench raising BenchmarkSkipped → skipped; other exception → errored."""

    # Create a fake bench module that we can swap in via patch on importlib.
    class _SkipMod:
        @staticmethod
        def run() -> dict[str, float]:
            raise BenchmarkSkipped("service down")

    class _ErrMod:
        @staticmethod
        def run() -> dict[str, float]:
            raise RuntimeError("boom")

    class _OkMod:
        @staticmethod
        def run() -> dict[str, float]:
            return {"x": 1.5}

    class _BadShapeMod:
        @staticmethod
        def run() -> list[float]:  # wrong return type
            return [1.0, 2.0]

    cases = [
        ("bench_skip", _SkipMod, "skipped"),
        ("bench_err", _ErrMod, "errored"),
        ("bench_ok", _OkMod, "ok"),
        ("bench_bad", _BadShapeMod, "errored"),
    ]
    for name, mod, expected_status in cases:
        with patch.object(runner_mod.importlib, "import_module", return_value=mod):
            outcome = run_one_bench(name)
        assert outcome.bench_name == name
        assert outcome.status == expected_status, outcome


def test_run_one_bench_import_error_is_errored() -> None:
    with patch.object(
        runner_mod.importlib,
        "import_module",
        side_effect=ImportError("no module"),
    ):
        outcome = run_one_bench("bench_missing")
    assert outcome.status == "errored"
    assert "import failed" in outcome.detail


# ----------------------------------------------------------------------
# grade_outcomes orchestrator
# ----------------------------------------------------------------------


def test_grade_outcomes_only_grades_ok_results() -> None:
    """Skipped/errored outcomes don't produce verdicts."""
    now = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    history = [
        _hist_row(bench="bench_a", metric="p50", value=1.0, ts=yesterday),
        _hist_row(bench="bench_a", metric="p50", value=1.0, ts=yesterday),
        _hist_row(bench="bench_a", metric="p50", value=1.0, ts=yesterday),
    ]
    outcomes = [
        BenchOutcome("bench_a", "ok", {"p50": 1.10}),
        BenchOutcome("bench_b", "skipped", {}, "down"),
        BenchOutcome("bench_c", "errored", {}, "boom"),
    ]
    verdicts = grade_outcomes(outcomes, history, now=now)
    assert len(verdicts) == 1
    assert verdicts[0].bench_name == "bench_a"
    assert verdicts[0].severity == "ok"


# ----------------------------------------------------------------------
# Summary formatting smoke
# ----------------------------------------------------------------------


def test_format_summary_includes_all_outcomes_and_verdicts() -> None:
    outcomes = [
        BenchOutcome("bench_a", "ok", {"p50_sec": 0.123}),
        BenchOutcome("bench_b", "skipped", {}, "valkey down"),
        BenchOutcome("bench_c", "errored", {}, "RuntimeError: nope\nmore detail"),
    ]
    verdicts = [
        RegressionVerdict("bench_a", "p50_sec", 0.123, 0.1, 1.23, "warn"),
    ]
    text = format_summary(outcomes, verdicts)
    assert "bench_a: ok" in text
    assert "bench_b: skipped" in text
    assert "valkey down" in text
    assert "bench_c: ERRORED" in text
    assert "RuntimeError: nope" in text  # first line of detail
    assert "more detail" not in text  # later lines truncated
    assert "[warn]" in text
    assert "1 fail" not in text  # we have 0 fails
    assert "0 fail, 1 warn" in text
