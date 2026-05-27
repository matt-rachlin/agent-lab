"""Phase 13.3 bench orchestrator.

Discover, run, capture, and grade.

Usage:
    python -m benchmarks.runner           # all benches
    python -m benchmarks.runner --quick   # only cheap benches (no GPU)

Exit codes:
    0 — all OK or only warnings (latest <= 1.50 * 7d median)
    1 — at least one regression > 1.50 * 7d median
    2 — runner itself errored
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import statistics
import subprocess
import sys
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from benchmarks import BenchmarkSkipped

BENCH_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BENCH_DIR / "history.csv"
LATEST_PATH = BENCH_DIR / "latest.md"

#: Bench files considered "quick" — safe to run in CI / pre-push.
QUICK_BENCHES: tuple[str, ...] = ("bench_kb_query",)

WARN_RATIO = 1.20
FAIL_RATIO = 1.50

HISTORY_HEADER: tuple[str, ...] = (
    "timestamp",
    "bench_name",
    "metric",
    "value",
    "commit_sha",
)


@dataclass(frozen=True)
class BenchOutcome:
    """One bench module's result."""

    bench_name: str
    status: str  # "ok" | "skipped" | "errored"
    metrics: dict[str, float]
    detail: str = ""

    @property
    def is_failure(self) -> bool:
        return self.status == "errored"


@dataclass(frozen=True)
class RegressionVerdict:
    """One metric's verdict against history."""

    bench_name: str
    metric: str
    latest: float
    median: float | None
    ratio: float | None  # latest / median, None if no median available
    severity: str  # "ok" | "warn" | "fail" | "no-history"


# ----------------------------------------------------------------------
# Bench discovery + execution
# ----------------------------------------------------------------------


def discover_benches(quick: bool = False) -> list[str]:
    """List bench module names (e.g. ``["bench_kb_query", ...]``) in import order."""
    names = sorted(p.stem for p in BENCH_DIR.glob("bench_*.py"))
    if quick:
        names = [n for n in names if n in QUICK_BENCHES]
    return names


def run_one_bench(name: str) -> BenchOutcome:
    """Import ``benchmarks.<name>`` and call its ``run()`` function."""
    try:
        mod = importlib.import_module(f"benchmarks.{name}")
    except ImportError as exc:
        return BenchOutcome(name, "errored", {}, f"import failed: {exc}")

    run_fn = getattr(mod, "run", None)
    if not callable(run_fn):
        return BenchOutcome(name, "errored", {}, "no top-level run() callable")

    try:
        metrics = run_fn()
    except BenchmarkSkipped as exc:
        return BenchOutcome(name, "skipped", {}, str(exc))
    except Exception:
        return BenchOutcome(name, "errored", {}, traceback.format_exc(limit=4))

    if not isinstance(metrics, dict):
        return BenchOutcome(
            name, "errored", {}, f"run() returned {type(metrics).__name__}, want dict"
        )

    # Coerce values to float; reject anything we can't
    clean: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            clean[str(k)] = float(v)
        except (TypeError, ValueError):
            return BenchOutcome(name, "errored", {}, f"metric {k!r}={v!r} not floatable")

    return BenchOutcome(name, "ok", clean)


# ----------------------------------------------------------------------
# History I/O
# ----------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=BENCH_DIR,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip() or "unknown"


def _ensure_history_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # lineterminator="\n" — pre-commit's mixed-line-ending hook flags CRLF.
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(HISTORY_HEADER)


def append_history(
    outcomes: Iterable[BenchOutcome],
    *,
    path: Path = HISTORY_PATH,
    now: Callable[[], str] = _utc_now_iso,
    commit_sha: Callable[[], str] = _git_short_sha,
) -> int:
    """Append one row per metric for each ``ok`` outcome. Returns row count."""
    _ensure_history_header(path)
    rows_written = 0
    timestamp = now()
    sha = commit_sha()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        for outcome in outcomes:
            if outcome.status != "ok":
                continue
            for metric, value in outcome.metrics.items():
                writer.writerow([timestamp, outcome.bench_name, metric, f"{value:.6g}", sha])
                rows_written += 1
    return rows_written


def read_history(path: Path = HISTORY_PATH) -> list[dict[str, str]]:
    """Read the full history file as a list of row dicts."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------------
# Regression detection
# ----------------------------------------------------------------------


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def compute_verdict(
    bench_name: str,
    metric: str,
    latest: float,
    history: list[dict[str, str]],
    *,
    window_days: int = 7,
    latest_sha: str | None = None,
    now: datetime | None = None,
) -> RegressionVerdict:
    """Compare ``latest`` against the median of recent history.

    Excludes rows whose ``commit_sha`` equals ``latest_sha`` (so repeated
    runs on the same commit don't dominate the median).
    """
    cutoff = (now or datetime.now(UTC)) - timedelta(days=window_days)
    samples: list[float] = []
    for row in history:
        if row.get("bench_name") != bench_name or row.get("metric") != metric:
            continue
        ts = _parse_ts(row.get("timestamp", ""))
        if ts is None or ts < cutoff:
            continue
        if latest_sha is not None and row.get("commit_sha") == latest_sha:
            continue
        try:
            samples.append(float(row["value"]))
        except (KeyError, TypeError, ValueError):
            continue

    if len(samples) < 3:
        return RegressionVerdict(
            bench_name=bench_name,
            metric=metric,
            latest=latest,
            median=None,
            ratio=None,
            severity="no-history",
        )

    median = statistics.median(samples)
    if median <= 0:
        # Can happen if a metric is logged as zero; treat as no useful history.
        return RegressionVerdict(bench_name, metric, latest, median, None, "no-history")

    ratio = latest / median
    if ratio > FAIL_RATIO:
        sev = "fail"
    elif ratio > WARN_RATIO:
        sev = "warn"
    else:
        sev = "ok"
    return RegressionVerdict(bench_name, metric, latest, median, ratio, sev)


def grade_outcomes(
    outcomes: Iterable[BenchOutcome],
    history: list[dict[str, str]],
    *,
    latest_sha: str | None = None,
    now: datetime | None = None,
) -> list[RegressionVerdict]:
    verdicts: list[RegressionVerdict] = []
    for outcome in outcomes:
        if outcome.status != "ok":
            continue
        for metric, value in outcome.metrics.items():
            verdicts.append(
                compute_verdict(
                    outcome.bench_name,
                    metric,
                    value,
                    history,
                    latest_sha=latest_sha,
                    now=now,
                )
            )
    return verdicts


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def format_summary(
    outcomes: list[BenchOutcome],
    verdicts: list[RegressionVerdict],
) -> str:
    """Plain-text human summary. Used by the CLI and the markdown snapshot."""
    buf = io.StringIO()
    buf.write("# benchmarks/runner\n\n")
    buf.write(f"timestamp: {_utc_now_iso()}\n")
    buf.write(f"benches run: {len(outcomes)}\n\n")

    buf.write("## Outcomes\n\n")
    for o in outcomes:
        if o.status == "ok":
            metrics_str = ", ".join(f"{k}={v:.4g}" for k, v in o.metrics.items())
            buf.write(f"- {o.bench_name}: ok ({metrics_str})\n")
        elif o.status == "skipped":
            buf.write(f"- {o.bench_name}: skipped — {o.detail}\n")
        else:
            first_line = o.detail.splitlines()[0] if o.detail else "(no detail)"
            buf.write(f"- {o.bench_name}: ERRORED — {first_line}\n")

    if verdicts:
        buf.write("\n## Verdicts vs 7-day median\n\n")
        for v in verdicts:
            if v.median is None:
                buf.write(
                    f"- {v.bench_name}/{v.metric}: latest={v.latest:.4g} "
                    f"(no-history, need >=3 samples)\n"
                )
            else:
                ratio_str = f"{v.ratio:.2f}x" if v.ratio is not None else "?"
                buf.write(
                    f"- {v.bench_name}/{v.metric}: latest={v.latest:.4g} "
                    f"median={v.median:.4g} ratio={ratio_str} [{v.severity}]\n"
                )

    fails = [v for v in verdicts if v.severity == "fail"]
    warns = [v for v in verdicts if v.severity == "warn"]
    buf.write(f"\n## Summary: {len(fails)} fail, {len(warns)} warn\n")
    return buf.getvalue()


def write_latest_snapshot(text: str, *, path: Path = LATEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def run_all(quick: bool = False) -> tuple[list[BenchOutcome], list[RegressionVerdict]]:
    """Run all (or quick) benches, append history, grade vs history."""
    names = discover_benches(quick=quick)
    outcomes = [run_one_bench(n) for n in names]
    append_history(outcomes)
    history = read_history()
    sha = _git_short_sha()
    verdicts = grade_outcomes(outcomes, history, latest_sha=sha)
    return outcomes, verdicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.runner")
    parser.add_argument("--quick", action="store_true", help="only the cheap benches")
    args = parser.parse_args(argv)

    try:
        outcomes, verdicts = run_all(quick=args.quick)
    except Exception:
        traceback.print_exc()
        return 2

    summary = format_summary(outcomes, verdicts)
    sys.stdout.write(summary)
    write_latest_snapshot(summary)

    if any(o.status == "errored" for o in outcomes):
        return 1
    if any(v.severity == "fail" for v in verdicts):
        return 1
    return 0


def _typed_main(argv: Any = None) -> int:  # pragma: no cover — argv typing shim
    return main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
