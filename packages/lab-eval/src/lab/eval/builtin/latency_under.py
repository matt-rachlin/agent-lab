"""latency_under: passes if latency_ms is under a threshold."""

from __future__ import annotations

import os

from lab.eval.framework import EvalResult, RunRow, TaskRow, evaluator

DEFAULT_THRESHOLD_MS = int(os.environ.get("LAB_LATENCY_THRESHOLD_MS", "10000"))


@evaluator(
    name="latency_under",
    version="1.0",
    description=f"run.latency_ms < {DEFAULT_THRESHOLD_MS} (configurable via LAB_LATENCY_THRESHOLD_MS)",
    threshold=1.0,
)
def latency_under(run: RunRow, task: TaskRow) -> EvalResult:
    if run.latency_ms is None:
        return EvalResult.skip("no latency_ms")
    if run.latency_ms <= DEFAULT_THRESHOLD_MS:
        return EvalResult.passed_(reasoning=f"{run.latency_ms}ms")
    return EvalResult.failed(reasoning=f"{run.latency_ms}ms > {DEFAULT_THRESHOLD_MS}ms")
