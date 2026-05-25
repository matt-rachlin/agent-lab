"""tokens_under: passes if tokens_out is under a threshold (cost discipline)."""

from __future__ import annotations

import os

from lab.eval.framework import EvalResult, RunRow, TaskRow, evaluator

DEFAULT_THRESHOLD = int(os.environ.get("LAB_TOKENS_OUT_THRESHOLD", "500"))


@evaluator(
    name="tokens_under",
    version="1.0",
    description=f"run.tokens_out <= {DEFAULT_THRESHOLD} (configurable via LAB_TOKENS_OUT_THRESHOLD)",
    threshold=1.0,
)
def tokens_under(run: RunRow, task: TaskRow) -> EvalResult:
    if run.tokens_out is None:
        return EvalResult.skip("no tokens_out")
    if run.tokens_out <= DEFAULT_THRESHOLD:
        return EvalResult.passed_(reasoning=f"{run.tokens_out} tokens")
    return EvalResult.failed(reasoning=f"{run.tokens_out} > {DEFAULT_THRESHOLD}")
