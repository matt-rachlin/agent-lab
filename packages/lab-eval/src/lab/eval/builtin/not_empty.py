"""not_empty: response_text is non-empty after stripping whitespace."""

from __future__ import annotations

from lab.eval.framework import EvalResult, RunRow, TaskRow, evaluator


@evaluator(
    name="not_empty",
    version="1.0",
    description="response_text is non-empty after stripping whitespace",
    threshold=1.0,
)
def not_empty(run: RunRow, task: TaskRow) -> EvalResult:
    if run.response_text is None:
        return EvalResult.failed(reasoning="response_text is None")
    if not run.response_text.strip():
        return EvalResult.failed(reasoning="response_text is empty / whitespace only")
    return EvalResult.passed_(reasoning=f"{len(run.response_text)} chars")
