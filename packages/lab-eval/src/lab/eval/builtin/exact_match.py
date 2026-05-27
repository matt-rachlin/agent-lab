"""exact_match: compare response_text against task.gold_answer."""

from __future__ import annotations

import re

from lab.eval.framework import EvalResult, RunRow, TaskRow, evaluator


@evaluator(
    name="exact_match",
    version="1.0",
    description="response_text contains task.gold_answer (case-insensitive, stripped, "
    "alphanumeric-only by default)",
    threshold=1.0,
    category="deterministic",
)
def exact_match(run: RunRow, task: TaskRow) -> EvalResult:
    gold = task.payload.get("gold_answer")
    rubric = task.payload.get("rubric") or {}
    case_sensitive = bool(rubric.get("case_sensitive", False))

    if not isinstance(gold, str) or not gold:
        return EvalResult.skip("no gold_answer")
    if not run.response_text:
        return EvalResult.failed(reasoning="empty response_text")

    response = run.response_text
    target = gold
    if not case_sensitive:
        response = response.lower()
        target = target.lower()

    # Look for the gold answer as a whole word/token in the response
    pattern = re.compile(r"(?<!\w)" + re.escape(target) + r"(?!\w)")
    if pattern.search(response):
        return EvalResult.passed_(reasoning=f"found {gold!r}")
    return EvalResult.failed(reasoning=f"gold {gold!r} not in response")
