"""regex_match: rubric.pattern matched against response_text."""

from __future__ import annotations

import re

from lab.eval.framework import EvalResult, RunRow, TaskRow, evaluator


@evaluator(
    name="regex_match",
    version="1.0",
    description="task.rubric.pattern (regex, DOTALL) matches response_text",
    threshold=1.0,
)
def regex_match(run: RunRow, task: TaskRow) -> EvalResult:
    rubric = task.payload.get("rubric") or {}
    if rubric.get("type") != "regex":
        return EvalResult.skip("rubric not regex")
    pattern_text = rubric.get("pattern")
    if not isinstance(pattern_text, str):
        return EvalResult.skip("rubric.pattern missing")
    if not run.response_text:
        return EvalResult.failed(reasoning="empty response_text")
    flags = re.DOTALL
    if not rubric.get("case_sensitive", False):
        flags |= re.IGNORECASE
    try:
        compiled = re.compile(pattern_text, flags)
    except re.error as exc:
        return EvalResult.skip(f"bad regex: {exc}")
    if compiled.search(run.response_text):
        return EvalResult.passed_(reasoning=f"matched {pattern_text!r}")
    return EvalResult.failed(reasoning=f"no match for {pattern_text!r}")
