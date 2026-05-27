"""json_valid: response is parseable JSON; extracts from common wrappers."""

from __future__ import annotations

import json
import re

from lab.eval.framework import EvalResult, RunRow, TaskRow, evaluator

# Strip ```json ... ``` and ``` ... ``` code fences if present
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


@evaluator(
    name="json_valid",
    version="1.0",
    description="response_text parses as JSON (after stripping common code fences)",
    threshold=1.0,
)
def json_valid(run: RunRow, task: TaskRow) -> EvalResult:
    # Only relevant when the task asks for JSON
    payload_input = task.payload.get("input") or ""
    rubric_type = (task.payload.get("rubric") or {}).get("type")
    if "json" not in str(payload_input).lower() and rubric_type != "json_schema":
        return EvalResult.skip("task does not ask for JSON")
    if not run.response_text:
        return EvalResult.failed(reasoning="empty response_text")
    candidate = _extract(run.response_text)
    try:
        json.loads(candidate)
    except json.JSONDecodeError as exc:
        return EvalResult.failed(reasoning=f"JSON parse error: {exc.msg} at {exc.pos}")
    return EvalResult.passed_()
