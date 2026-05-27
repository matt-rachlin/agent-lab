"""bfcl_ast_match: score a BFCL cell off the persisted trace.

Pulls the trace JSON, extracts the ``tool_calls`` field of the LiteLLM
response, and runs the vendored BFCL AST checker against the task's
``rubric.ground_truth`` payload.

The sweep runner pre-computes the AST grade at cell-execution time
(`_execute_bfcl_cell` writes the score straight into ``eval_results``),
so this evaluator is primarily defensive — it re-grades cells whose
inline score wasn't persisted for any reason (older runs from before
17.5, runs imported from another lab, etc.).

Cells whose task is not a BFCL task — i.e. ``rubric.type != "bfcl_ast"``
on the task — are SKIPPED, so this evaluator is safe to leave registered
alongside PBS-Agent / PBS-v0.1 evaluators (``lab eval apply`` filters by
category but doesn't gate on rubric type).
"""

from __future__ import annotations

import json
from typing import Any

from lab.eval.external.bfcl import grade_bfcl_response
from lab.eval.framework import EvalResult, RunRow, TaskRow, evaluator


def _load_trace(trace_path: str | None) -> dict[str, Any] | None:
    """Return the first JSON line of a trace file (local paths only).

    Trace files persisted to MinIO use ``s3://`` URIs. In that case the
    inline grader in ``_execute_bfcl_cell`` has already written the
    eval_results row; we don't need to re-fetch and can simply return
    ``None`` so the caller skips with a clean "not graded" message.
    """

    if not trace_path:
        return None
    if not trace_path.startswith(("/", "./")):
        # Anything that isn't a local-filesystem path is out of scope.
        return None
    try:
        with open(trace_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            return None
        return parsed
    return None


def _extract_tool_calls(raw_response: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the response's ``tool_calls`` list (OpenAI envelope)."""

    if not isinstance(raw_response, dict):
        return []
    choices = raw_response.get("choices") or []
    if not choices:
        return []
    msg = choices[0].get("message") or {}
    tcs = msg.get("tool_calls") or []
    if not isinstance(tcs, list):
        return []
    return tcs


@evaluator(
    name="bfcl_ast_match",
    version="1.0",
    description=(
        "Berkeley Function Calling Leaderboard v3 AST grader: model tool-call "
        "matches ground_truth (Python-only, simple/multiple/parallel categories)."
    ),
    threshold=1.0,
    category="deterministic",
)
def bfcl_ast_match(run: RunRow, task: TaskRow) -> EvalResult:
    rubric = task.payload.get("rubric") or {}
    if rubric.get("type") not in {"bfcl_ast", "custom"}:
        return EvalResult.skip("not a bfcl rubric")
    if "bfcl_category" not in rubric:
        return EvalResult.skip("not a bfcl task (no bfcl_category)")

    category = rubric.get("bfcl_category")
    ground_truth = rubric.get("ground_truth")
    raw_functions = rubric.get("raw_functions")
    if not (isinstance(ground_truth, list) and isinstance(raw_functions, list)):
        return EvalResult.skip("rubric missing ground_truth or raw_functions")

    trace = _load_trace(run.trace_path)
    if trace is None:
        return EvalResult.skip("trace not available for re-grading (inline grade only)")
    raw_response = trace.get("raw_response")
    tool_calls = _extract_tool_calls(raw_response)

    result = grade_bfcl_response(
        raw_functions=raw_functions,
        ground_truth=ground_truth,
        tool_calls=tool_calls,
        category=str(category),
    )
    passed = bool(result["valid"])
    return EvalResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        reasoning=("ok" if passed else (result.get("error_type") or "fail")),
        metadata={"bfcl": result, "tool_calls_observed": len(tool_calls)},
    )


__all__ = ["bfcl_ast_match"]
