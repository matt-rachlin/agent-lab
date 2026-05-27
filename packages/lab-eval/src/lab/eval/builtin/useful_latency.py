"""useful_latency: passes iff `latency_under` AND `not_empty` both pass.

Surfaced by EXP-001 postmortem: empty responses currently pass `latency_under`
because there's nothing to generate, which gives a misleadingly-fast pass.
"""

from __future__ import annotations

from lab.eval.builtin.latency_under import latency_under
from lab.eval.builtin.not_empty import not_empty
from lab.eval.framework import EvalResult, RunRow, TaskRow, evaluator


@evaluator(
    name="useful_latency",
    version="1.0",
    description="latency_under AND not_empty (compound; empty responses no longer pass on latency)",
    threshold=1.0,
)
def useful_latency(run: RunRow, task: TaskRow) -> EvalResult:
    lat = latency_under(run, task)
    if lat.skipped:
        return EvalResult.skip(lat.skip_reason or "latency_under skipped")
    if lat.passed is not True:
        return EvalResult.failed(reasoning=f"latency: {lat.reasoning}")
    ne = not_empty(run, task)
    if ne.passed is not True:
        return EvalResult.failed(reasoning=f"not_empty: {ne.reasoning}")
    return EvalResult.passed_(reasoning=f"{lat.reasoning}; {ne.reasoning}")
