"""cost_under: passes if cost_usd is under a threshold (Phase 4 will fill cost_usd)."""

from __future__ import annotations

import os

from lab.eval.framework import EvalResult, RunRow, TaskRow, evaluator

DEFAULT_THRESHOLD_USD = float(os.environ.get("LAB_COST_THRESHOLD_USD", "0.05"))


@evaluator(
    name="cost_under",
    version="1.0",
    description=f"run.cost_usd <= ${DEFAULT_THRESHOLD_USD} (configurable via LAB_COST_THRESHOLD_USD)",
    threshold=1.0,
)
def cost_under(run: RunRow, task: TaskRow) -> EvalResult:
    if run.cost_usd is None:
        return EvalResult.skip("no cost_usd recorded yet (Phase 4)")
    if run.cost_usd <= DEFAULT_THRESHOLD_USD:
        return EvalResult.passed_(reasoning=f"${run.cost_usd:.6f}")
    return EvalResult.failed(reasoning=f"${run.cost_usd:.6f} > ${DEFAULT_THRESHOLD_USD}")
