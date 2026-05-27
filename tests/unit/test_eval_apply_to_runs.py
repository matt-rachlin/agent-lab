"""Tests that `apply_to_runs` validates inputs the same way as `apply_to_experiment`."""

from __future__ import annotations

from typing import Any

import pytest
from lab.eval import apply_to_runs, clear_registry, evaluator
from lab.eval.framework import EvalResult, RunRow, TaskRow


@pytest.fixture(autouse=True)
def _isolate_registry() -> Any:
    clear_registry()
    yield
    clear_registry()


def test_apply_to_runs_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty run_ids"):
        apply_to_runs([])


def test_apply_to_runs_rejects_unknown_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    @evaluator(name="dummy", version="1.0")
    def _dummy(run: RunRow, task: TaskRow) -> EvalResult:
        return EvalResult.passed_()

    # short-circuit before any DB call
    with pytest.raises(ValueError, match="unknown evaluator"):
        apply_to_runs(["x"], evaluator_names=["does-not-exist"])
