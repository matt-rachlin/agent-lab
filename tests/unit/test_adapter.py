"""Unit tests for `lab.inspect_bridge.adapter.lab_task_to_inspect`.

We don't run the resulting Inspect task here — the integration test does
that. These tests just verify the shape of what we build.
"""

from __future__ import annotations

from typing import Any

import pytest
from lab.tasks.registry import Task

from lab.inspect_bridge.adapter import _select_scorers, lab_task_to_inspect


def _task(**overrides: Any) -> Task:
    base = {
        "suite": "test",
        "slug": "adapter-1",
        "input": "do the thing",
        "tools": [{"name": "fs_read"}],
        "max_turns": 3,
        "tool_budget": 2,
    }
    base.update(overrides)
    return Task.model_validate(base)


def test_adapter_builds_single_sample_task() -> None:
    """The Inspect Task has exactly one sample."""

    async def passthrough(state: Any, generate: Any) -> Any:
        return state

    # Pass a custom solver to avoid hitting LiteLLM/sandbox during unit tests.
    inspect_task = lab_task_to_inspect(_task(), model="qwen3-14b-q4", solver_override=passthrough)
    # Inspect Task carries .dataset; we just check there's exactly one item
    # and it has the expected metadata.
    samples = list(inspect_task.dataset)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.input == "do the thing"
    assert sample.id == "adapter-1"
    assert sample.metadata is not None
    lab_task = sample.metadata["lab_task"]
    assert lab_task.slug == "adapter-1"
    assert sample.metadata["lab_max_turns"] == 3
    assert sample.metadata["lab_tool_budget"] == 2


def test_adapter_uses_gold_answer_as_target() -> None:
    """`Sample.target` should reflect the lab task's gold answer."""

    async def passthrough(state: Any, generate: Any) -> Any:
        return state

    task = _task(gold_answer="42")
    inspect_task = lab_task_to_inspect(task, model="x", solver_override=passthrough)
    sample = next(iter(inspect_task.dataset))
    assert sample.target == "42"


def test_select_scorers_defaults_to_budget_only() -> None:
    """No predicate, no tool_call rubric → only `budget_respected`."""

    scorers = _select_scorers(_task())
    assert len(scorers) == 1


def test_adapter_validates_required_args() -> None:
    """Without solver_override the adapter must still accept model + sandbox."""

    # We don't actually invoke the solver here, just construct the Inspect
    # task and confirm no exception.
    task = _task()
    inspect_task = lab_task_to_inspect(task, model="x", sandbox=None)
    assert inspect_task is not None


def test_adapter_passes_tool_names_from_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Inspect task should filter to the names declared in task.tools."""

    from lab.inspect_bridge import adapter as adapter_mod

    captured: dict[str, Any] = {}

    def fake_solver(**kwargs: Any) -> Any:
        captured.update(kwargs)

        async def _s(state: Any, gen: Any) -> Any:
            return state

        return _s

    monkeypatch.setattr(adapter_mod, "model_with_tools", fake_solver)
    task = _task(tools=[{"name": "fs_read"}, {"name": "fs_write"}])
    lab_task_to_inspect(task, model="m")
    assert captured["tool_names"] == ["fs_read", "fs_write"]
