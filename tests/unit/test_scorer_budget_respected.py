"""Unit tests for `lab.inspect_bridge.scorer.budget_respected`."""

from __future__ import annotations

import asyncio
from typing import Any

from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from lab.inspect_bridge.scorer import budget_respected
from lab.tasks.registry import Task


def _state(
    *,
    max_turns: int,
    tool_budget: int,
    actual_turns: int,
    tool_calls: int,
    terminated: str,
) -> TaskState:
    task = Task.model_validate(
        {
            "suite": "test",
            "slug": "br",
            "input": "hi",
            "max_turns": max_turns,
            "tool_budget": tool_budget,
        }
    )
    lab_agent = {
        "actual_turns": actual_turns,
        "tool_call_count": tool_calls,
        "terminated_reason": terminated,
        "turns": [],
    }
    return TaskState(
        model="x",
        sample_id="s",
        epoch=0,
        input="hi",
        messages=[ChatMessageUser(content="hi")],
        metadata={"lab_task": task, "lab_agent": lab_agent},
    )


def _run(s: Any, state: TaskState) -> Any:
    return asyncio.run(s(state, Target("")))


def test_within_budget_passes() -> None:
    state = _state(
        max_turns=5,
        tool_budget=3,
        actual_turns=2,
        tool_calls=1,
        terminated="model_finished",
    )
    assert _run(budget_respected(), state).value == 1.0


def test_budget_exhausted_fails() -> None:
    state = _state(
        max_turns=5,
        tool_budget=2,
        actual_turns=3,
        tool_calls=2,
        terminated="budget_exhausted",
    )
    out = _run(budget_respected(), state)
    assert out.value == 0.0
    assert "budget_exhausted" in (out.explanation or "")


def test_max_turns_reached_fails() -> None:
    state = _state(
        max_turns=2,
        tool_budget=99,
        actual_turns=2,
        tool_calls=2,
        terminated="max_turns_reached",
    )
    out = _run(budget_respected(), state)
    assert out.value == 0.0
    assert "max_turns_reached" in (out.explanation or "")


def test_turn_overflow_fails() -> None:
    """If somehow actual_turns > max_turns, fail with a clear message."""
    state = _state(
        max_turns=2,
        tool_budget=10,
        actual_turns=3,
        tool_calls=0,
        terminated="model_finished",
    )
    out = _run(budget_respected(), state)
    assert out.value == 0.0
    assert "actual_turns" in (out.explanation or "")


def test_tool_overflow_fails() -> None:
    state = _state(
        max_turns=10,
        tool_budget=1,
        actual_turns=2,
        tool_calls=2,
        terminated="model_finished",
    )
    out = _run(budget_respected(), state)
    assert out.value == 0.0
    assert "tool_call_count" in (out.explanation or "")
