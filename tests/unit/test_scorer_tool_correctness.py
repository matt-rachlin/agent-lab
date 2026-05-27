"""Unit tests for `lab.inspect_bridge.scorer.tool_correctness`."""

from __future__ import annotations

import asyncio
from typing import Any

from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import NOANSWER, Target
from inspect_ai.solver import TaskState
from lab.tasks.registry import Task

from lab.inspect_bridge.scorer import tool_correctness


def _state(*, rubric: dict[str, Any] | None, turns: list[dict[str, Any]]) -> TaskState:
    task = Task.model_validate(
        {
            "suite": "test",
            "slug": "tc-test",
            "input": "hi",
            "rubric": rubric,
            "max_turns": 3,
            "tool_budget": 2,
        }
    )
    lab_agent: dict[str, Any] = {
        "actual_turns": len(turns),
        "tool_call_count": sum(len(t.get("tool_calls") or []) for t in turns),
        "terminated_reason": "model_finished",
        "turns": turns,
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


def test_matching_tool_call_passes() -> None:
    state = _state(
        rubric={
            "type": "tool_call",
            "target_tool": "fs_read",
            "expected_args": {"path": "x.txt"},
        },
        turns=[
            {
                "turn": 0,
                "tool_calls": [
                    {"tool": "fs_read", "args": {"path": "x.txt"}, "result": "ok"},
                ],
            }
        ],
    )
    out = _run(tool_correctness(), state)
    assert out.value == 1.0


def test_extra_args_allowed() -> None:
    state = _state(
        rubric={
            "type": "tool_call",
            "target_tool": "fs_read",
            "expected_args": {"path": "x.txt"},
        },
        turns=[
            {
                "turn": 0,
                "tool_calls": [
                    {
                        "tool": "fs_read",
                        "args": {"path": "x.txt", "max_bytes": 1024},
                        "result": "ok",
                    },
                ],
            }
        ],
    )
    assert _run(tool_correctness(), state).value == 1.0


def test_wrong_args_fails() -> None:
    state = _state(
        rubric={
            "type": "tool_call",
            "target_tool": "fs_read",
            "expected_args": {"path": "x.txt"},
        },
        turns=[
            {
                "turn": 0,
                "tool_calls": [
                    {"tool": "fs_read", "args": {"path": "y.txt"}, "result": "ok"},
                ],
            }
        ],
    )
    out = _run(tool_correctness(), state)
    assert out.value == 0.0
    assert "no call to" in (out.explanation or "")


def test_wrong_tool_fails() -> None:
    state = _state(
        rubric={
            "type": "tool_call",
            "target_tool": "fs_read",
            "expected_args": {"path": "x.txt"},
        },
        turns=[
            {
                "turn": 0,
                "tool_calls": [
                    {"tool": "shell_exec", "args": {"command": "cat x.txt"}, "result": "ok"},
                ],
            }
        ],
    )
    assert _run(tool_correctness(), state).value == 0.0


def test_non_tool_call_rubric_returns_noanswer() -> None:
    state = _state(
        rubric={"type": "exact_match"},
        turns=[],
    )
    out = _run(tool_correctness(), state)
    assert out.value == NOANSWER


def test_missing_rubric_returns_noanswer() -> None:
    state = _state(rubric=None, turns=[])
    out = _run(tool_correctness(), state)
    assert out.value == NOANSWER


def test_truncated_args_are_treated_as_no_match() -> None:
    """If the recorded args were truncated by the solver, we can't verify keys."""
    state = _state(
        rubric={
            "type": "tool_call",
            "target_tool": "fs_read",
            "expected_args": {"path": "x.txt"},
        },
        turns=[
            {
                "turn": 0,
                "tool_calls": [
                    {
                        "tool": "fs_read",
                        "args": {"_truncated": True, "preview": "..."},
                        "result": "ok",
                    },
                ],
            }
        ],
    )
    out = _run(tool_correctness(), state)
    # Treating as a non-match (0.0) is the conservative choice.
    assert out.value == 0.0
