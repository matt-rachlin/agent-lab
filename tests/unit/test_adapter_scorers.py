"""Unit tests for `lab.inspect_bridge.adapter._select_scorers`."""

from __future__ import annotations

from typing import Any

from inspect_ai._util.registry import registry_info
from lab.inspect_bridge.adapter import _select_scorers
from lab.tasks.registry import Task


def _task(**overrides: Any) -> Task:
    base: dict[str, Any] = {
        "suite": "test",
        "slug": "adapter-scorers",
        "input": "do the thing",
        "max_turns": 3,
        "tool_budget": 2,
    }
    base.update(overrides)
    return Task.model_validate(base)


def _names(scorers: list[Any]) -> list[str]:
    """Pull the registered scorer name off each Inspect scorer callable.

    Inspect registers names as `<namespace>/<name>`; we strip the prefix
    so the assertions read against the friendly name we passed in
    `@scorer(name=...)`.
    """
    out: list[str] = []
    for s in scorers:
        name = registry_info(s).name
        if "/" in name:
            name = name.split("/", 1)[1]
        out.append(name)
    return out


def test_default_only_budget_respected() -> None:
    """No predicate, no tool_call rubric → only `budget_respected`."""
    names = _names(_select_scorers(_task()))
    assert names == ["budget_respected"]


def test_with_predicate_adds_end_state() -> None:
    task = _task(
        success_predicate={
            "type": "workspace_file_contains",
            "path": "out.txt",
            "substring": "PASS",
        }
    )
    names = _names(_select_scorers(task))
    assert "end_state" in names
    assert "budget_respected" in names


def test_with_tool_call_rubric_adds_tool_correctness() -> None:
    task = _task(
        rubric={
            "type": "tool_call",
            "target_tool": "fs_read",
            "expected_args": {"path": "x"},
        }
    )
    names = _names(_select_scorers(task))
    assert "tool_correctness" in names
    assert "budget_respected" in names


def test_predicate_include_judge_adds_trajectory_judge() -> None:
    task = _task(
        success_predicate={
            "type": "workspace_file_exists",
            "path": "out.txt",
            "include_judge": True,
        }
    )
    names = _names(_select_scorers(task))
    assert "trajectory_judge" in names
    assert "end_state" in names
    assert "budget_respected" in names


def test_combined_shape() -> None:
    """A task with predicate + tool_call rubric + judge → all four scorers."""
    task = _task(
        success_predicate={
            "type": "workspace_file_contains",
            "path": "out.txt",
            "substring": "ok",
            "include_judge": True,
        },
        rubric={
            "type": "tool_call",
            "target_tool": "fs_write",
            "expected_args": {"path": "out.txt"},
        },
    )
    names = _names(_select_scorers(task))
    assert sorted(names) == sorted(
        ["end_state", "tool_correctness", "trajectory_judge", "budget_respected"]
    )


def test_exact_match_rubric_does_not_add_tool_correctness() -> None:
    task = _task(rubric={"type": "exact_match"}, gold_answer="42")
    names = _names(_select_scorers(task))
    assert "tool_correctness" not in names
