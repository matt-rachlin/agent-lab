"""Task registry agent-field tests (max_turns, tool_budget, success_predicate, sandbox)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lab.tasks.registry import Task, load_tasks


def test_defaults_are_backwards_compatible() -> None:
    t = Task.model_validate({"suite": "ut", "slug": "a", "input": "x"})
    assert t.max_turns == 1
    assert t.tool_budget == 0
    assert t.success_predicate is None
    assert t.sandbox is None


def test_max_turns_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate({"suite": "ut", "slug": "a", "input": "x", "max_turns": 0})


def test_tool_budget_must_be_nonnegative() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate({"suite": "ut", "slug": "a", "input": "x", "tool_budget": -1})


def test_success_predicate_and_sandbox_accept_opaque_dicts() -> None:
    t = Task.model_validate(
        {
            "suite": "ut",
            "slug": "a",
            "input": "x",
            "success_predicate": {"type": "file_exists", "path": "/workspace/out.txt"},
            "sandbox": {"image": "lab-agent-sandbox:0.1", "network": "none"},
        }
    )
    assert t.success_predicate == {"type": "file_exists", "path": "/workspace/out.txt"}
    assert t.sandbox == {"image": "lab-agent-sandbox:0.1", "network": "none"}


def test_load_yaml_round_trip_with_agent_fields(tmp_path: Path) -> None:
    yaml_text = """
suite: ut-agent
tasks:
  - slug: a1
    input: "do a thing"
    max_turns: 5
    tool_budget: 3
    success_predicate:
      type: file_exists
      path: /workspace/answer.txt
    sandbox:
      image: lab-agent-sandbox:0.1
      network: none
"""
    p = tmp_path / "agent_tasks.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    rows = load_tasks(p)
    assert len(rows) == 1
    t = rows[0]
    assert t.max_turns == 5
    assert t.tool_budget == 3
    assert t.success_predicate is not None
    assert t.success_predicate["type"] == "file_exists"
    assert t.sandbox is not None
    assert t.sandbox["network"] == "none"
