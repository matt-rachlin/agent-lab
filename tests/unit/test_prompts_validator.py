"""Tests for Task.system / Task.system_prompt_id mutual-exclusion."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lab.tasks.registry import Task


def _task_kwargs(**extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "suite": "test-suite",
        "slug": "t1",
        "input": "hello",
    }
    base.update(extra)
    return base


def test_task_with_neither_is_valid() -> None:
    Task.model_validate(_task_kwargs())


def test_task_with_only_system_is_valid() -> None:
    t = Task.model_validate(_task_kwargs(system="you are an assistant"))
    assert t.system == "you are an assistant"
    assert t.system_prompt_id is None


def test_task_with_only_system_prompt_id_is_valid() -> None:
    t = Task.model_validate(_task_kwargs(system_prompt_id="agent_system_v1"))
    assert t.system is None
    assert t.system_prompt_id == "agent_system_v1"


def test_task_with_both_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Task.model_validate(
            _task_kwargs(
                system="you are an assistant",
                system_prompt_id="agent_system_v1",
            )
        )
    msg = str(excinfo.value)
    assert "system" in msg
    assert "system_prompt_id" in msg


def test_task_payload_carries_system_prompt_id() -> None:
    """Ensure the Task pydantic model serialises system_prompt_id."""
    t = Task.model_validate(_task_kwargs(system_prompt_id="agent_system_v1"))
    dumped = t.model_dump()
    assert dumped["system_prompt_id"] == "agent_system_v1"
    assert dumped["system"] is None
