"""Tests for adapter-level prompt resolution.

The adapter's job: when ``task.system_prompt_id`` is set, resolve it via
:class:`PromptRegistry` and emit ``Sample.input`` as a list of
``[ChatMessageSystem, ChatMessageUser]`` so the system prompt lands in
``state.messages`` before the solver loop starts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from lab.eval.prompts import PromptNotFoundError, PromptRegistry
from lab.inspect_bridge.adapter import lab_task_to_inspect
from lab.tasks.registry import Task


def _write_prompt(root: Path, *, doc_id: str, body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    name = doc_id[len("prompt-") :] if doc_id.startswith("prompt-") else doc_id
    path = root / f"{name}.md"
    path.write_text(
        "---\n"
        f"doc_id: {doc_id}\n"
        "title: T\n"
        "zone: lab\n"
        "kind: prompt\n"
        "status: active\n"
        "owner: m\n"
        "created: 2026-05-27\n"
        "last_updated: 2026-05-27\n"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def _task(**overrides: Any) -> Task:
    base: dict[str, Any] = {
        "suite": "t",
        "slug": "s1",
        "input": "user-input",
    }
    base.update(overrides)
    return Task.model_validate(base)


async def _passthrough(state: Any, generate: Any) -> Any:
    return state


def test_adapter_resolves_prompt_id_into_system_message(tmp_path: Path) -> None:
    """Sample.input becomes [ChatMessageSystem(body), ChatMessageUser(input)]."""
    _write_prompt(tmp_path, doc_id="prompt-foo-v1", body="resolved-system")
    reg = PromptRegistry(root=tmp_path)
    inspect_task = lab_task_to_inspect(
        _task(system_prompt_id="foo_v1"),
        model="x",
        solver_override=_passthrough,
        prompt_registry=reg,
    )
    sample = next(iter(inspect_task.dataset))
    assert isinstance(sample.input, list)
    assert len(sample.input) == 2
    assert isinstance(sample.input[0], ChatMessageSystem)
    assert isinstance(sample.input[1], ChatMessageUser)
    assert "resolved-system" in str(sample.input[0].content)
    assert str(sample.input[1].content) == "user-input"


def test_adapter_stamps_prompt_id_used_metadata(tmp_path: Path) -> None:
    """The trajectory record needs to know which prompt body actually ran."""
    _write_prompt(tmp_path, doc_id="prompt-foo-v1", body="x")
    reg = PromptRegistry(root=tmp_path)
    inspect_task = lab_task_to_inspect(
        _task(system_prompt_id="foo_v1"),
        model="x",
        solver_override=_passthrough,
        prompt_registry=reg,
    )
    sample = next(iter(inspect_task.dataset))
    assert sample.metadata is not None
    assert sample.metadata["lab_prompt_id_used"] == "foo_v1"


def test_adapter_keeps_string_input_when_no_system(tmp_path: Path) -> None:
    """A task with neither system nor system_prompt_id keeps the simple shape."""
    reg = PromptRegistry(root=tmp_path)
    inspect_task = lab_task_to_inspect(
        _task(),
        model="x",
        solver_override=_passthrough,
        prompt_registry=reg,
    )
    sample = next(iter(inspect_task.dataset))
    # No system → input stays a plain string, prompt_id_used is None.
    assert sample.input == "user-input"
    assert sample.metadata is not None
    assert sample.metadata["lab_prompt_id_used"] is None


def test_adapter_passes_through_inline_system(tmp_path: Path) -> None:
    """A legacy `system` field still produces a system+user message pair."""
    reg = PromptRegistry(root=tmp_path)  # empty
    inspect_task = lab_task_to_inspect(
        _task(system="inline-sys"),
        model="x",
        solver_override=_passthrough,
        prompt_registry=reg,
    )
    sample = next(iter(inspect_task.dataset))
    assert isinstance(sample.input, list)
    assert isinstance(sample.input[0], ChatMessageSystem)
    assert str(sample.input[0].content) == "inline-sys"
    # No prompt id was used — only the inline string.
    assert sample.metadata is not None
    assert sample.metadata["lab_prompt_id_used"] is None


def test_adapter_raises_on_missing_prompt(tmp_path: Path) -> None:
    """Missing prompt → PromptNotFoundError at adapter build time, not runtime."""
    reg = PromptRegistry(root=tmp_path)  # empty
    with pytest.raises(PromptNotFoundError) as excinfo:
        lab_task_to_inspect(
            _task(system_prompt_id="absent_v1"),
            model="x",
            solver_override=_passthrough,
            prompt_registry=reg,
        )
    msg = str(excinfo.value)
    assert "absent_v1" in msg
    assert "t/s1" in msg
