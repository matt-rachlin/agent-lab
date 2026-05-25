"""Tests for the per-cell system-prompt precedence in _build_messages."""

from __future__ import annotations

from lab.sweep.runner import _build_messages


def test_task_system_wins_over_model_default() -> None:
    msgs = _build_messages(
        {"input": "hi", "system": "TASK"},
        model_default_system="MODEL",
        config_system="CONFIG",
    )
    assert msgs[0] == {"role": "system", "content": "TASK"}
    assert msgs[-1] == {"role": "user", "content": "hi"}


def test_model_default_wins_over_config_default() -> None:
    msgs = _build_messages(
        {"input": "hi"},
        model_default_system="MODEL",
        config_system="CONFIG",
    )
    assert msgs[0] == {"role": "system", "content": "MODEL"}


def test_config_default_used_when_others_absent() -> None:
    msgs = _build_messages({"input": "hi"}, config_system="CONFIG")
    assert msgs[0] == {"role": "system", "content": "CONFIG"}


def test_no_system_at_all() -> None:
    msgs = _build_messages({"input": "hi"})
    assert len(msgs) == 1
    assert msgs[0] == {"role": "user", "content": "hi"}


def test_task_system_overrides_even_when_set_to_qwen_no_think() -> None:
    msgs = _build_messages(
        {"input": "q", "system": "/no_think"},
        model_default_system="should be ignored",
    )
    assert msgs[0]["content"] == "/no_think"
