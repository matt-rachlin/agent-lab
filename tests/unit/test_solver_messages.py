"""Unit tests for the message-shaping helpers in `lab.inspect_bridge.solver`.

These are pure functions that don't need LiteLLM, MCP, or the sandbox.
"""

from __future__ import annotations

import json

import pytest

from lab.inspect_bridge.solver import (
    _chat_message_to_dict,
    _coerce_arguments,
    _serialise_messages,
    _truncate,
)


def test_truncate_returns_value_when_small() -> None:
    assert _truncate({"a": 1}) == {"a": 1}
    assert _truncate("short") == "short"


def test_truncate_emits_preview_when_too_large() -> None:
    big = "x" * 9000
    out = _truncate(big, cap=100)
    assert isinstance(out, dict)
    assert out["_truncated"] is True
    assert out["original_size"] == json.dumps(big).__len__()
    assert len(out["preview"]) == 100


def test_coerce_arguments_accepts_dict() -> None:
    assert _coerce_arguments({"a": 1}) == {"a": 1}


def test_coerce_arguments_parses_json_string() -> None:
    assert _coerce_arguments('{"a": 1}') == {"a": 1}


def test_coerce_arguments_empty_string_yields_empty_dict() -> None:
    assert _coerce_arguments("") == {}
    assert _coerce_arguments("   ") == {}


def test_coerce_arguments_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        _coerce_arguments("[1, 2]")


def test_coerce_arguments_rejects_bad_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        _coerce_arguments("{not-json}")


def test_chat_message_to_dict_passes_through_dicts() -> None:
    msg = {"role": "user", "content": "hi"}
    assert _chat_message_to_dict(msg) == msg


def test_chat_message_to_dict_handles_pydantic_assistant_like_object() -> None:
    class _Msg:
        role = "assistant"
        content = "answer"
        tool_calls = None

    assert _chat_message_to_dict(_Msg()) == {"role": "assistant", "content": "answer"}


def test_serialise_messages_is_idempotent_for_plain_dicts() -> None:
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    out = _serialise_messages(msgs)
    assert out == msgs
    # ensure we got copies, not aliases
    out[0]["content"] = "MUTATED"
    assert msgs[0]["content"] == "x"
