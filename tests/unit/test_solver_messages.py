"""Unit tests for the message-shaping helpers in `lab.inspect_bridge.solver`.

These are pure functions that don't need LiteLLM, MCP, or the sandbox.
"""

from __future__ import annotations

import json

import pytest

from lab.inspect_bridge.solver import (
    _chat_message_to_dict,
    _coerce_arguments,
    _extract_text_tool_calls,
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


# ---------------------------------------------------------------------------
# `_extract_text_tool_calls` — text-emitted tool-call recovery fallback.
#
# Some models (Llama-3.3-70B, Qwen2.5-Coder) emit tool calls as JSON *text*
# in message content instead of structured `tool_calls`. The solver recovers
# them (see the `if not tool_calls and content_text:` branch in the agent
# loop) and merges them in as OpenAI-format calls. These tests pin the
# recovery contract.
# ---------------------------------------------------------------------------

_VALID = {"fs_read", "fs_write", "shell_exec"}


def test_extract_plain_text_yields_nothing() -> None:
    assert _extract_text_tool_calls("I will now read the file for you.", _VALID) == []


def test_extract_valid_call_with_arguments_key() -> None:
    content = '{"name": "fs_read", "arguments": {"path": "data.csv"}}'
    out = _extract_text_tool_calls(content, _VALID)
    assert len(out) == 1
    call = out[0]
    # OpenAI-format contract: the agent loop feeds these straight into
    # `_execute_tool_calls`, which reads call["function"]["name"] and
    # JSON-decodes call["function"]["arguments"] via `_coerce_arguments`.
    assert call["id"] == "recovered_0"
    assert call["type"] == "function"
    assert call["function"]["name"] == "fs_read"
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"path": "data.csv"}


def test_extract_parameters_variant() -> None:
    content = '{"type": "function", "name": "shell_exec", "parameters": {"cmd": "ls"}}'
    out = _extract_text_tool_calls(content, _VALID)
    assert len(out) == 1
    assert out[0]["function"]["name"] == "shell_exec"
    assert json.loads(out[0]["function"]["arguments"]) == {"cmd": "ls"}


def test_extract_skips_unknown_tool_names() -> None:
    content = '{"name": "rm_rf_slash", "arguments": {"path": "/"}}'
    assert _extract_text_tool_calls(content, _VALID) == []


def test_extract_skips_object_without_arguments_or_parameters() -> None:
    # A JSON object that merely mentions a valid tool name is not a call.
    content = '{"name": "fs_read", "note": "I would call this"}'
    assert _extract_text_tool_calls(content, _VALID) == []


def test_extract_malformed_json_is_skipped_without_raising() -> None:
    content = '{"name": "fs_read", "arguments": {bad json,}'
    assert _extract_text_tool_calls(content, _VALID) == []


def test_extract_multiple_calls_in_order_with_sequential_ids() -> None:
    content = (
        'First: {"name": "fs_read", "arguments": {"path": "a.txt"}}\n'
        'Then: {"name": "fs_write", "arguments": {"path": "b.txt", "content": "hi"}}'
    )
    out = _extract_text_tool_calls(content, _VALID)
    assert [c["function"]["name"] for c in out] == ["fs_read", "fs_write"]
    assert [c["id"] for c in out] == ["recovered_0", "recovered_1"]


def test_extract_handles_nested_braces_in_arguments() -> None:
    content = '{"name": "fs_write", "arguments": {"path": "out.json", "content": "x"}}'
    out = _extract_text_tool_calls(content, _VALID)
    assert len(out) == 1
    assert json.loads(out[0]["function"]["arguments"]) == {
        "path": "out.json",
        "content": "x",
    }


def test_extract_finds_call_inside_markdown_fence_and_prose() -> None:
    content = (
        "Sure! I'll read the file. Here's the tool call:\n"
        "```json\n"
        '{"name": "fs_read", "arguments": {"path": "notes.md"}}\n'
        "```\n"
        "Let me know if you need anything else."
    )
    out = _extract_text_tool_calls(content, _VALID)
    assert len(out) == 1
    assert out[0]["function"]["name"] == "fs_read"
    assert json.loads(out[0]["function"]["arguments"]) == {"path": "notes.md"}


def test_extract_preserves_string_arguments_as_is() -> None:
    # Some models double-encode: arguments arrives as a JSON *string*. The
    # recovery keeps it verbatim (it is already the OpenAI wire shape).
    content = '{"name": "fs_read", "arguments": "{\\"path\\": \\"a.txt\\"}"}'
    out = _extract_text_tool_calls(content, _VALID)
    assert len(out) == 1
    assert out[0]["function"]["arguments"] == '{"path": "a.txt"}'
