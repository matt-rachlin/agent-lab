"""Unit tests for `_extract_assistant_text` (solver assistant-content extraction).

Forensic-audit follow-up: gemma4-12b episodes logged empty `content` /
`content_preview` on every turn while `tokens_out` ran into the hundreds or
thousands. Root cause (verified with a live call through the LiteLLM proxy):
the gemma4 lane is `ollama_chat/gemma4:12b`, a thinking model — LiteLLM maps
Ollama's `thinking` field to `reasoning_content` and returns `content` as an
empty string on think/tool-call turns:

    {"message": {"content": "", "role": "assistant",
                 "reasoning_content": "The user is asking for ..."}}

The old extraction (`message.get("content") or ""`) silently dropped the
reasoning. The fix keeps content extraction defensive (string, None,
list-of-parts) and surfaces `reasoning_content` as its own value — never
merged into content.
"""

from __future__ import annotations

from typing import Any

from lab.inspect_bridge.solver import _extract_assistant_text


def test_plain_string_content_no_reasoning() -> None:
    """The common lane shape (qwen3-coder, devstral, glm): content is a string."""
    content, reasoning = _extract_assistant_text(
        {"role": "assistant", "content": "The answer is 42."}
    )
    assert content == "The answer is 42."
    assert reasoning == ""


def test_gemma4_shape_empty_content_with_reasoning_content() -> None:
    """The real gemma4-12b (ollama_chat) shape, captured live via the proxy:
    empty-string content plus a populated `reasoning_content`."""
    msg = {
        "content": "",
        "role": "assistant",
        "reasoning_content": "The user is asking for the value of $17 \\times 23$.",
    }
    content, reasoning = _extract_assistant_text(msg)
    assert content == ""
    assert reasoning == "The user is asking for the value of $17 \\times 23$."


def test_none_content_with_reasoning_content() -> None:
    """Some providers return content=None (not "") on pure tool-call turns."""
    msg = {
        "content": None,
        "role": "assistant",
        "reasoning_content": "I should call the fs_read tool first.",
        "tool_calls": [{"id": "call_1", "type": "function"}],
    }
    content, reasoning = _extract_assistant_text(msg)
    assert content == ""
    assert reasoning == "I should call the fs_read tool first."


def test_list_of_parts_content_is_flattened() -> None:
    """Multi-part content (`[{"type": "text", "text": ...}, ...]`) flattens
    to the part texts, in order."""
    msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "First part."},
            {"type": "text", "text": "Second part."},
        ],
    }
    content, reasoning = _extract_assistant_text(msg)
    assert content == "First part.\nSecond part."
    assert reasoning == ""


def test_list_of_parts_skips_non_text_parts() -> None:
    """Non-text parts (image refs, malformed entries) are skipped, bare
    strings inside the list are kept."""
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": [
            "bare string part",
            {"type": "image_url", "image_url": {"url": "http://x/img.png"}},
            {"type": "text", "text": "trailing text"},
            42,
        ],
    }
    content, reasoning = _extract_assistant_text(msg)
    assert content == "bare string part\ntrailing text"
    assert reasoning == ""


def test_missing_content_key() -> None:
    content, reasoning = _extract_assistant_text({"role": "assistant"})
    assert content == ""
    assert reasoning == ""


def test_non_string_reasoning_content_is_ignored() -> None:
    """A defensive guard: reasoning_content of an unexpected type reads as empty."""
    msg = {"role": "assistant", "content": "ok", "reasoning_content": {"weird": True}}
    content, reasoning = _extract_assistant_text(msg)
    assert content == "ok"
    assert reasoning == ""


def test_reasoning_is_not_merged_into_content() -> None:
    """Both present → both returned, kept separate (do NOT concatenate)."""
    msg = {
        "role": "assistant",
        "content": "Final answer: 391.",
        "reasoning_content": "17*23 = 17*20 + 17*3 = 340 + 51 = 391.",
    }
    content, reasoning = _extract_assistant_text(msg)
    assert content == "Final answer: 391."
    assert reasoning == "17*23 = 17*20 + 17*3 = 340 + 51 = 391."
    assert reasoning not in content
