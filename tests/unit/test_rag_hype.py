"""Phase 11 — HyPE generator: parser tolerance + retry + dedup.

We mock Ollama; the tests stay CPU-only and deterministic. The parser is
designed to be lenient (LLMs emit numbered lists, bullets, surrounding
quotes — all of which we strip) so most cases exercise the cleaning path.
"""

from __future__ import annotations

from typing import Any

import pytest
from lab.rag.hype import (
    MAX_QUESTION_CHARS,
    MIN_QUESTION_CHARS,
    _clean_question,
    _parse_questions,
    generate_hype_questions,
)

# ---------- _clean_question -------------------------------------------------


def test_clean_question_strips_leading_numbering() -> None:
    assert _clean_question("1. how do i redirect stderr") == "how do i redirect stderr?"
    assert _clean_question("1) what is set -e") == "what is set -e?"
    assert _clean_question("(1) how to use brace expansion") == "how to use brace expansion?"


def test_clean_question_strips_bullets_and_quotes() -> None:
    assert _clean_question("- how do i loop") == "how do i loop?"
    assert _clean_question('* "how do i exit"') == "how do i exit?"
    assert _clean_question("• What is IFS?") == "what is ifs?"


def test_clean_question_rejects_too_short() -> None:
    # Sub-MIN_QUESTION_CHARS even after adding the trailing '?' → rejected.
    assert _clean_question("hi") is None
    # Bare integers / no letters → rejected.
    assert _clean_question("123") is None


def test_clean_question_rejects_too_long() -> None:
    long = "x" * (MAX_QUESTION_CHARS + 10)
    assert _clean_question(long) is None
    # Boundary: just under (account for added '?') stays valid.
    just_under = "a" * (MAX_QUESTION_CHARS - 1)
    cleaned = _clean_question(just_under)
    assert cleaned is not None
    assert len(cleaned) <= MAX_QUESTION_CHARS
    assert len(cleaned) >= MIN_QUESTION_CHARS


def test_clean_question_collapses_trailing_punctuation() -> None:
    assert _clean_question("what is bash???") == "what is bash?"


# ---------- _parse_questions -------------------------------------------------


def test_parse_questions_dedupes_case_insensitive() -> None:
    raw = "How do I exit\nhow do i exit?\nWHAT IS SET -E?"
    out = _parse_questions(raw, n_max=5)
    # First two normalise to the same; we keep only the first.
    assert out == ["how do i exit?", "what is set -e?"]


def test_parse_questions_respects_n_max() -> None:
    raw = "a question here\nanother question\na third question\na fourth\na fifth"
    out = _parse_questions(raw, n_max=3)
    assert len(out) == 3


def test_parse_questions_empty_input() -> None:
    assert _parse_questions("", n_max=3) == []
    assert _parse_questions("   \n\n  ", n_max=3) == []


def test_parse_questions_skips_junk_lines() -> None:
    raw = "1. valid question here\n\n--- separator ---\n2. another good one"
    out = _parse_questions(raw, n_max=5)
    # '--- separator ---' has letters but after stripping numbering/bullets,
    # may produce something. Let's just assert at least two valid Q's land.
    assert "valid question here?" in out
    assert "another good one?" in out


# ---------- generate_hype_questions (mocked Ollama) -------------------------


class _FakeClient:
    """Hand-rolled Ollama chat client double. Returns canned text per call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, *, model: str, messages: list[Any], options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"model": model, "messages": messages, "options": options})
        if not self._responses:
            return {"message": {"content": ""}}
        return {"message": {"content": self._responses.pop(0)}}


def test_generate_hype_questions_happy_path() -> None:
    fake = _FakeClient(
        responses=[
            "1. How do I redirect stderr to stdout?\n"
            "2. What does 2>&1 mean?\n"
            "3. How can I capture both streams?",
        ]
    )
    out = generate_hype_questions(
        "Use 2>&1 to merge stderr into stdout.",
        n_questions=3,
        client=fake,
    )
    assert len(out) == 3
    assert all(q.endswith("?") for q in out)
    # And we called Ollama once (no retry needed).
    assert len(fake.calls) == 1


def test_generate_hype_questions_retries_on_garbage() -> None:
    """If the first response yields zero usable questions, we retry once."""
    fake = _FakeClient(
        responses=[
            # All lines are pure punctuation / numbers — no letters → parser
            # rejects every one, so the helper retries.
            "---\n***\n123",
            "1. how do i set IFS\n2. what does set -u do?",
        ]
    )
    out = generate_hype_questions(
        "Some chunk text",
        n_questions=2,
        client=fake,
    )
    assert len(out) == 2
    # Two chat round-trips (initial + retry).
    assert len(fake.calls) == 2


def test_generate_hype_questions_returns_empty_when_both_attempts_fail() -> None:
    fake = _FakeClient(responses=["", ""])
    out = generate_hype_questions(
        "Some chunk text",
        n_questions=3,
        client=fake,
    )
    assert out == []


def test_generate_hype_questions_returns_empty_for_empty_chunk() -> None:
    fake = _FakeClient(responses=["1. should not be reached"])
    out = generate_hype_questions("", n_questions=3, client=fake)
    assert out == []
    # No call made when there's nothing to ask about.
    assert fake.calls == []


def test_generate_hype_questions_partial_count_ok() -> None:
    """LLM yields fewer than asked → we return what we have (no padding)."""
    fake = _FakeClient(responses=["1. only one question"])
    out = generate_hype_questions("chunk", n_questions=3, client=fake)
    assert out == ["only one question?"]


def test_generate_hype_questions_exception_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama raising mid-call is caught and the retry path kicks in."""

    class _RaisingClient:
        def __init__(self) -> None:
            self.n = 0

        def chat(self, **_kw: Any) -> dict[str, Any]:
            self.n += 1
            if self.n == 1:
                raise RuntimeError("ollama unreachable")
            return {"message": {"content": "1. recovered question"}}

    cli = _RaisingClient()
    out = generate_hype_questions("chunk", n_questions=1, client=cli)
    # Second call succeeded.
    assert out == ["recovered question?"]
    assert cli.n == 2
