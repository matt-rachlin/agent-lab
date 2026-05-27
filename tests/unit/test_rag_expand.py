"""Phase 12 — multi_query parser + LLM-failure fallback.

We mock Ollama; tests stay CPU-only.
"""

from __future__ import annotations

from typing import Any

import pytest

from lab.rag.expand import (
    MAX_PHRASING_CHARS,
    _clean_phrasing,
    _parse_phrasings,
    multi_query,
)

# ---------- _clean_phrasing -------------------------------------------------


def test_clean_phrasing_strips_numbering_and_bullets() -> None:
    assert _clean_phrasing("1. how to redirect stderr") == "how to redirect stderr"
    assert _clean_phrasing("- merge two output streams") == "merge two output streams"


def test_clean_phrasing_rejects_empty_and_too_long() -> None:
    assert _clean_phrasing("") is None
    assert _clean_phrasing("   ") is None
    assert _clean_phrasing("x" * (MAX_PHRASING_CHARS + 10)) is None


def test_clean_phrasing_keeps_statements_without_question_mark() -> None:
    """Unlike HyPE, multi-query phrasings don't have to be questions."""
    assert _clean_phrasing("redirect stderr to stdout in bash") == (
        "redirect stderr to stdout in bash"
    )


# ---------- _parse_phrasings ------------------------------------------------


def test_parse_phrasings_dedupes() -> None:
    raw = "Redirect stderr to stdout\nredirect stderr to stdout\nmerge error and out"
    out = _parse_phrasings(raw, n_max=5)
    assert len(out) == 2


def test_parse_phrasings_respects_n_max() -> None:
    raw = "a\nbb\nccc\ndddd\neeeee\nffffff"
    out = _parse_phrasings(raw, n_max=2)
    assert len(out) == 2


# ---------- multi_query (mocked Ollama) -------------------------------------


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, *, model: str, messages: list[Any], options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"model": model, "messages": messages, "options": options})
        if not self._responses:
            return {"message": {"content": ""}}
        return {"message": {"content": self._responses.pop(0)}}


def test_multi_query_returns_original_plus_alternates() -> None:
    fake = _FakeClient(
        responses=[
            "1. redirect stderr to stdout in bash\n"
            "2. merge bash error stream into the output stream\n"
            "3. how to combine stderr and stdout"
        ]
    )
    out = multi_query("How do I redirect stderr to stdout?", n=3, client=fake)
    # Element 0 is the original.
    assert out[0] == "How do I redirect stderr to stdout?"
    # And we got 3 alternates → 4 total.
    assert len(out) == 4
    # Alternates were normalised (no numbering).
    for alt in out[1:]:
        assert not alt[0].isdigit()


def test_multi_query_degrades_to_original_on_complete_failure() -> None:
    """If both LLM attempts fail, we still return the original question."""
    fake = _FakeClient(responses=["", ""])
    out = multi_query("How do I exit a loop?", n=3, client=fake)
    assert out == ["How do I exit a loop?"]


def test_multi_query_filters_duplicates_against_original() -> None:
    fake = _FakeClient(
        responses=[
            "How do I exit a loop?\nbreak out of a loop\nstop a for loop early"
        ]
    )
    out = multi_query("How do I exit a loop?", n=3, client=fake)
    # The duplicate phrasing of the original is dropped.
    assert out[0] == "How do I exit a loop?"
    assert "How do I exit a loop?".lower() not in [a.lower() for a in out[1:]]
    assert len(out) == 3  # original + 2 unique alternates


def test_multi_query_empty_question_returns_empty() -> None:
    fake = _FakeClient(responses=["unused"])
    assert multi_query("", n=3, client=fake) == []
    assert fake.calls == []


def test_multi_query_n_zero_returns_just_original() -> None:
    fake = _FakeClient(responses=["unused"])
    assert multi_query("q", n=0, client=fake) == ["q"]
    assert fake.calls == []


def test_multi_query_exception_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RaisingClient:
        def __init__(self) -> None:
            self.n = 0

        def chat(self, **_kw: Any) -> dict[str, Any]:
            self.n += 1
            raise RuntimeError("ollama unreachable")

    cli = _RaisingClient()
    out = multi_query("How do I exit?", n=3, client=cli)
    assert out == ["How do I exit?"]
    # Both attempts tried (first + retry).
    assert cli.n == 2
