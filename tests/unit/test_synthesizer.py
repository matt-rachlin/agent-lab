"""NS-3 Research Synthesizer v0 — pure unit tests. run_agent + get_settings are
monkeypatched: no GPU, no network, no live LLM. Asserts the anti-hallucination
property: citations are derived from URLs ACTUALLY fetched via fetch_url, and a
URL the model only *names* (never fetches) is excluded."""

from typing import Any

from lab.core.agent_runtime import AgentResult
from lab.synthesizer import _fetched_citations, search_tools, synthesize


class _FakeSettings:
    litellm_key = "k"


def _patch(monkeypatch: Any, result: AgentResult) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def _fake_run_agent(**kwargs: Any) -> AgentResult:
        captured.update(kwargs)
        return result

    def _fake_get_settings() -> _FakeSettings:
        return _FakeSettings()

    monkeypatch.setattr("lab.synthesizer.get_settings", _fake_get_settings)
    monkeypatch.setattr("lab.synthesizer.run_agent", _fake_run_agent)
    return captured


def _trajectory(answer: str) -> AgentResult:
    """search -> fetch X (200) -> final answer citing X.
    The agent also *names* a fabricated URL Y in prose but never fetched it."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "web_search"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "[]"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c2", "function": {"name": "fetch_url"}}],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "{}"},
        {"role": "assistant", "content": answer},
    ]
    tool_results: list[dict[str, Any]] = [
        {"name": "web_search", "args": {"query": "q"}, "result": []},
        {
            "name": "fetch_url",
            "args": {"url": "https://real.example/x"},
            "result": {"status": 200, "text": "real content"},
        },
    ]
    return AgentResult(
        messages=messages, tool_calls=2, tool_results=tool_results, stop_reason="stop"
    )


def test_synthesize_returns_answer_and_grounded_citations(monkeypatch: Any) -> None:
    answer = (
        "The answer is foo (https://real.example/x). "
        "Also see https://fabricated.example/y for more."
    )
    _patch(monkeypatch, _trajectory(answer))
    out = synthesize(question="what is foo?")
    assert out["answer"] == answer
    # Citation == the URL actually fetched, NOT the fabricated one named in prose.
    assert out["citations"] == ["https://real.example/x"]
    assert "https://fabricated.example/y" not in out["citations"]
    assert out["tool_calls"] == 2
    assert out["stop"] == "stop"


def test_fabricated_citation_excluded() -> None:
    # fetch_url that FAILED (status 0) must not become a citation either.
    tool_results: list[dict[str, Any]] = [
        {
            "name": "fetch_url",
            "args": {"url": "https://real.example/x"},
            "result": {"status": 200, "text": "ok"},
        },
        {
            "name": "fetch_url",
            "args": {"url": "https://dead.example/z"},
            "result": {"status": 0, "error": "boom", "text": ""},
        },
    ]
    cites = _fetched_citations(tool_results)
    assert cites == ["https://real.example/x"]
    assert "https://dead.example/z" not in cites


def test_citations_deduped_and_ordered() -> None:
    tool_results: list[dict[str, Any]] = [
        {"name": "fetch_url", "args": {"url": "https://a"}, "result": {"status": 200}},
        {"name": "fetch_url", "args": {"url": "https://b"}, "result": {"status": 301}},
        {"name": "fetch_url", "args": {"url": "https://a"}, "result": {"status": 200}},
        {"name": "web_search", "args": {"query": "q"}, "result": []},
    ]
    assert _fetched_citations(tool_results) == ["https://a", "https://b"]


def test_search_tools_are_read_only_no_scout_add() -> None:
    tools = {t.name: t for t in search_tools()}
    assert set(tools) == {"web_search", "arxiv_search", "github_search", "fetch_url"}
    assert "scout_add" not in tools
    assert all(t.side_effect == "external_read" for t in tools.values())


def test_synthesize_uses_read_only_side_effects(monkeypatch: Any) -> None:
    captured = _patch(monkeypatch, _trajectory("a (https://real.example/x)"))
    synthesize(question="q")
    # Default read-only gate: no write_local/irreversible authorized.
    allowed = captured.get("allow_side_effects")
    if allowed is not None:
        assert "write_local" not in allowed
        assert "irreversible" not in allowed
    assert captured["actor"] == "synthesizer"
    names = {t.name for t in captured["tools"]}
    assert "scout_add" not in names
