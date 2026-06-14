"""NS-3 Research Synthesizer v0 (ADR-012 / ADR-014) — a read-only Lab Agent
Runtime caller that ANSWERS a question with cited synthesis, rather than logging
recommendations like the scout (ADR-011).

It mirrors the scout's thin-caller pattern (`lab.scout_scan`): it supplies a
system prompt, a tool list, and a return shape, and delegates the bounded
tool-use loop, audit, and side-effect gate to `lab.core.agent_runtime.run_agent`.

Tools: the scout's SEARCH/FETCH impls only (web_search, arxiv_search,
github_search, fetch_url — all side_effect="external_read"), imported from
`lab.scout_tools` and re-wrapped as `Tool`s by `search_tools()`. The synthesizer
deliberately does NOT include `scout_add`: it answers, it does not write recs.
The run therefore uses the default read-only authorization
(allow_side_effects = {read, external_read}); no write/irreversible class is ever
authorized.

KEY DESIGN POINT — citation grounding (the NS-3 anti-hallucination property):
the returned `citations` are NOT parsed from whatever the model wrote in its
answer. They are the set of URLs the agent ACTUALLY fetched via `fetch_url`
during the run (read back from `run_agent`'s `tool_results`, restricted to
fetches that actually succeeded). Every cited source is therefore a real,
verified retrieval. This is also the eval signal for NS-3: a synthesis is
trustworthy to the degree the URLs it claims in prose correspond to URLs in this
grounded fetch set; URLs the model names but never fetched are, by construction,
excluded from `citations`.
"""

from __future__ import annotations

from typing import Any

from lab.core.agent_runtime import Tool, run_agent
from lab.core.settings import get_settings
from lab.scout_tools import arxiv_search, fetch_url, github_search, web_search

#: fetch_url returns these HTTP status codes when a URL was genuinely retrieved.
#: Used to ground citations in *successful* fetches only.
_FETCH_OK = frozenset({200, 301, 302, 303, 307, 308})

_INT: dict[str, str] = {"type": "integer"}

_SYSTEM = """You are the lab's research synthesizer. Your job is to ANSWER the
user's question with a well-sourced synthesis — NOT to log recommendations.

Method:
1. Search broadly first with web_search (blogs, docs, news, forums); use
   arxiv_search for papers and github_search for code/repos.
2. For every source you intend to rely on, call fetch_url to retrieve and read
   it. Only facts you have actually fetched may be used.
3. Then write a SYNTHESIS that directly answers the question. Every claim must
   carry an inline CITATION to the URL you fetched to support it, e.g.
   "X is true (https://example.com/page)". Do not cite a URL you did not fetch.

Be accurate and concise. If the sources do not answer the question, say so
rather than guessing. When you have enough verified material, write the final
answer as a plain assistant message with no further tool calls."""


def search_tools() -> list[Tool]:
    """The synthesizer's tools: the scout's SEARCH/FETCH impls only, wrapped as
    ADR-012 Tool ABI instances. All external_read; NO scout_add (no writes)."""
    return [
        Tool(
            name="web_search",
            description=(
                "General-web search (blogs, news, docs, forums) via SearXNG. Use this "
                "first for broad discovery; arxiv_search/github_search for papers/code."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": _INT,
                    "categories": {"type": "string"},
                },
                "required": ["query"],
            },
            impl=web_search,
            side_effect="external_read",
        ),
        Tool(
            name="arxiv_search",
            description="Search arXiv for papers.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}, "max_results": _INT},
                "required": ["query"],
            },
            impl=arxiv_search,
            side_effect="external_read",
        ),
        Tool(
            name="github_search",
            description="Search GitHub repos.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}, "max_results": _INT},
                "required": ["query"],
            },
            impl=github_search,
            side_effect="external_read",
        ),
        Tool(
            name="fetch_url",
            description="Fetch + extract a public URL to verify/quote a source.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            impl=fetch_url,
            side_effect="external_read",
        ),
    ]


def _final_answer(messages: list[dict[str, Any]]) -> str:
    """The last assistant message with text content and no tool calls = the
    synthesis. Falls back to the last assistant message with any content."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip() and not msg.get("tool_calls"):
            return content.strip()
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _fetched_citations(tool_results: list[dict[str, Any]]) -> list[str]:
    """Grounded citation set: URLs the agent ACTUALLY fetched via fetch_url and
    that returned successfully. Order-preserving, deduped. This is the NS-3
    anti-hallucination boundary — citations come from real fetches, never from
    text the model wrote."""
    seen: set[str] = set()
    out: list[str] = []
    for r in tool_results:
        if r.get("name") != "fetch_url":
            continue
        url = (r.get("args") or {}).get("url")
        if not isinstance(url, str) or not url:
            continue
        result = r.get("result")
        if not isinstance(result, dict) or result.get("status") not in _FETCH_OK:
            continue
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def synthesize(
    *,
    question: str,
    model: str = "qwen3-4b-ft-toolcall-q4-latest",
    max_tool_calls: int = 16,
    timeout: int = 90,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Answer `question` with cited synthesis via the Lab Agent Runtime.

    Read-only research: `allow_side_effects` is left at the runtime default
    ({read, external_read}), so no write/irreversible tool could run even if one
    were present. Returns the parsed final answer, the GROUNDED citation set
    (URLs actually fetched — see module docstring), the tool-call count, and the
    run's stop reason."""
    settings = get_settings()
    res = run_agent(
        settings=settings,
        litellm_key=settings.litellm_key,
        model=model,
        system=_SYSTEM,
        user=f"Question: {question}\n\nResearch and answer it with inline citations.",
        tools=search_tools(),
        actor="synthesizer",
        # default allow_side_effects = {read, external_read}: read-only research.
        max_turns=max_tool_calls,
        max_tool_calls=max_tool_calls,
        timeout=timeout,
        num_ctx=num_ctx,
    )
    return {
        "answer": _final_answer(res.messages),
        "citations": _fetched_citations(res.tool_results),
        "tool_calls": res.tool_calls,
        "stop": res.stop_reason,
    }
