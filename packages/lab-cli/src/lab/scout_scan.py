"""Scout driver (ADR-011) — now a thin caller of the Lab Agent Runtime
(ADR-012, lab.platform.agent_runtime). The bounded tool-loop, tool dispatch, audit,
and the side-effect authorization gate all live in the shared runtime; this module
supplies only the scout's system prompt, tools, and rec-count stop condition.
Proves the runtime end-to-end (the scout is its first caller)."""

from __future__ import annotations

from typing import Any

from lab.platform.agent_runtime import run_agent

from lab.core.settings import get_settings
from lab.scout import context_bundle
from lab.scout_tools import build_tools

_SYSTEM = """You are the lab's research scout. Read the lab context below; it is
your relevance filter. Find work HIGHLY relevant to this lab (local
tool-calling/agent models for ~12GB, eval/trust/verification methods, agent
scaffolds, deployable local agents replacing cloud agents).

Tools: use web_search FIRST for broad discovery across the general web (blogs,
release notes, news, docs, forums); use arxiv_search for papers and
github_search for code/repos. For each promising hit: fetch_url to verify the
source exists and read it, then scout_add a CITED, deduped recommendation (use a
real URL you actually fetched; set confidence honestly).

Pick the MOST SPECIFIC category for each rec: model (a specific model/release),
architecture (a system or agent design), software (a tool/library/framework),
paper (a research finding), method (a technique/recipe), benchmark (an
eval/dataset). Do NOT default everything to 'software'.

Skip anything already in the dedup list or low-relevance. Do several searches
across different sources, add the best findings, then stop. Be selective —
quality over quantity."""


def _recs_added(results: list[dict[str, Any]]) -> int:
    return sum(
        1
        for r in results
        if r["name"] == "scout_add"
        and isinstance(r["result"], dict)
        and r["result"].get("result") == "added"
    )


def run_scan(
    *,
    focus: str,
    model: str = "qwen3-4b-ft-toolcall-q4-latest",
    max_tool_calls: int = 24,
    max_recs: int = 6,
    timeout: int = 90,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    res = run_agent(
        settings=settings,
        litellm_key=settings.litellm_key,
        model=model,
        system=_SYSTEM + "\n\n" + context_bundle(),
        user=f"Scan focus: {focus}. Use the tools; log cited recs.",
        tools=build_tools(),
        actor="scout",
        # scout may search the web AND log recs (write_local); not irreversible.
        allow_side_effects=frozenset({"read", "external_read", "write_local"}),
        max_turns=max_tool_calls,
        max_tool_calls=max_tool_calls,
        timeout=timeout,
        num_ctx=num_ctx,
        stop_predicate=lambda results: _recs_added(results) >= max_recs,
    )
    return {
        "tool_calls": res.tool_calls,
        "recs_added": _recs_added(res.tool_results),
        "model": model,
        "stop": res.stop_reason,
    }
