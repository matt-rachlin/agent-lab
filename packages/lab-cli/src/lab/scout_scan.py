"""Scout driver loop (ADR-011): a bounded LiteLLM tool-use loop. NEW code (the
only existing loop is Inspect+sandbox-bound). Drives a model through the in-process
scout tools; audits every tool call; single-flight (global audit hash-chain)."""

from __future__ import annotations

import json
from typing import Any

from lab.core.control import record_action
from lab.core.llm import call_litellm_chat
from lab.core.settings import get_settings
from lab.scout import context_bundle
from lab.scout_tools import DISPATCH, TOOL_SCHEMAS

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


def run_scan(
    *,
    focus: str,
    model: str = "qwen3-4b-ft-toolcall-q4-latest",
    max_tool_calls: int = 24,
    max_recs: int = 6,
    timeout: int = 90,
) -> dict[str, Any]:
    settings = get_settings()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM + "\n\n" + context_bundle()},
        {"role": "user", "content": f"Scan focus: {focus}. Use the tools; log cited recs."},
    ]
    calls = 0
    added = 0
    for _ in range(max_tool_calls):
        resp, _ms = call_litellm_chat(
            settings=settings,
            litellm_key=settings.litellm_key,
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            extra={"think": False},
            timeout=timeout,
        )
        msg = ((resp.get("choices") or [{}])[0]).get("message") or {}
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break
        for tc in tool_calls:
            calls += 1
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            record_action(actor="scout", action=f"tool:{name}", args=args, outcome=None)
            try:
                result: Any = (
                    DISPATCH[name](**args) if name in DISPATCH else {"error": "unknown tool"}
                )
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            if name == "scout_add" and isinstance(result, dict) and result.get("result") == "added":
                added += 1
            messages.append(
                {"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(result)[:4000]}
            )
        if added >= max_recs or calls >= max_tool_calls:
            break
    return {"tool_calls": calls, "recs_added": added, "model": model}
