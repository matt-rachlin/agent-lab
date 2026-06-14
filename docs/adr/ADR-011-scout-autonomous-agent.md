---
doc_id: adr-011-scout-autonomous-agent
title: 'ADR-011: Scout as an autonomous lab agent (does its own web search)'
zone: lab
kind: adr
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, adr, research-agent, scout]
---
# ADR-011: Scout as an autonomous lab agent

Status: proposed
Date: 2026-06-14
Deciders: Matt Rachlin

## Context
The scout (ADR-010) should search the web ITSELF, not be human-driven. What
exists: the lab agent-tool framework (`lab.agent.tools` + tool_pool), `http_fetch`
(URL->content), `shell_exec` (-> `gh`), and the `lab scout` store/context. The only
missing primitive is **general web SEARCH** (query->URLs); the high-relevance
sources (arXiv, HF, GitHub) are reachable via public APIs / `gh` with no new
dependency.

## Decision
**The scout is an autonomous tool-use agent**, phased:
- **v1 (this build):** a bounded driver loop (`lab scout scan`) that gives a model
  source-API tools — `arxiv_search`, `github_search`, `fetch_url`, `scout_add` —
  and lets it search, verify, and log cited recs. **v1 is a NEW, in-process,
  non-sandboxed tool layer** (plain Python callables + hand-written OpenAI tool
  schemas) and a **new bounded loop built directly on `call_litellm_chat`** — it
  does NOT reuse the sandboxed `lab.agent.tools` (FastMCP-over-podman) mechanism
  or `inspect_bridge/solver.py` (those are sandbox-bound; they are the v2 path).
  Loop shape is borrowed from solver.py (turn cap, budget decrement, text-tool-call
  recovery), not its code. No general search engine needed for v1 (source APIs
  cover papers/models/repos = the high-relevance 80%). Driven via LiteLLM (model
  configurable).
- **v2 (deferred):** graduate to the Inspect agent path; add a `web_search` tool
  backed by **self-hosted SearXNG on m-box** (local, no API key) for general web
  (blogs/news); run inside the #13 sandboxed identity with allowlisted egress.

**Driver model:** configurable. **Local is the goal (dogfood the charter)** — the
scout is itself a deployment target: when a local model clears the scoreboard bar
it takes the driver seat, replacing a cloud driver. Cloud is acceptable v1 as
internal tooling. (This makes the scout the first concrete instance of the
mission: a local agent driving a real workflow.)

**Trust + control:**
- **Cited-fetch-before-add:** `scout_add` first checks the `source_url` is
  reachable — HTTP status in {200,301,302,303,307,308} after redirects, with a
  realistic browser UA; 403/429 = reachable-but-blocked -> accept with a flag (do
  NOT gate on content extraction; bot-walls would false-reject arXiv PDFs/HF/blogs).
- **Egress allowlist + SSRF guard (v1, even without the sandbox):** `fetch_url`
  permits only allow-listed hosts (arxiv.org, export.arxiv.org, huggingface.co,
  github.com, raw.githubusercontent.com, + the queried domain) and blocks
  private/link-local IP ranges (no `m-box:5050`, `169.254.169.254`, etc.).
- **Budgets:** per-scan caps (max tool calls, max recs, wall-clock); driver LLM
  spend routed through the existing `lab.core.control` budget plane.
- **Audit:** every tool call logged via `lab.core.control.record_action`
  (actor="scout"); scans are **single-flight** (the audit hash-chain is global +
  unsynchronised, so concurrent scans would fork it).
- **No destructive surface:** in-process tools only — `arxiv_search`,
  `github_search` (a FIXED-ARG `gh search repos --json` wrapper, never a `gh`/shell
  passthrough; run under a read-only fine-grained PAT, not the omni-scope keyring
  token), `fetch_url`, `scout_add`. No fs-write/shell/arbitrary-gh. v2 adds the
  sandboxed egress-allowlisted identity (#13).

## Consequences
- Scout becomes the lab's **first autonomous agent driving a real workflow**
  (mission milestone) and dogfoods the agent-tool + LiteLLM stack.
- Web search is the one new primitive — deferred to v2 (SearXNG); v1 ships value
  on source APIs alone.
- Risk: a weak local model may not drive a multi-tool loop reliably — v1 keeps the
  loop simple + bounded and the driver configurable, scans single-flight; "can model X drive the scout"
  is itself a measurable question for the scoreboard.

## Considered alternatives
- A web-search MCP server — viable; folded into v2 (SearXNG behind a tool).
- Hosted search API (Tavily/Brave) — rejected as default (local-first/cost);
  allowed as an optional `web_search` backend.
- Keep the scout human-driven — rejected (the ask is autonomy).
