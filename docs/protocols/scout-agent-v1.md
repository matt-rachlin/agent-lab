---
doc_id: scout-agent-v1
title: 'Spec: autonomous scout agent v1 (source-API tools + driver loop)'
zone: lab
kind: guide
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, guide, scout, spec]
---
# Spec: autonomous scout agent v1

Goal: `lab scout scan` runs the scout autonomously — the model searches source
APIs, verifies + cites, and logs deduped recs — within budget. (ADR-011.)

## Deliverables (v1 = NEW in-process layer; NOT the sandboxed lab.agent.tools)
- **D1 — in-process tools** (new module, e.g. `lab.scout.tools`; plain callables +
  hand-written OpenAI tool schemas):
  - `arxiv_search(query, max)` — httpx GET `export.arxiv.org/api/query`
    (`follow_redirects=True`; it 301s); parse Atom -> [{title,url,summary}].
  - `github_search(query, max)` — FIXED-ARG `gh search repos --json fullName,url,description --limit N -- <query>` (no passthrough); read-only PAT.
  - `fetch_url(url)` — in-process httpx GET, `follow_redirects=True`, browser UA,
    **host allowlist + SSRF guard** (block private/link-local IPs); returns
    extracted text (trafilatura best-effort) + final status.
  - `scout_add(source_url, title, category{enum}, why, confidence{low|medium|high})`
    — reachable-check (see Controls) then `lab.scout.add_recommendation`; returns
    `added|duplicate|unreachable` (surface to the model).
- **D2 — driver** `lab scout scan` (NEW bounded loop on `call_litellm_chat`):
  assemble `lab scout context` -> system prompt (mission + cite/dedup rules + budget)
  -> loop: POST with tools -> parse `tool_calls` -> dispatch in-process -> append
  `role:"tool"` results -> repeat until no tool_calls / budget / turn cap. (Shape
  from solver.py; not its code.) Flags: `--focus`, `--model`, `--max-recs`,
  `--max-tool-calls`, `--timeout`.
- **D3 — controls:** per-scan budget in the loop; audit each tool call via
  `lab.core.control.record_action(actor="scout", action="tool_call", ...)`;
  **single-flight** (global audit hash-chain); reachability + allowlist + SSRF in
  D1; driver spend via `control.budget_status`.

## Non-goals (v2)
General `web_search`/SearXNG; Inspect integration; #13 sandboxed-egress identity;
scheduled scans.

## Acceptance
1. `lab scout scan --focus "..."` runs autonomously and logs >=1 cited, verified,
   deduped recommendation, respecting the budget.
2. `scout_add` rejects an unreachable/hallucinated URL.
3. Tool calls are audited; no destructive tools available to the scan.

## Open questions
- Which local model drives reliably (qwen3-30b-a3b? qwen3-4b-ft? else cloud v1).
- arXiv API directly vs via `http_fetch` (keep a thin `arxiv_search` for clean parse).
