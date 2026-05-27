---
doc_id: model-gpt-oss-120b-cloud
title: openai / gpt-oss 120b
zone: lab
kind: card
status: active
owner: m
created: '2026-05-25'
last_updated: '2026-05-26'
last_verified: '2026-05-26'
tags:
- lab
- card
- model-cards
---

<!-- BEGIN AUTOGEN -->
# openai / gpt-oss 120b

`litellm_id`: `gpt-oss-120b-cloud` · backend: `ollama-cloud` · vram_gb: `n/a` · context_max: `131072`

## Usage

Most-used in (last 30d, top 3):
- `EXP-001` — 192 run(s)
- `EXP-002` — 96 run(s)
- `EXP-003b` — 48 run(s)

## Performance (lifetime aggregate)

- runs: 336 (done=336, error=0)
- mean latency: 7598.0 ms
- mean tokens_in: 96.6
- mean tokens_out: 263.9
- mean cost: n/a

## References

- [F-003](../findings/F-003-*.md) — F-003: The 12 GB Agent v0.1 — three of four pre-registered hypotheses refuted
- [F-003](../findings/F-003-*.md) — EXP-001 verdicts — 144 cells, computed automatically
- [F-005](../findings/F-005-*.md) — F-005: The 12 GB Agent v0.2 — local tool-call is real; end-state difficulty is the binding constraint
- [F-005](../findings/F-005-*.md) — EXP-002 verdicts — 480 cells (480 done, 0 error)
- [F-006](../findings/F-006-*.md) — F-006: The lab RAG stack v0.1 — hybrid retrieval beats endpoints, locals depend on kb_query more than cloud

<!-- END AUTOGEN -->

## Description

<!-- BEGIN HAND -->
General-purpose cloud orchestrator.
<!-- END HAND -->

## Known issues

<!-- BEGIN HAND -->
_Hand-curated list. Add entries as they're discovered._
<!-- END HAND -->
