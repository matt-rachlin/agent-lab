---
doc_id: model-qwen3-14b-q4
title: ollama / qwen3 14b
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
# ollama / qwen3 14b

`litellm_id`: `qwen3-14b-q4` · backend: `ollama-local` · vram_gb: `9.3` · context_max: `n/a`

## Usage

Most-used in (last 30d, top 5):
- `EXP-001b` — 384 run(s)
- `EXP-001` — 192 run(s)
- `EXP-002` — 96 run(s)

## Performance (lifetime aggregate)

- runs: 783 (done=782, error=1)
- mean latency: 26370.3 ms
- mean tokens_in: 41.7
- mean tokens_out: 571.1
- mean cost: n/a

## References

- [F-001](../findings/F-001-*.md) — F-001: Phase 1 sweep harness produces persisted, queryable runs end-to-end
- [F-001](../findings/F-001-*.md) — Sweep report: `SWEEP-SMOKE-001`
- [F-002](../findings/F-002-*.md) — Sweep report: `RELIABILITY-001`
- [F-002](../findings/F-002-*.md) — F-002: pass^k reveals systematic wrong answers that pass@1 misreports as noise
- [F-003](../findings/F-003-*.md) — F-003: The 12 GB Agent v0.1 — three of four pre-registered hypotheses refuted
- [F-003](../findings/F-003-*.md) — EXP-001 verdicts — 144 cells, computed automatically
- [F-004](../findings/F-004-*.md) — F-004: qwen3-14b-q4 reasoning mode is net-negative on PBS-v0.1
- [F-005](../findings/F-005-*.md) — F-005: The 12 GB Agent v0.2 — local tool-call is real; end-state difficulty is the binding constraint
- [F-005](../findings/F-005-*.md) — EXP-002 verdicts — 480 cells (480 done, 0 error)
- [F-006](../findings/F-006-*.md) — F-006: The lab RAG stack v0.1 — hybrid retrieval beats endpoints, locals depend on kb_query more than cloud

<!-- END AUTOGEN -->

## Description

<!-- BEGIN HAND -->
family=qwen3 fmt=gguf
<!-- END HAND -->

## Known issues

<!-- BEGIN HAND -->
_Hand-curated list. Add entries as they're discovered._
<!-- END HAND -->
