---
doc_id: model-llama3.1-8b-q4
title: ollama / llama3.1 8b
kind: card
status: active
owner: m
created: 2026-05-25
last_updated: 2026-05-26
litellm_id: llama3.1-8b-q4
backend: ollama-local
publisher: ollama
vram_gb: 4.9
context_max: null
capabilities: []
ollama_tag: "llama3.1:8b-instruct-q4_K_M"
source_url: null
license: null
known_issues: []
last_used_in: ['EXP-001', 'EXP-002', 'EXP-003b', 'RELIABILITY-001', 'SWEEP-SMOKE-001']
---

<!-- BEGIN AUTOGEN -->
# ollama / llama3.1 8b

`litellm_id`: `llama3.1-8b-q4` · backend: `ollama-local` · vram_gb: `4.9` · context_max: `n/a`

## Usage

Most-used in (last 30d, top 5):
- `EXP-001` — 192 run(s)
- `EXP-002` — 96 run(s)
- `EXP-003b` — 48 run(s)

## Performance (lifetime aggregate)

- runs: 402 (done=402, error=0)
- mean latency: 6359.0 ms
- mean tokens_in: 37.9
- mean tokens_out: 87.7
- mean cost: n/a

## References

- [F-001](../findings/F-001-*.md) — F-001: Phase 1 sweep harness produces persisted, queryable runs end-to-end
- [F-001](../findings/F-001-*.md) — Sweep report: `SWEEP-SMOKE-001`
- [F-002](../findings/F-002-*.md) — Sweep report: `RELIABILITY-001`
- [F-002](../findings/F-002-*.md) — F-002: pass^k reveals systematic wrong answers that pass@1 misreports as noise
- [F-003](../findings/F-003-*.md) — F-003: The 12 GB Agent v0.1 — three of four pre-registered hypotheses refuted
- [F-003](../findings/F-003-*.md) — EXP-001 verdicts — 144 cells, computed automatically
- [F-005](../findings/F-005-*.md) — F-005: The 12 GB Agent v0.2 — local tool-call is real; end-state difficulty is the binding constraint
- [F-005](../findings/F-005-*.md) — EXP-002 verdicts — 480 cells (480 done, 0 error)
- [F-006](../findings/F-006-*.md) — F-006: The lab RAG stack v0.1 — hybrid retrieval beats endpoints, locals depend on kb_query more than cloud

<!-- END AUTOGEN -->

## Description

<!-- BEGIN HAND -->
family=llama fmt=gguf
<!-- END HAND -->

## Known issues

<!-- BEGIN HAND -->
_Hand-curated list. Add entries as they're discovered._
<!-- END HAND -->
