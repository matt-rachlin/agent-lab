---
doc_id: model-gemma3-12b-q4
title: ollama / gemma3 12b
kind: card
status: active
owner: m
created: 2026-05-25
last_updated: 2026-05-26
litellm_id: gemma3-12b-q4
backend: ollama-local
publisher: ollama
vram_gb: 8.1
context_max: null
capabilities: []
ollama_tag: "gemma3:12b-it-q4_K_M"
source_url: null
license: null
known_issues: []
last_used_in: ['EXP-001', 'RELIABILITY-001', 'SWEEP-SMOKE-001']
---

<!-- BEGIN AUTOGEN -->
# ollama / gemma3 12b

`litellm_id`: `gemma3-12b-q4` · backend: `ollama-local` · vram_gb: `8.1` · context_max: `n/a`

## Usage

Most-used in (last 30d, top 3):
- `EXP-001` — 192 run(s)
- `RELIABILITY-001` — 40 run(s)
- `SWEEP-SMOKE-001` — 28 run(s)

## Performance (lifetime aggregate)

- runs: 260 (done=258, error=2)
- mean latency: 3738.8 ms
- mean tokens_in: 38.9
- mean tokens_out: 94.6
- mean cost: n/a

## References

- [F-001](../findings/F-001-*.md) — F-001: Phase 1 sweep harness produces persisted, queryable runs end-to-end
- [F-001](../findings/F-001-*.md) — Sweep report: `SWEEP-SMOKE-001`
- [F-002](../findings/F-002-*.md) — Sweep report: `RELIABILITY-001`
- [F-002](../findings/F-002-*.md) — F-002: pass^k reveals systematic wrong answers that pass@1 misreports as noise
- [F-003](../findings/F-003-*.md) — F-003: The 12 GB Agent v0.1 — three of four pre-registered hypotheses refuted
- [F-003](../findings/F-003-*.md) — EXP-001 verdicts — 144 cells, computed automatically
- [F-004](../findings/F-004-*.md) — F-004: qwen3-14b-q4 reasoning mode is net-negative on PBS-v0.1

<!-- END AUTOGEN -->

## Description

<!-- BEGIN HAND -->
family=gemma3 fmt=gguf
<!-- END HAND -->

## Known issues

<!-- BEGIN HAND -->
_Hand-curated list. Add entries as they're discovered._
<!-- END HAND -->
