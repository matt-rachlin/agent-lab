---
doc_id: model-phi-4-reasoning-14b
title: microsoft / phi-4-reasoning-plus 14b
zone: lab
kind: card
status: active
owner: m
created: 2026-05-27
last_updated: 2026-05-27
last_verified: 2026-05-27
tags:
- lab
- card
- model-cards
- phase-19a
---

<!-- BEGIN AUTOGEN -->
# microsoft / phi-4-reasoning-plus 14b

`litellm_id`: `phi-4-reasoning-14b` · backend: `llama.cpp` · vram_gb: `9` · context_max: `32768`

## Usage

No `experiment_runs` rows in the last 30 days.

## Performance (lifetime aggregate)

- runs: 0 (done=0, error=0)
- mean latency: n/a
- mean tokens_in: n/a
- mean tokens_out: n/a
- mean cost: n/a

## References

No findings cite this model yet.

Source: <https://huggingface.co/bartowski/microsoft_Phi-4-reasoning-plus-GGUF>
<!-- END AUTOGEN -->

## Description

<!-- BEGIN HAND -->
Microsoft's Phi-4-reasoning-plus — a 14B dense reasoning specialist. Plan name was
`Phi-4-Reasoning` but the +plus variant (longer reasoning tokens, MIT license)
is the lab-relevant one. Tracks `bartowski/microsoft_Phi-4-reasoning-plus-GGUF`.

family=phi4 fmt=gguf source=bartowski/microsoft_Phi-4-reasoning-plus-GGUF

### VRAM math (Q4_K_M)

- Weights on disk: ~8.6 GB Q4_K_M GGUF
- 14B dense
- KV cache at 8K ctx (Q8): ~0.6 GB
- Headroom: ~2 GB on a 12 GB card — tight but workable; consider 4K ctx for
  multi-model swap groups

### Recommended use

- Math / quantitative reasoning tasks (AIME, MATH-500)
- Long-form reasoning workflows where reasoning-token budget matters
- Cells where qwen3-14b-q4 reasoning-ON is net-negative (F-004) but a reasoning
  model is still wanted

### Throughput expectations

- 30-50 tok/s on RTX 3080 Ti (14B dense Q4_K_M)
- Cold load: ~5-10 s from NVMe; <1 s from page cache

### License + source

- License: MIT — commercial use OK
- Upstream: https://huggingface.co/microsoft/Phi-4-reasoning-plus
- Local: `/data/models/gguf/phi-4-reasoning-14b/microsoft_Phi-4-reasoning-plus-Q4_K_M.gguf`

### Verified by lab

Phase 19a smoke deferred to EXP-002b completion (GPU contention). GGUF download
verified; llama-cli load test pending GPU availability.
<!-- END HAND -->

## Known issues

<!-- BEGIN HAND -->
- **Microsoft's benchmark claims (beats DeepSeek-R1-Distill-Llama-70B on AIME 2025)
  are single-source.** Replicate before relying. Phase 19d / EXP-006 follow-on may
  include this.
- Reasoning-mode token consumption is high; budget `max_tokens` accordingly (2K+).
- "plus" variant differs from base Phi-4-reasoning by a longer-reasoning preference
  bake; if you want shorter chains use the base variant instead.
<!-- END HAND -->
