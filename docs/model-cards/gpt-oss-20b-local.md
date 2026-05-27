---
doc_id: model-gpt-oss-20b-local
title: openai / gpt-oss 20b
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
# openai / gpt-oss 20b

`litellm_id`: `gpt-oss-20b-local` · backend: `llama.cpp` · vram_gb: `14` · context_max: `131072`

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

Source: <https://huggingface.co/professorf/gpt-oss-20b-mxfp4-gguf>
<!-- END AUTOGEN -->

## Description

<!-- BEGIN HAND -->
OpenAI's 20B (21B total / 3.6B active MoE) reasoning model, native MXFP4 quantization.
A local alternative to `gpt-oss-20b-cloud` for offline / cost-sensitive lanes.
Apache 2.0 license; benchmarks claim near-o3-mini quality.

family=gptoss fmt=gguf-mxfp4 source=professorf/gpt-oss-20b-mxfp4-gguf

### VRAM math (MXFP4 native)

- Weights on disk: ~12.8 GB MXFP4 GGUF
- 21B total / 3.6B active per token
- KV cache at 8K ctx (Q8): ~0.4 GB
- Headroom: tight on 12 GB; expect expert-offload similar to qwen3-30b-a3b-moe

### Recommended use

- Reasoning + tool-use workloads where MXFP4 is acceptable
- A local sibling to gpt-oss-20b-cloud — useful for local-vs-cloud comparisons
  at the same nominal parameter count
- Offline-mode workflows where the cloud variant isn't reachable

### Throughput expectations

- 25-40 tok/s on RTX 3080 Ti expected (3.6B active, MXFP4)
- Cold load: ~10-20 s from NVMe; ~1-3 s from page cache

### License + source

- License: Apache 2.0 — commercial use OK
- Upstream weights: https://huggingface.co/openai/gpt-oss-20b
- Local: `/data/models/gguf/gpt-oss-20b-local/gpt-oss-20b-mxfp4.gguf`

### Verified by lab

Phase 19a smoke deferred to EXP-002b completion (GPU contention). GGUF download
verified; MXFP4 llama.cpp load test pending GPU availability. If load fails,
escalate to vLLM lane per the MXFP4 known-issue mitigations.
<!-- END HAND -->

## Known issues

<!-- BEGIN HAND -->
- **MXFP4 compat with llama.cpp can be flaky in some versions** (per Phase 19
  research, frameworks.md). Smoke before relying. If llama.cpp can't load the
  MXFP4 GGUF, vLLM serving is the documented fallback.
- Q4_K_M re-quant from BF16 reportedly degrades reasoning more than the same step
  on dense models — stick to MXFP4 native if the loader supports it.
- Active expert count (3.6B) means raw token speed is similar to a small dense
  model, but cold load is slower (full 12.8 GB to map).
<!-- END HAND -->
