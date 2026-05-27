---
doc_id: model-llama-3-3-70b-q4
title: meta / llama-3.3 70b
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
# meta / llama-3.3 70b

`litellm_id`: `llama-3.3-70b-q4` · backend: `llama.cpp` · vram_gb: `40` · context_max: `131072`

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

Source: <https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF>
<!-- END AUTOGEN -->

## Description

<!-- BEGIN HAND -->
Meta's Llama-3.3-70B-Instruct quantized to Q4_K_M — the lab's offline-only
**quality ceiling** model. Per Phase 19 strategic decisions: 6-10 tok/s with
hybrid GPU+CPU offload, useful only for non-interactive batch eval. Tagged
`slow_mode` so sweep runners refuse to use it unless `--allow-slow-models` is
explicit.

family=llama3 fmt=gguf source=bartowski/Llama-3.3-70B-Instruct-GGUF

### VRAM math (Q4_K_M hybrid offload)

- Weights on disk: ~39.6 GB Q4_K_M GGUF
- 70B dense — does NOT fit in 12 GB even quantized
- Research-best hybrid offload for this card: `--n-gpu-layers 21 --ctx-size 8192 -ctk q8_0 -ctv q8_0`
- KV cache at 8K ctx (Q8): ~2.5 GB
- VRAM resident: ~11 GB (21 layers @ Q4_K_M + KV + scratch)
- DDR5 resident: ~30 GB (remaining layers as CPU offload)

### Recommended use

- Offline batch reference cells (the "what if we had unlimited budget" baseline)
- Pre-registered comparison runs where ceiling is the question
- NOT for: interactive workflows, latency-sensitive cells, sweeps without
  `--allow-slow-models`

### Throughput expectations

- 6-10 tok/s on RTX 3080 Ti with the hybrid-offload command line above
- Cold load: 60-90 s from NVMe; ~10 s from page cache
- A typical PBS-Agent task (~1024 output tokens) = 100-170 s

### License + source

- License: Llama 3.3 community — commercial OK with restrictions (no training
  competing models on outputs, attribution required)
- Upstream: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct (gated upstream;
  bartowski's GGUF dump is unauth-accessible)
- Local: `/data/models/gguf/llama-3.3-70b-q4/Llama-3.3-70B-Instruct-Q4_K_M.gguf`

### Verified by lab

Phase 19a smoke deferred to EXP-002b completion (GPU contention and 40 GB
cold-load wall). Phase 19e will perform the formal smoke (load + one PBS-Agent
turn end-to-end) before EXP-006b.
<!-- END HAND -->

## Known issues

<!-- BEGIN HAND -->
- **Hybrid offload mode only on this hardware.** 6-10 tok/s wall throughput; a
  PBS-Agent-style multi-turn task can run 30-60s+ per turn. Be patient or skip.
- 21 GPU layers is the researched-best for 12 GB; auto-split heuristics can
  mis-allocate, so always pass the explicit flag.
- Llama 3.3 community license — commercial OK with restrictions (no
  training-on-outputs to make competing LLMs).
- Cold-load from NVMe is ~60-90 s (40 GB read); page-cache hit drops to ~10 s
  but it'll evict everything else.
<!-- END HAND -->
