---
doc_id: model-hermes-4-3-36b
title: nous / hermes-4.3 36b
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
# nous / hermes-4.3 36b

`litellm_id`: `hermes-4.3-36b` · backend: `llama.cpp` · vram_gb: `22` · context_max: `524288`

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

Source: <https://huggingface.co/bartowski/NousResearch_Hermes-4.3-36B-GGUF>
<!-- END AUTOGEN -->

## Description

<!-- BEGIN HAND -->
NousResearch's Hermes-4.3-36B — high-context generalist (claimed 512K context),
strong on tool-use, JSON mode, and structured output. The lab's choice for
long-context agent workloads. Built on Bytedance Seed; Apache 2.0.

family=hermes fmt=gguf source=bartowski/NousResearch_Hermes-4.3-36B-GGUF

### VRAM math (Q4_K_M)

- Weights on disk: ~20.3 GB Q4_K_M GGUF
- 36B dense — exceeds 12 GB on its own; needs hybrid offload (GPU layers + CPU
  layers) similar to the 70B
- KV cache at 8K ctx (Q8): ~1.4 GB
- KV cache at 64K ctx (Q8): ~11 GB — long-context use is RAM-bound, not VRAM-bound
- Headroom: needs `--n-gpu-layers ≈ 18-22` (research-based) to fit on 12 GB

### Recommended use

- Long-context agent tasks (>32K ctx) where qwen3-30b-a3b-moe's 40K ceiling is
  insufficient
- Tool-use cells that need reliable JSON output
- Roleplay / persona-driven workflows (Nous fine-tunes are strong here)

Not recommended for: latency-sensitive interactive use (hybrid offload halves
throughput vs full-GPU resident models).

### Throughput expectations

- 10-15 tok/s on RTX 3080 Ti with hybrid offload at 8K ctx
- Cold load: ~25-40 s from NVMe (20 GB), ~3-6 s from page cache

### License + source

- License: Apache 2.0 — commercial use OK
- Upstream: https://huggingface.co/NousResearch/Hermes-4.3-36B
- Local: `/data/models/gguf/hermes-4.3-36b/NousResearch_Hermes-4.3-36B-Q4_K_M.gguf`

### Verified by lab

Phase 19a smoke deferred to EXP-002b completion (GPU contention). GGUF download
verified; llama-cli load test pending GPU availability. Cold-load wall time
expected to dominate first invocation.
<!-- END HAND -->

## Known issues

<!-- BEGIN HAND -->
- **Single-source benchmarks** — Nous's published numbers (MATH-500 93.8% etc.)
  haven't been independently replicated. Treat with care; the lab should sanity-
  check before deferring to Hermes for headline results.
- 512K context is a *claim*; in practice usable context is limited by KV cache
  size and our 12 GB VRAM + 96 GB DDR5 budget. Realistic: 32K-64K.
- "hybrid-mode" tag in the Nous card refers to reasoning toggling, not
  retrieval-augmented mode — easy to confuse.
<!-- END HAND -->
