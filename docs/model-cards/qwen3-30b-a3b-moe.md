---
doc_id: model-qwen3-30b-a3b-moe
title: qwen / qwen3-30b-a3b 30b
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
# qwen / qwen3-30b-a3b 30b

`litellm_id`: `qwen3-30b-a3b-moe` · backend: `llama.cpp` · vram_gb: `17` · context_max: `40960`

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

Source: <https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF>
<!-- END AUTOGEN -->

## Description

<!-- BEGIN HAND -->
Qwen3-30B-A3B MoE (30B total parameters, 3B active per token). Designated Phase 19's
headline local model: ArenaHard 91.0 (above GPT-4o per the Qwen3 paper), Apache 2.0,
and with expert offload via `--n-cpu-moe` it reaches 20-25 tok/s on consumer 12 GB
cards. Replaces `qwen3-14b-q4` as the lab default once EXP-006 (Phase 19d) confirms
the gap-closure hypothesis.

family=qwen3moe fmt=gguf source=unsloth/Qwen3-30B-A3B-GGUF

### VRAM math (Q4_K_M)

- Weights on disk: ~17.3 GB Q4_K_M GGUF
- Active per token: 3B (Q4_K_M ≈ 1.8 GB)
- With `--n-cpu-moe` expert offload: shared layers + active experts fit in 12 GB; non-active expert weights stay in DDR5
- KV cache at 8K ctx (Q8): ~0.6 GB
- Headroom: comfortable on a 12 GB card with the expert-offload flag

### Recommended use

- PBS-Agent v0.1 + v0.2 suites — primary local-tier candidate
- Tool-use cells where qwen3-14b-q4 underperforms cloud
- Long-form planning tasks (40K context window)
- Default fallback for kb_query / RAG-augmented workflows

Not recommended for: math-heavy reasoning where Phi-4-Reasoning is a better fit;
strict function-calling-only workloads (xLAM-2 is the specialist there if/when it
lands).

### Throughput expectations

- 20-25 tok/s expected on RTX 3080 Ti with `--n-cpu-moe` (research-based; not yet
  measured here)
- Cold-load from NVMe ≈ 8-15 s; from page cache ≈ 0.5-2 s (per Phase 19c
  pre-flight discipline)

### License + source

- License: Apache 2.0 — commercial use OK
- Local: `/data/models/gguf/qwen3-30b-a3b-moe/Qwen3-30B-A3B-Q4_K_M.gguf`

### Verified by lab

Phase 19a smoke deferred to EXP-002b completion (GPU contention). GGUF download
verified; llama-cli load test pending GPU availability.
<!-- END HAND -->

## Known issues

<!-- BEGIN HAND -->
- **Expert-offload mechanics depend on llama.cpp `--n-cpu-moe` or `-ot "exps=CPU"`** —
  Ollama does not expose these flags. Must run via direct llama.cpp / llama-server.
- 12 GB-card throughput claims (20-25 tok/s) come from 24 GB reports in practitioner
  posts; Phase 19a pilot SHOULD confirm before EXP-006 dispatches with 288 cells.
- imatrix variants may differ in tool-call fidelity vs the base unsloth GGUF;
  treat this as one quant choice, not the only one.
- **Reasoning ON eats the tool-call budget** — F-009 (2026-05-27) found
  this model fires zero tool calls on 40/96 PBS-Agent cells when reasoning
  is left at the Jinja default (`enable_thinking=true`). The lab serves
  it with `--chat-template-kwargs '{"enable_thinking":false}'` for tool
  parity with the dense arm. See runbook §"Qwen3 MoE — reasoning OFF for
  tool-call parity" and `conf/llama-swap.yaml`.
<!-- END HAND -->
