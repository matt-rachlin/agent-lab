---
doc_id: adr-015-sglang-serving
title: 'ADR-015: SGLang serving for the small-dense throughput tier'
zone: lab
kind: adr
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, adr, inference, serving, sglang, quantization, throughput]
---
# ADR-015: SGLang serving for the small-dense throughput tier

Status: proposed
Date: 2026-06-14
Deciders: Matt Rachlin

## Context

Sweeps are **orchestration-bound, not GPU-bound** on small models. Evidence (RTX
3080 Ti, measured 2026-06-14):

- Sweep configs run `max_concurrency: 1` — e.g. CAND-BFCL-QWEN3-14B-001 is 32
  tasks x 16 seeds = 512 single-turn requests, fired one at a time.
- GPU mean util 32.6% over 24h; util `<30%` for 58% of the last 3 days.
- Even *while actively inferring* (`>150W`), util averages 64% and is `<50%` for
  41% of busy-time. Nearly half of working GPU-seconds are wasted to serialization.

Current local serving (ADR-002, inference-routing): Ollama (11434) + llama-swap
(8080)/llama.cpp for GGUF, behind LiteLLM (4000). Ollama serves near-serially and
continuous batching is not exploited.

Two distinct regimes:

- **Small dense (4B-9B)** leave VRAM headroom after weights for a large KV-cache
  pool — the ideal case for continuous batching. This is where serialization hurts.
- **Big models (20B-70B)** run via llama.cpp hybrid CPU offload (llama-3.3-70b
  ~1.8 tok/s) and are memory-bandwidth bound — batching cannot help them.

SGLang's RadixAttention reuses shared prefixes across concurrent requests; our
trajectories share a system prompt + tool schemas, a structural match. Constraint:
12 GB VRAM (~10.5 usable; desktop shares the GPU, no usable iGPU). SGLang serves HF
AWQ/GPTQ/FP8 safetensors, **not** GGUF — so adopting it changes engine *and*
quantization together.

## Decision

Adopt **SGLang as a serving backend for the 4B-9B dense throughput tier**, exposed
as **new model arms** (`*-awq`), never as in-place replacements for existing
GGUF/Ollama arms.

1. **New backend `sglang-local`** in `lab.models`, launched as a **podman container
   managed by llama-swap** in an exclusive VRAM group. Podman **graphroot on
   `/data`** (the 93%-full root fs must not hold the image/weights). Reuses the
   existing Valkey GPU lease and `model_pool` teardown unchanged.
2. **AWQ quantized in-house from official FP16**, with quant recipe + calibration
   dataset + weight sha256 recorded in a `MANIFEST.json`. Quant provenance is a
   controlled variable, not an inherited unknown.
3. **New arms registered distinctly** (e.g. `qwen3-4b-awq` separate from
   `qwen3-4b`). Engine+quant change is not a drop-in; this preserves scoreboard
   comparability and adds an AWQ-vs-GGUF point to the quantization research axis.
4. **Routing unchanged**: LiteLLM -> llama-swap:8080 -> SGLang. Sweep runner
   untouched; `*-awq` sweeps raise `max_concurrency` to 16-32.
5. **Latency discipline**: latency/cost reported only from a concurrency-1
   reference pass; batched runs feed quality/throughput metrics only.
6. **Scope guard**: SGLang only for models whose AWQ weights + a meaningful KV pool
   fit 12 GB (<= ~9B). 20B+ stay on llama.cpp.

Rollout is phased. **Phase 0 (spike):** qwen3-4b, in-house AWQ, load-test
concurrency 1 vs 16/32 vs the Ollama baseline; **kill criterion: must clear ~3x
batched throughput** or we stop. **Phase 1:** integrate the five seams (llama-swap,
LiteLLM, `lab.models`, manifest, sweep configs). **Phase 2:** re-baseline the BFCL
cohort on the `-awq` arms; report wall-clock + util delta. **Phase 3:** roll out to
the small-model matrix + SOP.

## Consequences

- Easier: 5-10x faster small-model sweeps at full GPU util; AWQ-vs-GGUF becomes
  measurable; better ROI on existing hardware before any GPU purchase is considered.
- Harder: a second serving engine to maintain (container + CUDA/torch/flashinfer);
  an in-house quantization pipeline to build and document; more VRAM-fit tuning.
- Risks: 12 GB headroom is thin for 14B AWQ (limited batch win — out of initial
  scope); SGLang batching is non-deterministic across batch sizes (mitigated by
  N>=8 + bootstrap CIs and the concurrency-1 latency pass); desktop shares VRAM
  (mitigate later via headless/iGPU); image+weights must stay off the root fs
  (graphroot on /data).

## Considered alternatives

- **vLLM** — industry-default batching, but does not exploit our shared-prefix
  structure as well as SGLang's RadixAttention, and has a weaker GGUF/offload
  story. Kept as the fallback engine.
- **Ollama `OLLAMA_NUM_PARALLEL` only** — cheapest; captures some of the win but a
  lower throughput ceiling, no RadixAttention, and no quant-axis data point.
- **llama.cpp `--parallel` continuous batching** — keeps GGUF parity (direct
  comparability) and one engine; lower ceiling, no AWQ point. Remains the choice
  for the >12 GB models.
- **Drop-in replace GGUF arms with AWQ** — rejected: silently changes the SUT and
  corrupts scoreboard comparability.

Relates to ADR-002 (inference-routing): extends it with a third local backend;
does not supersede it.
