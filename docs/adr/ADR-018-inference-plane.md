---
doc_id: adr-018-inference-plane
title: 'ADR-018: Inference plane — LiteLLM + llama-swap + Ollama + SGLang'
zone: lab
kind: adr
status: draft
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, adr, inference, serving, litellm, llama-swap, ollama, sglang]
supersedes: adr-002-inference-routing
---
# ADR-018: Inference plane — LiteLLM + llama-swap + Ollama + SGLang

Status: proposed
Date: 2026-06-14
Deciders: Matt Rachlin

## Context

ADR-002 (accepted 2026-05-25) documented a two-engine stack: LiteLLM proxy (port
4000) over a single local Ollama daemon (11434). Since then the stack has grown
to four engines across three size regimes, and ADR-015 adds a fourth (SGLang,
Phase 1). ADR-002's routing table is no longer accurate. Wave-1 architecture
review (2026-06-14) identified this as a gap: no single ADR names all four
engines, their roles, or which config file is canonical for each seam.

This ADR supersedes ADR-002. ADR-002 is marked `superseded_by: ADR-018`.

## Decision

### Four-engine routing matrix

| Engine | Port | Protocol | Size regime | Quantization |
|--------|------|----------|-------------|--------------|
| LiteLLM proxy | 4000 | OpenAI-compatible | all (router) | n/a |
| Ollama | 11434 | Ollama native + OpenAI-compat | ≤14B fast-load, cloud proxy | GGUF Q4/Q5 |
| llama-swap | 8080 | OpenAI-compatible | 14B–70B VRAM-managed | GGUF Q4/Q5 |
| SGLang | (container, managed by llama-swap) | OpenAI-compatible | 4B–9B dense throughput | AWQ |

All client code calls LiteLLM (port 4000) only. Engine selection is pure routing
config — no client code is aware of which engine serves a request.

### Routing rules (as of Phase 19b + ADR-015 Phase 1)

- `*-cloud` models: LiteLLM → Ollama (11434), which proxies to Ollama Cloud.
- Small local models with fast cold-load (qwen3-4b, llama3.1-8b, nemotron-3-nano,
  nemotron-nano-9b-v2 Ollama variants): LiteLLM → Ollama (11434).
- Medium/large VRAM-managed (qwen3-30b-a3b-moe, gpt-oss-20b-local,
  phi-4-reasoning-14b, llama-3.3-70b-q4-local, nemotron-nano-9b-v2 llama-server):
  LiteLLM → llama-swap (8080).
- Dense throughput tier (qwen3-4b-awq and future `*-awq` arms):
  LiteLLM → llama-swap (8080) → SGLang container (VRAM exclusive group).

### Canonical config artifacts

- `conf/serving/litellm-config.yaml` — all model_list entries and litellm_settings.
- `conf/serving/llama-swap.yaml` — model groups, VRAM exclusive groups, TTL eviction.

Both files are versioned in the repo. Backup copies (`*.bak-YYYYMMDD-*`) in
`conf/` are scratch; the un-suffixed files are always authoritative.

Future work: move serving configs under `conf/serving/` to separate them from
Hydra sweep configs (not yet done — the move requires a llama-swap and LiteLLM
container restart coordinated with the Phase 1 integration).

### Retry / fallback policy

Defined in `conf/serving/litellm-config.yaml` under `litellm_settings.fallbacks` and
`router_settings`. Key invariant: never fall back across size regimes (a
qwen3-4b call must not silently fall back to a 120B cloud model and produce
a misleadingly high score). Fallbacks are defined only within the cloud tier.

### SGLang integration seams

Per ADR-015 Phase 1: llama-swap manages the SGLang container as an exclusive
VRAM group. LiteLLM routes `*-awq` model names to llama-swap:8080, which
dispatches to the running SGLang container. `lab.core.model_pool` manages the
GPU lease via Valkey, unchanged.

## Consequences

- Easier: full routing picture is in one ADR; config ownership is clear; ADR-015
  integration has an explicit home.
- Harder: four engines means four failure modes; llama-swap cold-load 502 race
  (documented in `conf/serving/litellm-config.yaml` comments, Phase 19e #74) is a known
  fragile seam.
- Risks: SGLang container shares the same 12 GB VRAM as other engines; scheduling
  conflicts must be managed by llama-swap exclusive groups. Desktop background
  VRAM steal is not mitigated by software alone.

## Considered alternatives

- **Amend ADR-002** — rejected; ADR-002's scope (two-engine, single Ollama) is
  too narrow to accommodate a fourth engine without rewriting it entirely.
- **Single engine (vLLM)** — evaluated, rejected per ADR-015: weaker GGUF/offload
  story and no RadixAttention for our shared-prefix trajectories.
