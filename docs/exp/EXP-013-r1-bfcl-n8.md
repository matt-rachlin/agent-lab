---
doc_id: exp-013-r1-bfcl-n8
title: 'EXP-013-R1: BFCL N=8 confirmation pass for the FT-TOOLCALL-001 H1 claim'
zone: lab
kind: exp
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags:
- lab
- exp
- bfcl
- fine-tuning
- adr-004
- exp-013-followup
depends_on:
- kind: doc
  target: exp-013-ft-toolcall
- kind: doc
  target: adr-004-reliability-discipline
- kind: doc
  target: f-019-bfcl-perturbed-twin-contamination
---

# EXP-013-R1: BFCL N=8 confirmation

Date created: 2026-06-14
Pre-registered: this commit
Status: planned (config queued at `conf/sweep/ft-eval-bfcl-n8.yaml`)

## Background

EXP-013's BFCL arm (FT-EVAL-BFCL-001) ran at **N=1** seed per the original pre-reg (which permitted "single deterministic pass for BFCL only"). The wave-2 contamination + research-rigor audit (2026-06-14) found that the +19pp BFCL headline cannot ride on a single greedy seed before public citation — ADR-004 requires N≥8 with bootstrap CIs for any result that gates downstream decisions, and the public writeup `docs/writeups/eval-train-eval-loop.md` is exactly such a decision.

This experiment is the N=8 confirmation pass.

## Hypothesis

- **H1**: At N=8, ft − base ≥ +5pp on `bfcl_ast_match` overall AND the bootstrap 95% CI of (ft − base) does NOT cross zero.

Re-confirmation of H1 from EXP-013 with proper statistical discipline.

## Method

Sweep config: `conf/sweep/ft-eval-bfcl-n8.yaml`. Run via `lab sweep run ft-eval-bfcl-n8` once the gpu_lease is free.

- Models: `qwen3-4b` (base) and `qwen3-4b-ft-toolcall-q4-latest` (ft)
- Suite: bfcl-v3-ast (1000 tasks)
- Scaffold: single-turn, greedy-1024 (temperature 0)
- Seeds: [1..8]
- N = 1000 tasks × 8 seeds × 2 arms = 16,000 cells
- Expected wall-clock: ft arm ~33 min × 8 seeds = ~4.4 hours; base arm similar; total ~9 hours at max_concurrency=1.

## Why N=8 matters when temperature=0

Greedy decoding at temperature 0 should be deterministic. In practice, the lab's serving stack (ollama + llama-swap + LiteLLM proxy + continuous batching) introduces per-seed variation through:

- KV cache pressure under concurrent batches
- Float-point order in batched matmul
- llama.cpp `--parallel` non-determinism
- Optional `seed` parameter that propagates through the request → forward

These are real sources of variance that an N=1 pass hides. ADR-004's N≥8 + bootstrap CI is the discipline that captures them.

## Pass conditions

- **H1 CONFIRMED** if Δ_overall ≥ +5pp AND the 1000-bootstrap-resample 95% CI excludes 0.
- **H1 SOFT** if Δ_overall ≥ +5pp but the CI crosses 0.
- **H1 REFUTED** if Δ_overall < +5pp.

## Followups gated on this result

- **F-019 update**: the perturbed-twin contamination probe (n=20 smoke) should be re-run at n=50-100 stratified by BFCL category for any publication-grade citation of EXP-013's H1.
- **Writeup**: only update the eval-train-eval-loop writeup with N-discipline-confirmed numbers if H1 confirms here.
- **Scoreboard**: tier-0-measured may pick up the FT model on the BFCL axis at this confirmation pass; current scoreboard query filters by `trust_level = 'verified'` which requires this to land.

## Out of scope

- Brutal and hard arms. Those ran at N=3 — wave-2 noted N=3 is below ADR-004 but the brutal H2 survives audit as the most defensible single result. Separate follow-ups (EXP-013-R2 brutal-N=8, EXP-013-R3 hard-N=8) are queued.
- Any code, model, or training-data change. This is a pure re-run with `seeds: [1..8]`.
