---
doc_id: exp-013
title: 'EXP-013: FT-TOOLCALL-001 — close the eval→train→eval loop: QLoRA fine-tune
  Qwen3-4B for agentic tool-calling on the lab''s own verified trajectories
  (pre-registered)'
zone: lab
kind: exp
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
depends_on:
- kind: doc
  target: lab-roadmap-2026-06
- kind: doc
  target: f-012-agentic-tool-calling-failure-modes
- kind: doc
  target: f-014-cloud-anchor-hard-suite
tags:
- lab
- exp
- fine-tuning
- qlora
- tool-use
- eval-train-eval
---

# EXP-013: FT-TOOLCALL-001 — the eval→train→eval loop

Date created: 2026-06-11
Status: planned
Pre-registered: <commit SHA filled by `lab exp register` at registration time>

## Question

F-012 established that agentic tool-calling fidelity — not size or
coding skill — gates local agent performance, and attributed the gap to
training. Can a 12 GB lab *change* that training? Concretely: does QLoRA
SFT of Qwen3-4B on a 60/40 mix of tool-calling data — including 659 of
the lab's own verified-successful agent trajectories (STaR/RFT-style,
with 62 frontier-teacher episodes from HARD-BENCH-CLOUD-001) — measurably
improve agentic performance on held-out suites?

## Hypothesis

- **H1 (held-out format generalization):** fine-tuned ≥ base + 5pp on
  BFCL v3 AST overall (1000 tasks, never in training data).
- **H2 (held-out task generalization):** fine-tuned > base on
  pbs-agent-brutal-v0.1 pass@1 by ≥ 2 tasks-equivalent (≥ 8.3pp at
  n=24). The brutal suite postdates the dataset build — zero episodes
  of it exist in training data.
- **H3 (seen-task gain):** fine-tuned ≥ base + 10pp on
  pbs-agent-hard-v0.1. DISCLOSED CONTAMINATION: training data contains
  successful hard-suite trajectories (its own + cloud teachers), so H3
  measures memorization-inclusive gain and is reported separately from
  H1/H2 — never headline.
- **H4 (no protocol regression):** fine-tuned shows zero F-012 failure
  modes (narration / text-emitted calls) on the audited suites —
  fine-tuning must not break what works (trajectory_audit.py is the
  check).

## Method

### Training (prepared by the ft pipeline, /data/lab/ft/)

- base: unsloth/Qwen3-4B, QLoRA 4-bit, r=16 α=32, lr 2e-4, ≤2 epochs,
  batch 2 × grad-accum 8, bf16, responses-only masking via the Qwen3
  chat template, seed 1, MLflow experiment FT-TOOLCALL-001.
- data: train_mix.jsonl — 20,000 samples, 60% tool (659 lab
  trajectories + 5,671 ToolACE + 5,670 Hermes-FC) / 40% general
  (ultrachat). Lab trajectories: end_state==1.0 only, 14/1,295 rejected
  by faithfulness cross-checks (alignment, truncation, recovered-call
  filters); rendered against the real MCP tool schemas.
- xLAM-60k excluded (gate not yet accepted) — noted; a future revision
  may add it.

### Evaluation (both arms identical; q4_k_m GGUF parity)

| eval | status vs training data | seeds |
|---|---|---|
| BFCL v3 AST (vendored, 1000) | clean | 1 (deterministic single-turn) |
| pbs-agent-brutal-v0.1 (24) | clean | 3 |
| pbs-agent-hard-v0.1 (32) | CONTAMINATED (disclosed) | 3 |

Arms: `qwen3-4b-base-q4` vs `qwen3-4b-ft-toolcall-q4`, both served via
ollama from q4_k_m GGUFs, same litellm lane config, v2 prompt, react
scaffold, temp 0. Baseline arm runs are part of this experiment (the
base 4B has never been benchmarked in the lab).

## Success / failure criteria

- H1: CONFIRMED iff ft_overall − base_overall ≥ 0.05 on BFCL AST;
  REFUTED if ≤ 0; between → INCONCLUSIVE.
- H2: CONFIRMED iff ft − base ≥ +2/24 tasks mean pass@1 on brutal;
  REFUTED if ft < base; between → INCONCLUSIVE.
- H3: CONFIRMED iff ft − base ≥ 0.10 on hard suite (reported with the
  contamination label regardless).
- H4: CONFIRMED iff trajectory_audit narration+text_emitted == 0 for
  the ft arm across both agent suites; any episode REFUTES.

## Kill criteria

- Kill training on OOM unrecoverable at --max-seq-length 4096 fallback,
  or loss divergence (train loss not decreasing over first 200 steps).
- Kill eval if GGUF export produces template-broken serving (tool calls
  unparseable at inference — the documented Ollama Modelfile/template
  risk); fix the template, re-export, restart eval (training stands).
- Train loss < 0.2 sustained ⇒ overfitting per unsloth guidance — stop
  at the checkpoint before it.

## Analysis plan

Per-eval before/after table with deltas + CIs where seeded; H1–H4
verdicts regardless of direction; trajectory_audit report on the ft
arm; if H1 or H2 confirm, a public writeup follows ("closing the
eval→train→eval loop on a 12 GB GPU") — the lab's flagship artifact.
