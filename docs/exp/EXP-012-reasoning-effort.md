---
doc_id: exp-012
title: 'EXP-012: REASONING-EFFORT-001 — does reasoning effort help or hurt agentic
  task performance? (HAL replication at lab scale, pre-registered)'
zone: lab
kind: exp
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
depends_on:
- kind: doc
  target: exp-009
- kind: doc
  target: lab-roadmap-2026-06
tags:
- lab
- exp
- agentic
- reasoning-effort
- pbs-agent-hard-v0.1
---

# EXP-012: REASONING-EFFORT-001 — reasoning effort vs agentic performance

Date created: 2026-06-11
Status: planned
Pre-registered: <commit SHA filled by `lab exp register` at registration time>

## Question

Princeton HAL (arXiv:2510.11977) reported that higher reasoning effort
*reduced* agent accuracy in 21/36 settings — a slim majority, but enough
to demand the knob be swept rather than assumed monotonic. Locally:
does gpt-oss-20b's selectable reasoning effort (low/medium/high, the
only registered local model exposing a true tri-level effort knob)
help, hurt, or do nothing on the lab's hard agentic suite — and what
does it cost in latency?

## Hypothesis

- **H1 (non-monotonic):** pass@1(medium) ≥ pass@1(high) — i.e. the top
  effort level is NOT the top scorer, replicating HAL's direction on at
  least one boundary.
- **H2 (floor):** pass@1(low) < pass@1(medium) — some reasoning helps;
  the curve is inverted-U rather than monotonically decreasing.
- **H3 (latency):** mean per-cell latency at high ≥ 2× low; report the
  pass-per-second tradeoff regardless of H1/H2.

## Method

- model: gpt-oss-20b-local (MXFP4, fits 12 GB after sweep queue frees)
- suite: pbs-agent-hard-v0.1, v2 prompt (sealed revision)
- config arms: react-4096 with `extra: {think: "low"|"medium"|"high"}`
  (the per-config `extra` plumbing built in EXP-002 for qwen3
  think:false; verify the value reaches ollama via a 1-cell smoke and
  trace inspection before the sweep — if `think` levels are not
  honored by the ollama lane, kill and re-plumb first)
- seeds: [1, 2, 3] (3 × 32 × 3 arms = 288 cells; effort comparisons are
  within-model so 3 seeds gives paired per-task comparisons; full
  ADR-004 N=8 escalation only if the H1 delta is within noise)
- queued on the gpu pueue group behind BRUTAL-BENCH-001.

## Success / failure criteria

- H1: CONFIRMED iff pass@1(medium) ≥ pass@1(high); margin reported with
  per-task paired counts. REFUTED if high > medium by > 1 task.
- H2: CONFIRMED iff pass@1(low) < pass@1(medium) by > 1 task; within
  ±1 task → INCONCLUSIVE.
- H3: CONFIRMED iff latency(high) ≥ 2× latency(low); report the
  pass-rate-per-wall-clock curve either way.

## Kill criteria

- Kill at smoke stage if the `think` level demonstrably does not reach
  the model (identical token counts/latency across arms ⇒ knob inert).
- Kill if > 10% of cells fail on harness/transport errors.

## Analysis plan

Per-arm pass@1 + per-category table + per-task paired flip matrix
(which tasks flip between effort levels), latency distribution per arm,
H1–H3 verdicts. Feeds a short public note: local replication (or not)
of HAL's reasoning-effort result on a 12 GB box.
