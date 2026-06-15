---
doc_id: exp-014
title: 'EXP-014: ARCH-BENCH-001 — SSM-hybrid vs MoE-transformer as local agents:
  granite4-tiny-h vs gpt-oss-20b on the hard suite (pre-registered)'
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
  target: exp-008
tags:
- lab
- exp
- agentic
- architecture
- ssm
- moe
- pbs-agent-hard-v0.1
---

# EXP-014: ARCH-BENCH-001 — SSM-hybrid vs MoE-transformer as local agents

Date created: 2026-06-11
Status: planned
Pre-registered: c6d6e1f  (registered by `lab exp register` at file-creation time; backfilled 2026-06-14)

## Question

The 2025–26 efficiency architectures are barely evaluated *agentically*
(almost all public numbers are single-turn). On the lab's hard agentic
suite, how do two small-activated-parameter architectures compare —
IBM granite4:tiny-h (Mamba-2/transformer hybrid, ~7B total / ~1B
active) vs OpenAI gpt-oss-20b (MoE transformer, ~21B total / 3.6B
active, MXFP4) — against each other and against the dense incumbent
gemma4-12b (0.938, HARD-BENCH-002)?

## Hypothesis

- **H1 (MoE > SSM-hybrid):** gpt-oss-20b beats granite4-tiny-h on
  pass@1 by ≥ 3 tasks-equivalent (≥ 9.4pp) — more active params +
  agentic-era training should dominate.
- **H2 (dense ceiling):** neither efficiency architecture reaches
  gemma4-12b's band — both < 0.90 pass@1.
- **H3 (failure-mode shape):** granite4-tiny-h's failures include ≥ 1
  F-012-class protocol failure mode (narration or text-emitted calls,
  via trajectory_audit), reflecting weaker agentic post-training in
  hybrid-architecture small models.

## Method

- suite: pbs-agent-hard-v0.1 (sealed), tool_use_system_v2, react-4096,
  temp 0
- models: granite4-tiny-h (new lane), gpt-oss-20b-local (default
  reasoning effort = medium; the effort axis is EXP-012's question,
  not this one)
- comparator: gemma4-12b numbers quoted from HARD-BENCH-002/003 — not
  re-run here
- seeds: [1, 2, 3] — 192 cells, est. 2–4 h; queued at the gpu ladder
  tail
- trajectory_audit runs on completion (auto-queued, default group)

## Success / failure criteria

- H1: CONFIRMED iff gpt-oss − granite ≥ 0.094 mean pass@1; REFUTED if
  granite ≥ gpt-oss; between → INCONCLUSIVE.
- H2: CONFIRMED iff both < 0.90; either ≥ 0.938 − 0.031 (within 1 task
  of gemma4) REFUTES for that model.
- H3: CONFIRMED iff audit flags ≥ 1 narration or text_emitted episode
  for granite4-tiny-h; zero across all 96 granite cells REFUTES.

## Kill criteria

- Kill if either model's lane fails transport/template on > 10% of
  cells (e.g. granite chat template issues through ollama_chat) — fix
  the lane, restart the experiment.
- Kill if a suite defect is discovered (none expected; suite is sealed
  and 5 experiments deep).

## Analysis plan

Per-model overall + per-category table with the gemma4 comparator row,
per-task seed matrix, H1–H3 verdicts, audit-report cross-reference.
Public note candidate: "are efficiency architectures ready to be
agents?" — timely, near-zero public data exists.
