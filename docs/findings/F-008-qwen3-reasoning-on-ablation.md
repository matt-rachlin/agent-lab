---
doc_id: f-008-qwen3-reasoning-on-ablation
title: 'F-008: qwen3-14b-q4 reasoning-ON ablation on PBS-Agent v0.1 — pre-registered
  verdict (placeholder)'
zone: lab
kind: finding
status: draft
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
depends_on:
- kind: doc
  target: exp-002b
- kind: doc
  target: exp-002
- kind: doc
  target: f-005-12gb-agent-v0-2-tool-use
- kind: doc
  target: f-004-qwen3-reasoning-ablation
- kind: code
  target: lab:scripts/analyze_exp002b.py
- kind: artifact
  target: lab:analysis/EXP-002b/SUMMARY.md
- kind: artifact
  target: lab:analysis/EXP-002b/verdicts.md
tags:
- lab
- finding
- findings
- qwen3
- reasoning
- ablation
---

# F-008: qwen3-14b-q4 reasoning-ON ablation on PBS-Agent v0.1

(Placeholder — final numbers and verdict are filled in by
`scripts/analyze_exp002b.py` after the EXP-002b sweep completes.)

## TL;DR

(To be filled.)

## Setup

- **Experiment**: EXP-002b
  ([`docs/exp/EXP-002b-qwen3-reasoning-on-ablation.md`](../exp/EXP-002b-qwen3-reasoning-on-ablation.md),
  pre-reg SHA `6fbb2b91bf30`)
- **Sweep config**: [`conf/sweep/EXP-002b.yaml`](../../conf/sweep/EXP-002b.yaml)
- **Baseline reference**: EXP-002 / [F-005](./F-005-12gb-agent-v0.2-tool-use.md)
  qwen3-14b-q4 think:false end_state mean = 0.750.
- **Total cells**: 12 PBS-Agent v0.1 tasks × 1 model × 1 config × N=8 = **96**.
- **Wall time**: (filled in after sweep completes).
- **Pass rate**: (filled in after sweep completes).

## Pre-registered hypothesis verdict

H1: think:true qwen3-14b-q4 end_state mean is materially lower than
EXP-002's 0.750 baseline.

| think:true mean | Verdict   |
| --------------- | --------- |
| ≥ 0.55          | REFUTED   |
| 0.30 – 0.55     | MIXED     |
| < 0.30          | CONFIRMED |

**Observed think:true end_state mean: (TBD).**

**Verdict: (TBD).**

## Comparison to think:false baseline

| | think:false (EXP-002) | think:true (EXP-002b) | delta (false − true) |
| --- | --- | --- | --- |
| end_state mean | 0.750 | (TBD) | (TBD) |
| n cells | 96 | 96 | — |
| 95 % CI | (TBD) | (TBD) | — |

Per-task breakdown in `analysis/EXP-002b/verdicts.md` and
`analysis/EXP-002b/per_task_endstate.csv`.

## Relation to F-004

F-004 established reasoning-ON is net-negative on **single-turn**
PBS-v0.1 (-12.5 to -28 pp depending on category, +32 pp empty-rate
penalty). EXP-002b is the agentic regime extension. F-008 reports
whether the gap holds, widens, or collapses when the task scaffolding
moves from single-turn to multi-turn tool-use.

## Operational implication

(To be filled per verdict.)

## Limitations

- Single model (qwen3-14b-q4). No transfer claim to other reasoning
  models.
- PBS-Agent v0.1 only. No claim about reasoning-mode behaviour on
  other task suites (BFCL, τ²-bench, OSWorld, etc.).
- Single Ollama daemon revision. Reasoning-mode quality can shift
  across Ollama upgrades; the EXP-002 and EXP-002b daemons are
  intended to be the same revision but were not pinned via DVC at
  this phase (a known Phase 15.4 gap).

## Verified

(To be filled after analyze script runs.)
trust_level: unverified
