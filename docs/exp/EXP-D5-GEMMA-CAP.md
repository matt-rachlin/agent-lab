---
doc_id: exp-d5-gemma-cap
title: 'EXP-D5-GEMMA-CAP: D5 scoreboard capability baseline — gemma4-12b on BFCL v3 (N=16)'
zone: lab
kind: exp
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
depends_on:
- kind: doc
  target: adr-004-reliability-discipline
- kind: doc
  target: adr-009-scoreboard
tags:
- lab
- exp
- bfcl
- scoreboard
- d5
- gemma4
- retroactive
---

# EXP-D5-GEMMA-CAP: D5 scoreboard capability baseline — gemma4-12b on BFCL v3

Date created: 2026-06-14
Status: results pending (sweep not yet executed)
Pre-registered: **NO — retroactive record** (config committed before this
doc was written, per same self-flagging discipline as EXP-007/008).

## Question

What is the verified BFCL v3 capability standing of `gemma4-12b` at N=16
seeds, to pair with its safety baseline (EXP-D5-SAFETY-CONSTRAINT) for a
complete ADR-009 scoreboard tier?

## Hypothesis

Reverse-engineered from `conf/sweep/d5-gemma-cap.yaml`: gemma4-12b will
produce a stable BFCL capability estimate at N=16 that, combined with its
proven 0-violation safety record (EXP-016 / EXP-D5-SAFETY-CONSTRAINT),
completes the evidence required for a full scoreboard tier entry for this
model.

## Setup

- **Sweep config**: [`conf/sweep/d5-gemma-cap.yaml`](../../conf/sweep/d5-gemma-cap.yaml)
- **Experiment slug**: D5-BASELINE-BFCL-GEMMA4-001
- **Suite**: BFCL v3 AST, 32-task balanced sample (8 tasks × 4 categories;
  same slugs as D5-BASELINE-BFCL for cross-model comparability)
- **Model**: gemma4-12b
- **Config**: greedy (temperature=0.0, top_p=1.0, max_tokens=4096,
  scaffold=single_turn, think=false)
- **Seeds**: N=16 (exceeds ADR-004 N≥8 requirement)
- **Concurrency**: 1 (serial)

## Gating function

Results complete the gemma4-12b scoreboard tier when combined with its
safety baseline. Both axes (capability + safety) must be verified before
the model can be promoted on the ADR-009 scoreboard.

## Results

**Results pending** — sweep not yet executed. No MLflow run IDs available.
