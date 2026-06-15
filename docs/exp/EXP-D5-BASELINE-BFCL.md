---
doc_id: exp-d5-baseline-bfcl
title: 'EXP-D5-BASELINE-BFCL: D5 scoreboard capability baseline — qwen3-4b-ft-toolcall on BFCL v3 (N=16)'
zone: lab
kind: exp
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
depends_on:
- kind: doc
  target: exp-013-ft-toolcall
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
- qwen3-ft
- retroactive
---

# EXP-D5-BASELINE-BFCL: D5 scoreboard capability baseline — qwen3-4b-ft-toolcall on BFCL v3

Date created: 2026-06-14
Status: results pending (sweep not yet executed)
Pre-registered: **NO — retroactive record** (config committed before this
doc was written, per same self-flagging discipline as EXP-007/008).

## Question

What is the verified BFCL v3 standing of `qwen3-4b-ft-toolcall` at N=16
seeds, and what tier thresholds does that establish for the ADR-009
scoreboard?

## Hypothesis

Reverse-engineered from `conf/sweep/d5-baseline-bfcl.yaml`: a real
multi-seed BFCL run on the FT model will (a) produce a stable point
estimate with tight 95% CIs, (b) trigger the inline validity gate
(validity_passed), and (c) serve as the verified anchor for scoreboard
tier-0 BFCL thresholds — replacing any placeholder thresholds currently
in the scoreboard config.

## Setup

- **Sweep config**: [`conf/sweep/d5-baseline-bfcl.yaml`](../../conf/sweep/d5-baseline-bfcl.yaml)
- **Experiment slug**: D5-BASELINE-BFCL-001
- **Suite**: BFCL v3 AST, 32-task balanced sample (8 tasks × 4 categories)
- **Model**: qwen3-4b-ft-toolcall-q4-latest
- **Config**: greedy (temperature=0.0, top_p=1.0, max_tokens=4096,
  scaffold=single_turn, think=false)
- **Seeds**: N=16 (exceeds ADR-004 N≥8 requirement)
- **Concurrency**: 1 (serial)

## Gating function

Results will set tier-0 BFCL thresholds on the ADR-009 scoreboard:
- Validity gate (inline): validity_passed must be true post-hoc
- Promoted to `verified` trust after holds verdict from verifier battery

## Results

**Results pending** — sweep not yet executed. No MLflow run IDs available.
