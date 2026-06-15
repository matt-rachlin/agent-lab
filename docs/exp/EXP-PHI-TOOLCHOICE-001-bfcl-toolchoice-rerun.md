---
doc_id: exp-phi-toolchoice-001-bfcl-toolchoice-rerun
title: 'EXP-PHI-TOOLCHOICE-001: BFCL phi tool_choice A/B rerun'
zone: lab
kind: exp
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
depends_on:
- kind: doc
  target: f-017-bfcl-tool-choice-artefact
- kind: doc
  target: adr-008-trust-lifecycle
tags:
- lab
- exp
- bfcl
- phi
- tool-choice
- retroactive
---

# EXP-PHI-TOOLCHOICE-001: BFCL phi tool_choice A/B rerun

Date created: 2026-06-14 (run executed 2026-06-13)
Status: complete
Pre-registered: **NO — retroactive record.** This A/B was run as a
diagnostic/fix-validation step during the BFCL harness audit before a
pre-registration was written. Documented after the fact per the same
self-flagging discipline applied to EXP-007 and EXP-008. The sweep config
(`conf/sweep/phi-toolchoice-rerun.yaml`, committed in 5baf0cf) is the
closest thing to a pre-registration this experiment has.

## Question

Does `tool_choice=required` (vs the historical default `tool_choice=auto`)
account for phi-4-reasoning-plus scoring ~1% on BFCL v3 AST — and if so,
what is the model's true function-calling accuracy?

## Hypothesis

H1: `tool_choice=required` raises phi-4-reasoning emission from ~2% toward
parity with other llama.cpp models. `acc_given_emit` (pass rate among
trials that emitted a call) is the true capability signal and will be
materially higher than the ~1% headline figure.

## Setup

- **Sweep config**: [`conf/sweep/phi-toolchoice-rerun.yaml`](../../conf/sweep/phi-toolchoice-rerun.yaml)
- **Suite**: BFCL v3 AST, balanced 60-task sample (15 tasks per category:
  simple, multiple, parallel, parallel_multiple)
- **Model**: phi-4-reasoning-14b (via llama.cpp backend)
- **Arms**:
  - `auto-bug`: `tool_choice=auto` (historical default, expected to reproduce ~1%)
  - `required-fix`: `tool_choice=required` (fix under test)
- **Seeds**: N=1 (diagnostic A/B; not a multi-seed reliability study)
- **Same tasks across arms**: `tool_choice` is the only varying factor.

## Results

| arm | emission | pass | accuracy-given-emission | avg output tokens |
|---|---|---|---|---|
| `auto` (old default) | 1.7% | 0.0% | 0.0% | 332 (prose) |
| `required` (fix) | 100.0% | 45.0% | 45.0% | 37 (direct call) |

H1 confirmed: `required` raises emission from 1.7% → 100% and reveals
~45% true function-calling accuracy. See F-017 for full analysis and
implications.

## Limitations

- N=1 seed: establishes the existence and direction of the effect, not a
  stable estimate of absolute accuracy. Full 1000-task phi number under
  `required` is an open item (F-017 open questions).
- 60-task balanced sample, not the full BFCL v3 suite.

## Finding

[F-017: BFCL tool_choice artefact](../findings/F-017-bfcl-toolchoice-artefact.md)
