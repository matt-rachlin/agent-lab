---
doc_id: steward-architecture
title: 'The steward: a deployed agent whose permissions are derived from this
  lab''s benchmark evidence (architecture note)'
zone: lab
kind: guide
status: active
owner: m
created: '2026-06-12'
last_updated: '2026-06-12'
last_verified: '2026-06-12'
tags:
- lab
- steward
- deployment
- trust-tiers
- safety
---

# The steward — eval-gated agent deployment

The lab's question is "can you trust a local agent, and how would you
know?" The steward is the answer applied to itself: a persistent agent
runtime (separate, private deployment) doing real daily work on the
homelab, whose **permissions are mechanically derived from this lab's
published measurements**. This note documents the architecture; the
deployment's job content is private.

## Trust tiers

| tier | powers | gate — evidence required (machine-checked at dispatch) |
|---|---|---|
| 0 | read-only: summarize collector output into reports | champion status on capability suites |
| 1 | writes inside per-job scoped directories | constraint-compliance scanner: zero readonly/scope violations for the deployed model (EXP-016 / CONSTRAINT-GATE-001) |
| 2 | act on untrusted content (live web pages) | injection-compliance scanner: compliance rate ≤ 0.10 for the deployed model (EXP-015 / INJECT-BENCH-001) |
| 3 | external drafts via human approval queue | tier 2 + audited shadow history, manual approval flag |
| 4 | autonomous external actions | not grantable |

Gates are evaluated **per run** by reading the lab's scanner CSVs —
when a sweep lands, the corresponding tier opens (or stays shut)
automatically. Missing evidence = locked: *no evidence is not
clearance.*

## Design rules (each traceable to a lab finding)

- **Tier-0 jobs are model-as-writer, not model-as-actor**: deterministic
  collector scripts gather data; the model gets a bundle and zero
  tools. (F-012: protocol failures live in tool loops — so don't give
  reporting jobs a tool loop.)
- **Structural asking**: the runtime provides a question/answer channel
  and prompts for it, because models measured 0/9 at asking
  spontaneously on ambiguous tasks (ask-vs-assume domain).
- **Memory is runner-applied**: tier-0 models emit fenced memory-update
  blocks; the runtime validates and applies them (cross-episode memory
  domain proved the pattern; ablation: 5/5 chains → 0/5 without it).
- **Every run is audited**: lab-style trajectory JSONL + the
  trajectory-audit classifiers; summarize-mode runs assert zero tool
  calls; anomalies become incidents surfaced in the next digest.
- **The flywheel**: steward incidents become eval-suite tasks; model
  upgrades re-run the gates before the deployment switches (the
  nightly canary already enforces stack-level regression checking).
- **GPU discipline**: steward runs enqueue into the same serialized
  queue as lab sweeps — measurement and production share hardware
  without contention.

## Status

Deployed 2026-06-12 with three tier-0 jobs (lab digest, host health,
job-market scan). Its first health report identified two genuinely
failed services, including the lab's own canary unit. Tier-1 evidence
sweep pre-registered and running (EXP-016); tier-2 evidence queued
(EXP-015). Phases C (memory), D (approval outbox + ask-channel), and
E (gated browser jobs) in build.
