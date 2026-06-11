---
doc_id: exp-009
title: 'EXP-009: HARD-BENCH-003 — N=8 seed confirmation of the hard-suite ranking
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
  target: exp-008
- kind: doc
  target: adr-004-reliability-discipline
- kind: doc
  target: reliability-sweep
tags:
- lab
- exp
- agentic
- tool-use
- reliability
- pbs-agent-hard-v0.1
---

# EXP-009: HARD-BENCH-003 — N=8 seed confirmation of the hard-suite ranking

Date created: 2026-06-11
Status: planned
Pre-registered: <commit SHA filled by `lab exp register` at registration time>
Protocol: [reliability-sweep](../protocols/reliability-sweep.md) (ADR-004)

## Question

EXP-008 ranked gemma4-12b ≫ qwen3-coder-30b ≫ devstral-24b on
pbs-agent-hard-v0.1 from a single seed — below the lab's reporting bar,
and with observed run-to-run variance (gemma4 failed different tasks in
two runs at the same aggregate). At N=8 seeds under the v2 prompt: does
the ranking hold with non-overlapping CIs, where do the true pass rates
land, and which tasks are flaky vs deterministically failed?

## Hypothesis

- **H1 (ranking):** gemma4-12b > qwen3-coder-30b > devstral-24b in
  pass@1, with the gemma4-vs-qwen3 bootstrap 95% CIs non-overlapping.
- **H2 (anchoring):** each model's pass@1 lands within ±5pp of its
  single-seed HARD-BENCH-002 number (gemma4 0.938, qwen3 0.812,
  devstral 0.531).
- **H3 (variance):** per-model across-seed spread (max−min seed pass
  rate) is 2–6pp despite temperature 0, consistent with the protocol's
  literature claim and the EXP-008 anecdote.
- **H4 (reliability gap):** pass^8 < pass@1 for every model, and gemma4
  pass^8 ≥ 0.75 — i.e. most of its passes are stable, not luck.
- **H5 (failure structure):** qwen3-coder's `code`-category misses
  (fibonacci-bug-fix, interval-merge-fix, topo-sort, expr-parser-fix)
  are deterministic (0/8), not flaky — its gap is capability, not
  variance.

## Method

### Models

| litellm_id | role |
|---|---|
| gemma4-12b | subject — incumbent local coding agent |
| qwen3-coder-30b | subject |
| devstral-24b | subject |

### Matrix

- suite: pbs-agent-hard-v0.1 (32 tasks, sealed; no task edits since c4e56a7)
- prompt: tool_use_system_v2 for all arms
- config: react, temperature 0.0, top_p 1.0, max_tokens 4096
- seeds: [1, 2, 3, 4, 5, 6, 7, 8] (protocol schedule)
- 256 cells/model, 768 total; single sweep slug HARD-BENCH-003
- est. wall clock ~14 h serialized on the gpu pueue group
  (HARD-BENCH-002's 96 cells took 1 h 45 m)

### Metrics

Per protocol: pass@1 (mean), pass^k for k ∈ {1, 4, 8}, bootstrap 95% CI
per model; per-task seed-pass counts to classify flaky (1–7/8) vs
deterministic-fail (0/8) vs solid (8/8).

## Success / failure criteria

Each hypothesis gets an independent verdict (CONFIRMED / REFUTED /
INCONCLUSIVE), reported regardless of direction:

- H1: CONFIRMED iff both orderings hold in pass@1 AND the gemma4-vs-
  qwen3 bootstrap 95% CIs do not overlap; ordering-holds-but-CIs-overlap
  → INCONCLUSIVE.
- H2: CONFIRMED iff |pass@1 − HARD-BENCH-002 value| ≤ 0.05 for all
  three models; any model outside the band refutes for that model.
- H3: CONFIRMED iff every model's max−min seed pass-rate spread lies in
  [0.02, 0.06]; spread of exactly 0 for any model refutes (would imply
  full determinism, contradicting EXP-008's observation).
- H4: CONFIRMED iff pass^8 < pass@1 for all models AND gemma4 pass^8 ≥ 0.75.
- H5: CONFIRMED iff all four named qwen3-coder `code` tasks score 0/8;
  any seed-pass on any of them refutes.

## Kill criteria

- Kill and postmortem if > 5% of cells (≥ 39/768) terminate on harness
  error (sandbox, tool-server, timeout-by-infrastructure) rather than
  model behavior.
- Kill if any task is discovered unsolvable as specified (fixture or
  predicate defect) — fix and restart the full sweep rather than
  patching cells mid-run; EXP-008's mid-run cell re-run is not a
  precedent for a pre-registered experiment.
- Abort-and-rescope if wall clock exceeds 3× the ~14 h estimate
  (queue contention or model-swap thrash).

## Analysis plan

One report: per-model pass@1/pass^k/CI table, per-category table,
per-task seed-pass matrix, verdicts for H1–H5 regardless of direction.
Writeup update only after this lands (the public writeup currently
discloses single-seed; HARD-BENCH-003 either upgrades it with CIs or
corrects it).
