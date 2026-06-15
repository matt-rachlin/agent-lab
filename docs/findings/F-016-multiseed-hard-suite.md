---
doc_id: f-016-multiseed-hard-suite
title: 'F-016: EXP-009 — N=8 confirmation on the hard suite. Ranking direction
  holds but single-seed numbers flattered qwen3-coder by 6.6pp; temp-0 seed
  spread reaches 12.5pp (devstral); qwen3''s narration failures are
  deterministic 0/8; gemma4 pass^8 = 0.875. H1 INCONCLUSIVE, H2 REFUTED
  (qwen3), H3 REFUTED, H4-H5 CONFIRMED.'
zone: lab
kind: finding
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
depends_on:
- kind: doc
  target: exp-009
- kind: doc
  target: f-013-prompt-robustness-model-property
- kind: doc
  target: adr-004-reliability-discipline
- kind: code
  target: lab:scripts/analyze_exp009.py
- kind: artifact
  target: lab:analysis/EXP-009/SUMMARY.md
- kind: artifact
  target: lab:analysis/EXP-009/per_task_seed_matrix.csv
tags:
- lab
- finding
- findings
- reliability
- multi-seed
- pbs-agent-hard-v0.1
- confidence-high
- importance-8
---

# F-016: The N=8 hard-suite picture — single seeds lie by up to 6.6pp

## TL;DR

HARD-BENCH-003 (32 tasks × 3 models × 8 seeds × v2 prompt, 768 cells,
765 clean + 3 transport-healed):

| model | pass@1 | 95% CI | pass^8 | seed spread | single-seed said |
|---|---|---|---|---|---|
| gemma4-12b | **0.914** | [0.820, 0.988] | 0.875 | 0.062 | 0.938 (+2.4) |
| qwen3-coder-30b | 0.746 | [0.590, 0.875] | 0.719 | 0.031 | 0.812 (**+6.6**) |
| devstral-24b | 0.520 | [0.355, 0.684] | 0.469 | **0.125** | 0.531 (+1.1) |

Pre-registered verdicts:

- **H1 (ranking) INCONCLUSIVE** — the ordering gemma4 > qwen3-coder >
  devstral holds at every seed, but the gemma4-vs-qwen3 bootstrap CIs
  overlap: 32 tasks cannot CI-separate a 17pp gap when per-task
  outcomes are this correlated. Direction: robust. Magnitude: soft.
- **H2 (±5pp anchoring) REFUTED for qwen3-coder** — its single-seed
  0.812 was 6.6pp flattering, and the mechanism is CROSS-RUN, not
  within-run: three tasks it passed in HARD-BENCH-002
  (multi-http-catalog-pricing, multi-http-index-aggregate,
  shell-fragment-reassembly) went **0/8** in HARD-BENCH-003, while
  shell-access-log-slow-error-endpoints flipped 0 → 7/8. Outcomes are
  near-deterministic within a run but flip between identical-config
  runs — replication variance exceeds seed variance. (CORRECTED
  2026-06-12: an earlier wording attributed this to within-run
  flakiness; the seed matrix refutes that.) gemma4's anchor held
  (−2.4pp), but its code-lru-cache-trace pass was a 1-in-8 fluke that
  seed 1 happened to catch.
- **H3 (2–6pp spread) REFUTED** — devstral's across-seed spread is
  **12.5pp** at temperature 0 (gemma4 6.2pp, qwen3 3.1pp). The
  reliability protocol's literature-derived 2–6pp band understates
  variance for weak/flaky models; spread appears to scale with
  failure rate, concentrated in devstral's multi tasks (e.g.
  multi-config-driven-transform 4/8, multi-grep-weighted-todo 5/8).
- **H4 (reliability gap) CONFIRMED** — pass^8 < pass@1 for all; gemma4
  pass^8 = 0.875: when it passes, it almost always passes every time.
- **H5 (deterministic narration) CONFIRMED** — qwen3-coder's four
  code-fix tasks (fibonacci, interval-merge, topo-sort, expr-parser)
  are 0/8: the residual narration failure mode (F-013, F-014) is fully
  deterministic, not variance. Its code category sits at 0.500 vs
  1.000 on data.

## Consequences

- **The public writeup's hard-suite table must be updated** to the N=8
  numbers with CIs; qwen3-coder's headline drops 81→75. The
  single-seed-disclosed table was honest but materially off for one
  model — which is itself the strongest possible argument for the
  lab's multi-seed discipline.
- ADR-004's expected-variance band needs an amendment: 2–6pp holds for
  strong models; budget 10–15pp for sub-0.6 scorers.
- 9 of 96 (model, task) cells are flaky (neither 0/8 nor 8/8) —
  per-task claims from any single seed on those cells are noise.
- gemma4-12b remains the local champion with the cleanest reliability
  profile (highest pass^8, modest spread); its true weakness at N=8 is
  `code` (0.766), not shell.
trust_level: unverified
