---
doc_id: pbs-agent-brutal-v0-1-card
title: pbs-agent-brutal-v0.1 — task suite
zone: lab
kind: card
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
tags:
- lab
- card
- tasks
- pbs-agent-brutal-v0.1
---

## Purpose

<!-- BEGIN HAND -->
The tier above pbs-agent-hard-v0.1, built because gemma4-12b reached
0.938 there (2 tasks of headroom). Targets the two behaviors the hard
suite doesn't test (its CARD's known-limitations list): **reading
failing output and recovering from a wrong intermediate step.** Four
categories × 6 tasks, all difficulty `hard`:

- `debug` — test-driven bug-hunt loops: multi-file Python projects with
  3 planted logic bugs each; the model must run `run_tests.py` via
  shell_exec, read real failures, fix sources (editing tests/main is
  forbidden by the spec), re-run to green, then compute a data-derived
  final answer. Per-bug isolation verified: fixing all-but-one bug
  still yields a wrong answer.
- `recovery` — the happy path is broken: moved files, truncated JSON
  with a salvage+merge rule, mixed-validity rows, decoy files, a buggy
  helper script, unit-normalization traps. The naive answer is verified
  to FAIL the predicates for every task.
- `longhaul` — 8–13-call sequential chains: HTTP fixture pagination with
  data-dependent routing, staged ETL with predicate-checked intermediate
  artifacts, stage-specific rounding rules. Wrong-branch and
  per-item-rounding answers verified to fail.
- `spec` — adversarial precision: 8–14-rule specs over data crafted so
  every edge case matters; 34 distinct plausible misreadings across the
  category are verified to produce failing answers.

Every task was machine-verified at authoring time by a generator/checker
script (correct answer matches predicates; trap answers don't; predicate
substrings absent from all workspace files). All predicates are
composite `all_of` with prefixed `key=value` substrings. All tasks use
`system_prompt_id: tool_use_system_v2`. HTTP fixtures use reserved
domains only (example.com/org/net) under `_http_fixtures/` workspace
paths with sandbox network allowlists.
<!-- END HAND -->

<!-- BEGIN AUTOGEN -->

## Categories

- `debug` — 6 task(s)
- `longhaul` — 6 task(s)
- `recovery` — 6 task(s)
- `spec` — 6 task(s)

## Difficulty distribution

- hard: 24

## Tools used (union across tasks)

- `fs_grep`
- `fs_read`
- `fs_write`
- `http_fetch`
- `python_eval`
- `shell_exec`

## Budgets

- max_turns: 10–20 (debug 16–18, longhaul 16–20, recovery 12–14, spec 10–12)
- tool_budget: 14–24

## Pre-reg shape

- success_predicate types:
  - `all_of` — 24 task(s)

## Experiments using this suite

- `EXP-010` (BRUTAL-BENCH-001)

## Findings citing this suite

- (none yet)

<!-- END AUTOGEN -->

## Known limitations

<!-- BEGIN HAND -->
- Not yet validated against live models (EXP-010 seed-1 pass is the
  validation gate); authoring-time verification proves answers and traps,
  not turn-budget fairness. Budgets were set at ≥1.5× the minimum
  perfect-agent call count, but a verbose-but-correct agent could still
  exhaust turns — audit any all-models-fail task before trusting it.
- `python_eval`-capable tasks can be solved in fewer calls than the
  nominal minimum by inlining stages into one eval; min-call counts are
  conservative.
- `debug` predicates rely on the spec's "don't edit tests/main.py" rule
  being followed; a model that rewrites the harness to print expected
  strings would pass the predicate while violating the spec (end-state
  predicates can't see process). Trajectory audit needed if a weak model
  suspiciously aces this category.
- No multi-seed data yet; per-task flakiness unknown until an N=8 run.
<!-- END HAND -->
