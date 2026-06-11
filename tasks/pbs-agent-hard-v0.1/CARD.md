---
doc_id: pbs-agent-hard-v0-1-card
title: pbs-agent-hard-v0.1 — task suite
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
- pbs-agent-hard-v0.1
---

## Purpose

<!-- BEGIN HAND -->
The difficulty tier above pbs-agent-v0.1, built after that suite
saturated (two models at 1.000 in EXP-007/CODER-BENCH-001). 32 tasks
designed to separate strong local coding agents: multi-file bug hunts,
ETL pipelines over multiple files, multi-hop HTTP fixture chains, shell
log-analysis pipelines with deliberate edge cases. Every task has a
machine-verified expected answer (computed and checked before the
predicate was written) and a deterministic end-state predicate. All
tasks reference `system_prompt_id: tool_use_system_v2` (act-don't-narrate;
see F-013). HTTP tasks use offline fixtures under `_http_fixtures/` —
fixture hosts must be DNS-resolvable reserved domains
(example.com/org/net); the sandbox resolves before fixture lookup.
<!-- END HAND -->

<!-- BEGIN AUTOGEN -->

## Categories

- `code` — 8 task(s)
- `data` — 8 task(s)
- `multi` — 8 task(s)
- `shell` — 8 task(s)

## Difficulty distribution

- hard: 26
- medium: 6

## Tools used (union across tasks)

- `fs_grep`
- `fs_read`
- `fs_write`
- `http_fetch`
- `python_eval`
- `shell_exec`

## Budgets

- max_turns: 8–18
- tool_budget: 10–20

## Pre-reg shape

- success_predicate types:
  - `workspace_file_contains` — 25 task(s)
  - `all_of` — 7 task(s)

## Experiments using this suite

- `EXP-008` (HARD-BENCH-001 / HARD-BENCH-002)
- `EXP-009` (HARD-BENCH-003)

## Findings citing this suite

- [F-012](../../docs/findings/F-012-agentic-tool-calling-failure-modes.md) — tool-calling fidelity gates local agent performance
- [F-013](../../docs/findings/F-013-prompt-robustness-model-property.md) — prompt robustness is a model property

<!-- END AUTOGEN -->

## Known limitations

<!-- BEGIN HAND -->
- Headroom is thin at the top: gemma4-12b scores 0.938 (2 tasks of
  margin). A harder tier is needed to rank frontier-quality local
  agents; until it exists, ties near 0.94 are uninformative.
- Run-to-run variance at temperature 0 is real (~2 tasks for gemma4
  across reruns); single-seed per-task claims are soft. Multi-seed
  results: EXP-009.
- Original revision had 3 `multi` tasks pointing at invented fixture
  subdomains that NXDOMAIN'd in the sandbox; fixed in c4e56a7 by moving
  to reserved domains. Lesson recorded above in Purpose.
- `code` category leans on bug-fix/algorithm-implementation patterns;
  no tasks yet require reading failing test output or recovering from a
  wrong intermediate step (planned for the next tier).
<!-- END HAND -->
