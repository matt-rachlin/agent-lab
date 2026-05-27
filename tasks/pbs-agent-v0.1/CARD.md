---
doc_id: pbs-agent-v0-1-card
title: pbs-agent-v0.1 — task suite
zone: lab
kind: card
status: active
owner: m
created: '2026-05-26'
last_updated: '2026-05-26'
last_verified: '2026-05-26'
tags:
- lab
- card
- tasks
- pbs-agent-v0.1
---

## Purpose

<!-- BEGIN HAND -->
TODO: one-paragraph description of what pbs-agent-v0.1 measures.
<!-- END HAND -->

<!-- BEGIN AUTOGEN -->

## Categories

- `code` — 3 task(s)
- `fs` — 3 task(s)
- `http` — 2 task(s)
- `multi` — 2 task(s)
- `shell` — 2 task(s)

## Difficulty distribution

- easy: 4
- hard: 1
- medium: 7

## Tools used (union across tasks)

- `fs_grep`
- `fs_read`
- `fs_write`
- `http_fetch`
- `python_eval`
- `shell_exec`

## Pre-reg shape

- success_predicate types:
  - `db_query` — 1 task(s)
  - `workspace_file_contains` — 8 task(s)
  - `workspace_file_equals` — 2 task(s)
  - `workspace_file_exists` — 1 task(s)
- rubric types:
  - `tool_call` — 12 task(s)

## Experiments using this suite

- `EXP-002`

## Findings citing this suite

- [F-005](../../docs/findings/F-005-*.md) — F-005: The 12 GB Agent v0.2 — local tool-call is real; end-state difficulty is the binding constraint

<!-- END AUTOGEN -->

## Known limitations

<!-- BEGIN HAND -->
_Hand-curated list. Add limits, gotchas, and known-bad behaviour here._
<!-- END HAND -->
