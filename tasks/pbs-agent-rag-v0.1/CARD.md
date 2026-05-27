---
doc_id: task-suite-pbs-agent-rag-v0.1
title: pbs-agent-rag-v0.1 — task suite
kind: card
status: active
owner: m
created: 2026-05-26
last_updated: 2026-05-26
suite: pbs-agent-rag-v0.1
task_count: 6
categories: ['rag']
last_used_in: ['EXP-003b']
---

## Purpose

<!-- BEGIN HAND -->
TODO: one-paragraph description of what pbs-agent-rag-v0.1 measures.
<!-- END HAND -->

<!-- BEGIN AUTOGEN -->

## Categories

- `rag` — 6 task(s)

## Difficulty distribution

- easy: 1
- hard: 2
- medium: 3

## Tools used (union across tasks)

- `fs_write`
- `kb_query`

## Pre-reg shape

- success_predicate types:
  - `retrieval_recall` — 1 task(s)
  - `workspace_file_contains` — 3 task(s)
  - `workspace_file_exists` — 2 task(s)
- rubric types:
  - `tool_call` — 6 task(s)

## Experiments using this suite

- `EXP-003b`

## Findings citing this suite

- [F-006](../../docs/findings/F-006-*.md) — F-006: The lab RAG stack v0.1 — hybrid retrieval beats endpoints, locals depend on kb_query more than cloud

<!-- END AUTOGEN -->

## Known limitations

<!-- BEGIN HAND -->
_Hand-curated list. Add limits, gotchas, and known-bad behaviour here._
<!-- END HAND -->
