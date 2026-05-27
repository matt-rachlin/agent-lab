---
doc_id: adr-003-task-taxonomy
title: 'ADR-003: PBS task taxonomy'
zone: lab
kind: adr
status: active
owner: m
created: '2026-05-25'
last_updated: '2026-05-25'
last_verified: '2026-05-25'
tags:
- lab
- adr
---
# ADR-003: PBS task taxonomy

Status: accepted
Date: 2026-05-25
Deciders: Matt Rachlin

## Context

The Personal Benchmark Suite (PBS) needs an explicit taxonomy of task categories. Without one, tasks accumulate ad-hoc and the per-model summary tables become uninformative ("model X is 73% across the board" tells us nothing about *what kind of work* X is good at).

The RESEARCH_OPS_PLAN proposed four categories: tool-using coding, multi-turn reasoning, desktop, research-workflow. Two of those (desktop, research-workflow) require environments we don't have yet (OS sandbox, real tool registry). Decision needed on PBS v0.1 scope.

## Decision

**PBS v0.1 uses three single-turn text-in/text-out categories** with deterministic rubrics:

1. **`math-reasoning`** — multi-step arithmetic, logic, probability. Verifiable answers (regex / exact match).
2. **`format-following`** — output respects a stated structural constraint (JSON, CSV, exact-char-count, markdown table). Rubric: regex on the shape.
3. **`knowledge-recall`** — factual recall with a verifiable answer. Mix of common-knowledge (cheap) and specialised (discriminating).

PBS v0.2 (Phase 5) will add:

- **`tool-use`** — single-tool function calls with schema-validated arguments. Requires the MCP-based tool harness to exist.
- **`multi-turn-reasoning`** — conversation-state tasks (tau-bench-style). Requires conversation simulator.

PBS v0.3+ (later) may add:

- **`desktop`** — OSWorld-style. Requires sandboxed VM env.
- **`research-workflow`** — paper triage, repo navigation, citation chains. Requires retrieval infrastructure.

## Consequences

- **Easier**: v0.1 tasks all run in <30s on 12 GB-class models with no external dependencies. Adding tasks is just YAML. Scoring is deterministic — no judge needed for the bulk.
- **Harder**: claims about general "agentic capability" require categories the lab hasn't built yet. Until then, every claim is scoped to text-in/text-out.
- **Risks**: format-following overlaps with math-reasoning ("output the number as JSON"). Mitigated by category being a primary tag, with secondary tags as labels in `payload.metadata`.

## Considered alternatives

- **Use BFCL v3** as PBS. Rejected for v0.1 because (a) BFCL is contaminated for many recent models, (b) it requires a tool harness, (c) we want lab-native data for the Pareto we plan to report.
- **Skip categories entirely, just use a flat task list**. Rejected — per-category summaries are the most useful axis of the report.
- **Start with all four planned categories**. Rejected — desktop and research-workflow need infrastructure that's a phase or two away.
