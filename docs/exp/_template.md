---
doc_id: lab-exp-template
title: 'EXP-NNN: <one-line title>'
zone: lab
kind: exp
status: active
owner: m
created: '2026-05-26'
last_updated: '2026-05-26'
last_verified: '2026-05-26'
tags:
- lab
- exp
---
# EXP-NNN: <one-line title>

<!--
Pre-registration template. Pattern after EXP-003a.md for retrieval/RAG
experiments and EXP-002.md for agent sweeps. The pre-reg discipline:
every section below must be filled in BEFORE the sweep runs, and the
file must be committed (the commit SHA is the pre-registration record).
-->

Date created: YYYY-MM-DD
Status: planned
Pre-registered: <commit SHA filled by `lab exp register` at registration time>

## Question

<One-paragraph statement of the decision this experiment informs. Frame
it as a question the data will answer — not the hypothesis itself. e.g.
"Within the lab's existing RAG infrastructure, how do hybrid blend alpha
and top-k affect synthetic-query recall, MRR, and nDCG?">

## Setup

<Concrete inventory of what is being held fixed and what is varied.>

### KB / corpora (if any)

| field | value |
|---|---|
| name | <kb-name or n/a> |
| status | sealed |
| chunks | <N> |
| embedding_model | <litellm_id> |

### Models

| litellm_id | role |
|---|---|
| <model-a> | <subject under test> |
| <model-b> | <subject under test> |
| <judge-model> | <judge / scorer, if used> |

### Matrix

- <axis-1> ∈ {…} — N values
- <axis-2> ∈ {…} — M values
- seeds: [1, 2, 3, 4, 5, 6, 7, 8]
- total cells: <N × M × …>

### Tasks

- Suite: <suite-name>
- Slug filter: <none | list of slugs>
- N tasks: <count>

### Estimated cost

- GPU-hours: <…>
- Cloud calls: <…>
- Wall time: <…>

## Hypotheses

<Two to four falsifiable, pre-registered hypotheses with explicit
pass/fail thresholds. Each hypothesis names the metric, the
comparison, and the rule that decides confirm/refute/inconclusive.>

- **H1 — <short claim>.** <Operational rule: "metric X ≥ threshold Y on
  condition Z", "config A beats config B on metric M at p < 0.05", etc.>
- **H2 — <short claim>.** <…>
- **H3 — <short claim>.** <…>

These hypotheses are independent; each is judged on its own evidence.

## Kill criteria

<Conditions under which we abort the sweep early rather than finish for
the sake of completeness. Examples:
- "If the pilot's mean tool_correctness < 0.20 across all models,
  abort and write a postmortem on tool wiring instead of completing
  the matrix."
- "If wall-time exceeds 12h on the first model, cancel and re-scope
  the matrix.">

## Confounders to control

<Axes we are deliberately holding fixed (and why), axes we are deliberately
varying, axes we know to be noisy but cannot control. Mention prior findings
that bear on the design (e.g. F-005 wired http fixtures; F-006 established
default alpha=0.5).>

## Reproduction

<How a reader can reproduce this experiment from a clean checkout.
Should reference the sweep YAML (`conf/sweep/EXP-NNN.yaml`), the
analysis script (`scripts/analyze_expNNN.py` or
`analysis/EXP-NNN/`), and any one-shot setup steps (KB build, query
cache, etc.).>

```bash
# 1. Register the plan (this file)
lab exp register docs/exp/EXP-NNN-<slug>.md

# 2. Run the sweep (enforces pre-registration)
lab sweep run conf/sweep/EXP-NNN.yaml --enforce-pre-registration

# 3. Analyze
uv run python scripts/analyze_expNNN.py

# 4. Write the finding
docs/findings/F-NNN-<short-claim>.md
```

## Pre-mortem

<Imagine it's <end date> and this experiment failed badly. List
plausible reasons + cheap mitigations now.>

- Risk: …  Mitigation: …
- Risk: …  Mitigation: …
