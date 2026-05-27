# EXP-006b — SUMMARY

Cells: 288/288 (dense=96, moe=96, cloud=96, errored=0).

## Pre-registered hypotheses

- **H1 — Baseline measurement (no gate).** qwen3-14b-q4 end_state = **0.667** [0.573, 0.760] (n=96). This is the new lab reference for the post-fix PBS-Agent v0.1 surface.
- **H2 — Headline (relative delta).** qwen3-30b-a3b-moe end_state = **0.833** [0.750, 0.906]; lower-CI = 0.750; threshold = dense_pe + 0.10 = 0.767. -> **REFUTED**
- **H3 — Gap closure.** gap_closure_pe = **0.552** (dense=0.667, moe=0.833, cloud=0.969); threshold >= 0.50. -> **CONFIRMED**
- **H4 — Tool-correctness ceiling (relaxed).** qwen3-30b-a3b-moe tool_correctness = **0.917** [0.865, 0.969]; lower-CI = 0.865; threshold >= 0.90. -> **REFUTED**

## Decision

**Promotion: NO — qwen3-30b-a3b-moe does NOT promote. H2 = REFUTED, H3 = CONFIRMED. Lab default stays on qwen3-14b-q4 (reasoning-OFF).**

## Headline

qwen3-30b-a3b-moe does not clear the H2 + H3 promotion gate on the re-anchored surface. Lab default stays on qwen3-14b-q4 (reasoning-OFF). F-010 records the verdicts and operational notes; further MoE work is deferred to a separate experiment.
