---
doc_id: f-003-exp-001-verdicts-tmp
title: EXP-001 verdicts — 144 cells, computed automatically
zone: lab
kind: finding
status: draft
owner: m
created: '2026-05-25'
last_updated: '2026-05-25'
last_verified: '2026-05-25'
tags:
- lab
- finding
- findings
---
# EXP-001 verdicts — 144 cells, computed automatically

## H1 — Reasoning gap on math

- gpt-oss-120b-cloud mean pass@1 on math-reasoning: **0.609**
- best local model mean pass@1 on math-reasoning: **0.500**
- delta: **+0.109** (rule: ≥ +0.15)
- Welch's t-test p-value (frontier vs all-locals per-task means): **0.1948**
**H1**: REFUTED (observed: +0.109, rule: ≥ +0.15)

## H2 — Knowledge near-parity

- gpt-oss-120b-cloud mean pass@1 on knowledge-recall: **0.984**
- best local model mean pass@1 on knowledge-recall: **1.000**
- delta: **-0.016** (rule: ≤ +0.10)
**H2**: CONFIRMED (observed: -0.016, rule: ≤ +0.10)

## H3 — Reasoning-mode advantage on format-following

- qwen3-14b-q4 mean pass@1: **0.500**
- gemma3-12b-q4 mean pass@1: **0.875** (delta vs qwen3 = -0.375, p=0.1235)
- llama3.1-8b-q4 mean pass@1: **0.750** (delta vs qwen3 = -0.250, p=0.3346)
**H3**: REFUTED (observed: -0.375, rule: ≥ +0.20)

## H4 — Reliability cliff

| model | reliability ratio (mean p^8 / mean p@1) |
|---|---|
| qwen3-14b-q4 | 0.854 |
| llama3.1-8b-q4 | 0.881 |
| gemma3-12b-q4 | 1.000 |
| phi4 | 0.976 |
| gpt-oss-20b-cloud | 0.907 |
| gpt-oss-120b-cloud | 0.895 |
- gpt-oss-120b-cloud reliability ratio: **0.895** (rule: ≥ 0.75)
- minimum local reliability ratio: **0.854** (rule: ≤ 0.50)
**H4**: REFUTED (frontier ≥0.75 ✓, some local >0.50 ✗)

## Sample sizes (must be 24 cells/model for full coverage)

| model | cells |
|---|---|
| qwen3-14b-q4 | 24/24 |
| llama3.1-8b-q4 | 24/24 |
| gemma3-12b-q4 | 24/24 |
| phi4 | 24/24 |
| gpt-oss-20b-cloud | 24/24 |
| gpt-oss-120b-cloud | 24/24 |

Total cells evaluated: **144** (target 144 = 6 models × 24 tasks)
