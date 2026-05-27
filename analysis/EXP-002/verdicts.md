---
doc_id: exp-002-verdicts
title: EXP-002 verdicts — 480 cells (480 done, 0 error)
zone: lab
kind: guide
status: active
owner: m
created: '2026-05-26'
last_updated: '2026-05-26'
last_verified: '2026-05-26'
tags:
- lab
- guide
- analysis
- exp-002
---
# EXP-002 verdicts — 480 cells (480 done, 0 error)

## Per-(model, scorer) means with bootstrap 95% CI

| model | scorer | mean | n | 95% CI |
|---|---|---|---|---|
| qwen3-14b-q4 | end_state | 0.750 | 96 | [0.667, 0.833] |
| qwen3-14b-q4 | tool_correctness | 1.000 | 96 | [1.000, 1.000] |
| qwen3-14b-q4 | budget_respected | 1.000 | 96 | [1.000, 1.000] |
| qwen3-14b-q4 | trajectory_judge | 1.000 | 8 | [1.000, 1.000] |
| llama3.1-8b-q4 | end_state | 0.250 | 96 | [0.167, 0.333] |
| llama3.1-8b-q4 | tool_correctness | 0.667 | 96 | [0.562, 0.760] |
| llama3.1-8b-q4 | budget_respected | 1.000 | 96 | [1.000, 1.000] |
| llama3.1-8b-q4 | trajectory_judge | 0.875 | 8 | [0.625, 1.000] |
| gpt-oss-20b-cloud | end_state | 0.833 | 96 | [0.760, 0.906] |
| gpt-oss-20b-cloud | tool_correctness | 0.948 | 96 | [0.896, 0.990] |
| gpt-oss-20b-cloud | budget_respected | 0.990 | 96 | [0.969, 1.000] |
| gpt-oss-20b-cloud | trajectory_judge | 1.000 | 8 | [1.000, 1.000] |
| glm-5.1-cloud | end_state | 0.833 | 96 | [0.760, 0.906] |
| glm-5.1-cloud | tool_correctness | 0.990 | 96 | [0.969, 1.000] |
| glm-5.1-cloud | budget_respected | 1.000 | 96 | [1.000, 1.000] |
| glm-5.1-cloud | trajectory_judge | 1.000 | 8 | [1.000, 1.000] |
| gpt-oss-120b-cloud | end_state | 0.833 | 96 | [0.760, 0.906] |
| gpt-oss-120b-cloud | tool_correctness | 0.958 | 96 | [0.917, 0.990] |
| gpt-oss-120b-cloud | budget_respected | 0.979 | 96 | [0.948, 1.000] |
| gpt-oss-120b-cloud | trajectory_judge | 1.000 | 8 | [1.000, 1.000] |

## end_state pass@1 / pass^8 per (model, task)

| model | task | pass@1 | pass^8 |
|---|---|---|---|
| qwen3-14b-q4 | code-find-and-fix-bug | 1.000 | 1.000 |
| qwen3-14b-q4 | code-read-and-explain | 1.000 | 1.000 |
| qwen3-14b-q4 | code-write-and-execute | 1.000 | 1.000 |
| qwen3-14b-q4 | fs-grep-extract-and-write | 1.000 | 1.000 |
| qwen3-14b-q4 | fs-read-and-copy | 1.000 | 1.000 |
| qwen3-14b-q4 | fs-write-csv-summary | 1.000 | 1.000 |
| qwen3-14b-q4 | http-fetch-and-count | 0.000 | 0.000 |
| qwen3-14b-q4 | http-fetch-and-extract | 1.000 | 1.000 |
| qwen3-14b-q4 | multi-db-self-check | 1.000 | 1.000 |
| qwen3-14b-q4 | multi-words-and-hash | 1.000 | 1.000 |
| qwen3-14b-q4 | shell-count-lines | 0.000 | 0.000 |
| qwen3-14b-q4 | shell-pipeline-extract | 0.000 | 0.000 |
| llama3.1-8b-q4 | code-find-and-fix-bug | 1.000 | 1.000 |
| llama3.1-8b-q4 | code-read-and-explain | 1.000 | 1.000 |
| llama3.1-8b-q4 | code-write-and-execute | 0.000 | 0.000 |
| llama3.1-8b-q4 | fs-grep-extract-and-write | 0.000 | 0.000 |
| llama3.1-8b-q4 | fs-read-and-copy | 0.000 | 0.000 |
| llama3.1-8b-q4 | fs-write-csv-summary | 0.000 | 0.000 |
| llama3.1-8b-q4 | http-fetch-and-count | 0.000 | 0.000 |
| llama3.1-8b-q4 | http-fetch-and-extract | 0.000 | 0.000 |
| llama3.1-8b-q4 | multi-db-self-check | 1.000 | 1.000 |
| llama3.1-8b-q4 | multi-words-and-hash | 0.000 | 0.000 |
| llama3.1-8b-q4 | shell-count-lines | 0.000 | 0.000 |
| llama3.1-8b-q4 | shell-pipeline-extract | 0.000 | 0.000 |
| gpt-oss-20b-cloud | code-find-and-fix-bug | 1.000 | 1.000 |
| gpt-oss-20b-cloud | code-read-and-explain | 1.000 | 1.000 |
| gpt-oss-20b-cloud | code-write-and-execute | 1.000 | 1.000 |
| gpt-oss-20b-cloud | fs-grep-extract-and-write | 1.000 | 1.000 |
| gpt-oss-20b-cloud | fs-read-and-copy | 1.000 | 1.000 |
| gpt-oss-20b-cloud | fs-write-csv-summary | 1.000 | 1.000 |
| gpt-oss-20b-cloud | http-fetch-and-count | 0.000 | 0.000 |
| gpt-oss-20b-cloud | http-fetch-and-extract | 0.000 | 0.000 |
| gpt-oss-20b-cloud | multi-db-self-check | 1.000 | 1.000 |
| gpt-oss-20b-cloud | multi-words-and-hash | 1.000 | 1.000 |
| gpt-oss-20b-cloud | shell-count-lines | 1.000 | 1.000 |
| gpt-oss-20b-cloud | shell-pipeline-extract | 1.000 | 1.000 |
| glm-5.1-cloud | code-find-and-fix-bug | 1.000 | 1.000 |
| glm-5.1-cloud | code-read-and-explain | 1.000 | 1.000 |
| glm-5.1-cloud | code-write-and-execute | 1.000 | 1.000 |
| glm-5.1-cloud | fs-grep-extract-and-write | 1.000 | 1.000 |
| glm-5.1-cloud | fs-read-and-copy | 1.000 | 1.000 |
| glm-5.1-cloud | fs-write-csv-summary | 1.000 | 1.000 |
| glm-5.1-cloud | http-fetch-and-count | 0.000 | 0.000 |
| glm-5.1-cloud | http-fetch-and-extract | 0.000 | 0.000 |
| glm-5.1-cloud | multi-db-self-check | 1.000 | 1.000 |
| glm-5.1-cloud | multi-words-and-hash | 1.000 | 1.000 |
| glm-5.1-cloud | shell-count-lines | 1.000 | 1.000 |
| glm-5.1-cloud | shell-pipeline-extract | 1.000 | 1.000 |
| gpt-oss-120b-cloud | code-find-and-fix-bug | 1.000 | 1.000 |
| gpt-oss-120b-cloud | code-read-and-explain | 1.000 | 1.000 |
| gpt-oss-120b-cloud | code-write-and-execute | 1.000 | 1.000 |
| gpt-oss-120b-cloud | fs-grep-extract-and-write | 1.000 | 1.000 |
| gpt-oss-120b-cloud | fs-read-and-copy | 1.000 | 1.000 |
| gpt-oss-120b-cloud | fs-write-csv-summary | 1.000 | 1.000 |
| gpt-oss-120b-cloud | http-fetch-and-count | 0.000 | 0.000 |
| gpt-oss-120b-cloud | http-fetch-and-extract | 0.000 | 0.000 |
| gpt-oss-120b-cloud | multi-db-self-check | 1.000 | 1.000 |
| gpt-oss-120b-cloud | multi-words-and-hash | 1.000 | 1.000 |
| gpt-oss-120b-cloud | shell-count-lines | 1.000 | 1.000 |
| gpt-oss-120b-cloud | shell-pipeline-extract | 1.000 | 1.000 |

## H1 — Cloud tool-call accuracy ≥ 0.60

- cells used: 288 (cloud × tasks × seeds)
- mean tool_correctness: **0.965** (95% CI [0.941, 0.986])
- rule: ≥ 0.60
- **H1: CONFIRMED**

## H2 — Local tool-call accuracy ≥ 0.40

- cells used: 192 (local × tasks × seeds)
- mean tool_correctness: **0.833** (95% CI [0.776, 0.880])
- rule: ≥ 0.40
- **H2: CONFIRMED**

## H3 — Multi-turn reliability cliff (∃ local L with mean pass^8/pass^1 < 0.70 on end_state)

| local model | reliability_ratio | n_tasks_with_p1>0 | verdict |
|---|---|---|---|
| qwen3-14b-q4 | 1.000 | 9 | ≥ 0.70 ✗ |
| llama3.1-8b-q4 | 1.000 | 3 | undefined (n_tasks<6) |

- **H3: REFUTED**

## H4 — cost/turn ratio ≥ 1.5 × latency/turn ratio (gpt-oss-120b vs gpt-oss-20b)

- gpt-oss-20b-cloud: cost/turn weight = 1.000, latency/turn = 2374.4 ms (n=96)
- gpt-oss-120b-cloud: cost/turn weight = 6.000, latency/turn = 3073.0 ms (n=96)
- cost_ratio = 6.000, latency_ratio = 1.294, 1.5×latency_ratio = 1.941
- rule: cost_ratio ≥ 1.5 × latency_ratio
- **H4: CONFIRMED**

## Per-tool success rate (across all cells)

| tool | attempts | errors | success rate |
|---|---|---|---|
| fs_grep | 88 | 16 | 0.818 |
| fs_read | 244 | 16 | 0.934 |
| fs_write | 444 | 1 | 0.998 |
| http_fetch | 99 | 2 | 0.980 |
| python_eval | 137 | 2 | 0.985 |
| shell_exec | 114 | 2 | 0.982 |

## Per-model trajectory patterns

| model | done | error | budget_exhausted | max_turns_reached | litellm_error | model_finished | over-budget | hallucinated tools | never invoked |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-14b-q4 | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| llama3.1-8b-q4 | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| gpt-oss-20b-cloud | 96 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| glm-5.1-cloud | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gpt-oss-120b-cloud | 96 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |

## Cell coverage (expected 96/model = 12 tasks × 8 seeds)

| model | done | error | total |
|---|---|---|---|
| qwen3-14b-q4 | 96 | 0 | 96/96 |
| llama3.1-8b-q4 | 96 | 0 | 96/96 |
| gpt-oss-20b-cloud | 96 | 0 | 96/96 |
| glm-5.1-cloud | 96 | 0 | 96/96 |
| gpt-oss-120b-cloud | 96 | 0 | 96/96 |

## Scorer coverage (rows with non-null value)

| scorer | non-null cells | null/missing |
|---|---|---|
| end_state | 480 | 0 |
| tool_correctness | 480 | 0 |
| budget_respected | 480 | 0 |
| trajectory_judge | 40 | 440 |
