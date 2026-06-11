---
doc_id: exp-007
title: 'EXP-007: CODER-BENCH-001 — coding-specialized 30–32B vs generalist gemma4-12b
  on local agentic coding (retroactive record)'
zone: lab
kind: exp
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
depends_on:
- kind: doc
  target: f-005
- kind: doc
  target: adr-004-reliability-discipline
tags:
- lab
- exp
- agentic
- tool-use
- coding
- pbs-agent-v0.1
- retroactive
---

# EXP-007: CODER-BENCH-001 — coding-specialized 30–32B vs generalist gemma4-12b

Date created: 2026-06-11 (run executed 2026-06-10)
Status: complete
Pre-registered: **NO — retroactive record.** This sweep was run during an
interactive debugging/benchmarking session before the plan was written.
The hypothesis below is quoted verbatim from the sweep config
(`conf/sweep/coder-bench-v1.yaml`, committed pre-run in 4863c22), which is
the closest thing to a pre-registration this experiment has. Documented
after the fact so the lab record is complete; the lapse itself is noted
in the 2026-06-10 log entry.

## Question

On the lab's 12-task agentic suite (pbs-agent-v0.1), do coding-specialized
30–32B local models beat the generalist gemma4-12b? This decides the lab's
default local *coding agent* model and tests whether coding-leaderboard
strength transfers to agentic tool-loop work.

## Hypothesis

(Quoted verbatim from the sweep config, committed pre-run:) "A
coding-specialized 30-32B model beats the general gemma4-12b on local
agentic coding tasks."

## Method

### Models

| litellm_id | role |
|---|---|
| gemma4-12b | generalist incumbent |
| qwen3-coder-30b | coding-specialized MoE (~3B active), agentic-trained |
| qwen2.5-coder-32b-q4_k_m | coding-specialized dense, code-completion-trained |

Reference arm (separate experiment, LLAMA-70B-AGENT-BENCH-001, same suite
and config): llama-3.3-70b hybrid CPU/GPU offload — 0.583.

### Matrix

- suite: pbs-agent-v0.1 (12 tasks: code 3, fs 3, http 2, multi 2, shell 2)
- scaffold: react, temp 0.0, top_p 1.0, max_tokens 4096
- seeds: [1, 2, 3] — **below the ADR-004 N≥8 bar**; accepted because the
  outcome was saturated (see Results), making more seeds uninformative
  for the ranking question.
- 36 cells per model, 108 total.

## Success / failure criteria

(Reconstructed; not committed pre-run.) Hypothesis confirmed if either
specialist's pass@1 exceeds gemma4-12b's on the suite; refuted on tie or
reversal. Ranking decides the lab's local coding-agent default.

## Kill criteria

(Reconstructed.) Kill on harness fault — sandbox/tool-server errors or
scorer failures contaminating cells — rather than model behavior.
Not triggered: all 108 cells completed and scored.

## Results

| model | pass@1 (36 cells) | per-seed |
|---|---|---|
| gemma4-12b | **1.000** | 12/12, 12/12, 12/12 |
| qwen3-coder-30b | **1.000** | 12/12, 12/12, 12/12 |
| qwen2.5-coder-32b-q4_k_m | **0.000** | 0/12, 0/12, 0/12 |

**Hypothesis REFUTED** (and the refutation is bimodal): one specialist ties
the generalist at ceiling; the other scores zero — on every task, every
seed, every category. Trajectory inspection shows qwen2.5-coder-32b emits
its tool calls as JSON text inside `content` rather than structured
`tool_calls`, so the harness (correctly) treats each reply as a final
answer. The 70B reference arm shows the same text-emission failure plus
premature chain termination. Distilled in [F-012](../findings/F-012-agentic-tool-calling-failure-modes.md).

A harness-side fallback parser (FALLBACK-TEST, commit 600b7a8) that
extracts and executes content-embedded JSON tool calls recovered the
mechanics (1 → 6 executed calls on the probe task) but not the scores:
recovered calls contain placeholder pseudo-code (`content =
$response['content']`) assuming a variable-binding environment the
stateless tool loop doesn't provide. The weakness is agentic training,
not output format.

## Consequences

- Suite saturated at the top (two models at 1.000) → built the 32-task
  hard suite (pbs-agent-hard-v0.1) and ran [EXP-008](EXP-008-hard-bench.md).
- gemma4-12b remains the lab default coding agent pending EXP-008/009.
- Public writeup: `docs/writeups/local-coding-agent-benchmark.md`.
