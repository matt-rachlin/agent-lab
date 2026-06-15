---
doc_id: f-013-prompt-robustness-model-property
title: 'F-013: EXP-008 — prompt robustness is a model property and single-prompt
  benchmarks silently measure it. One act-don''t-narrate sentence: Devstral-24B
  +15.6pp, Qwen3-Coder +3.1pp, gemma4-12b ±0 on the same 32 tasks.'
zone: lab
kind: finding
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
depends_on:
- kind: doc
  target: exp-008
- kind: doc
  target: f-012
- kind: artifact
  target: 'lab:prompts/library/tool-use-system-v2.md'
- kind: artifact
  target: 's3://lab trajectory JSONL: HARD-BENCH-001 vs HARD-BENCH-002, devstral exact-replay A/B'
tags:
- lab
- finding
- findings
- agentic
- prompt-robustness
- system-prompt
- confidence-medium
- importance-7
---

# F-013: Prompt robustness is a model property; single-prompt benchmarks silently measure it

## TL;DR

**Appending one sentence to the shared system prompt — "act only via
tool calls; never describe or plan in text; a reply without a tool call
ends the session" (tool_use_system_v2) — moved the three models by
+15.6pp, +3.1pp, and 0pp on the identical 32-task hard suite.** The
size of a model's response to prompt strictness is itself a stable,
measurable property:

| model | v1 | v2 | Δ |
|---|---|---|---|
| gemma4-12b | 0.938 | 0.938 | 0 |
| qwen3-coder-30b | 0.781 | 0.812 | +0.031 |
| devstral-24b | 0.375 | 0.531 | **+0.156** |

The mechanism was isolated by an exact-replay A/B on a single Devstral
trajectory *before* the suite rerun: same task, same messages, polite v1
prompt → plan narration with markdown pseudo-code and zero tool calls;
v1 + the one sentence → immediate structured tool call. The suite-level
rerun then reproduced it at scale.

## Interpretation

- A model that needs scaffold-flavored imperatives to act (Devstral)
  carries its home harness's prompt as an implicit dependency; its
  agentic training did not generalize to a generic scaffold. A model
  that acts correctly under any reasonable prompt (gemma4) has the
  robust version of the same training.
- **Benchmark methodology consequence:** a fixed shared prompt — the
  fair default, and what this lab and most public agent benchmarks do —
  partially measures prompt robustness, not just capability. Cross-model
  agent comparisons should either report a prompt-sensitivity delta per
  model (as here) or state the single-prompt caveat.
- The fix is one-directional and bounded: v2 hurt nothing (no model
  regressed) but closed only ~40% of Devstral's gap — its remaining
  41pp deficit vs gemma4, concentrated in long-horizon `multi` tasks
  (3/8), is capability, not prompting.

## Caveats

- Single seed (n=32 cells/arm): Devstral's +15.6pp (5 tasks) far exceeds
  observed run variance (~2 tasks for gemma4 across reruns) and has a
  replay-isolated mechanism — confidence high for the direction, medium
  for magnitude. Qwen3-Coder's +3.1pp is one task and **within run
  variance — treat as noise until EXP-009.** Hence
  `confidence-medium` overall; EXP-009/HARD-BENCH-003 (N=8) tightens
  this.

## Consequences

- tool_use_system_v2 is the suite default going forward (all
  pbs-agent-hard-v0.1 tasks reference it as of de54642).
- Any future cross-model agent comparison in the lab reports results
  under v2 and, where a model underperforms expectations, runs the
  narrate-vs-act replay check before concluding capability deficit.
- Public writeup: `docs/writeups/local-coding-agent-benchmark.md`.
trust_level: unverified
