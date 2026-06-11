---
doc_id: exp-008
title: 'EXP-008: HARD-BENCH-001/002 — 32-task hard agentic suite + act-don''t-narrate
  prompt A/B (retroactive record)'
zone: lab
kind: exp
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
depends_on:
- kind: doc
  target: exp-007
- kind: doc
  target: pbs-agent-hard-v0-1-card
tags:
- lab
- exp
- agentic
- tool-use
- prompt-robustness
- pbs-agent-hard-v0.1
- retroactive
---

# EXP-008: HARD-BENCH-001/002 — hard suite ranking + prompt A/B

Date created: 2026-06-11 (runs executed 2026-06-10)
Status: complete
Pre-registered: **NO — retroactive record.** Hypotheses below are quoted
verbatim from the sweep configs (`conf/sweep/hard-bench-v1.yaml` in
bc9475b, `hard-bench-v2.yaml` in de54642), each committed before its run.
Same discipline lapse as [EXP-007](EXP-007-coder-bench.md); the
properly pre-registered follow-up is [EXP-009](EXP-009-hard-bench-multiseed.md).

## Question

EXP-007 saturated pbs-agent-v0.1 (two models at 1.000). On a genuinely
hard 32-task suite, (a) does the field separate, and (b) how much of the
separation is prompt-sensitivity rather than capability — specifically,
does a one-sentence "act, don't narrate" addition to the system prompt
close Devstral-24B's narration-driven gap?

## Setup

- suite: pbs-agent-hard-v0.1 (32 tasks; code/data/shell/multi × 8;
  26 hard / 6 medium; see suite CARD)
- models: gemma4-12b, qwen3-coder-30b, devstral-24b
- scaffold: react, temp 0.0, top_p 1.0, max_tokens 4096
- seeds: [1] — **single seed, well below ADR-004.** Run as a
  validation+first-ranking pass; the N=8 confirmation is EXP-009.
- HARD-BENCH-001: `system_prompt_id: tool_use_system_v1`
- HARD-BENCH-002: identical except `tool_use_system_v2` = v1 + one
  sentence: act only via tool calls; never describe or plan in text; a
  reply without a tool call ends the session.
- Mid-001 fix: 3 multi tasks used invented fixture subdomains that
  NXDOMAIN'd in the sandbox (resolver runs before fixture lookup);
  replaced with reserved domains and re-ran the 9 affected cells
  (c4e56a7). All 96 cells valid in the final table.

## Hypotheses (from sweep configs, pre-run)

- H-001: "On genuinely hard agentic tasks the field separates (vs 100%
  ties on the easy suite)."
- H-002: "A strict act-dont-narrate system prompt (v2) closes
  Devstral-24B's gap without hurting models that already act."

## Results

pass@1, n=32 cells/model/arm, single seed:

| model | v1 prompt | v2 prompt | Δ |
|---|---|---|---|
| gemma4-12b | **0.938** | **0.938** | 0 |
| qwen3-coder-30b | 0.781 | 0.812 | +0.031 |
| devstral-24b | 0.375 | 0.531 | **+0.156** |

Per-category (v2): gemma4 7-8/8 everywhere; qwen3-coder's losses
concentrate in `code` (4/8 — repeated misses on fibonacci-bug-fix,
interval-merge-fix, topo-sort, expr-parser-fix); devstral's in `multi`
(3/8, long-horizon chains).

- **H-001 CONFIRMED** — 56pp spread (94 vs 53→38) on the same tasks two
  models had tied at 100%-adjacent levels.
- **H-002 PARTIALLY CONFIRMED** — +15.6pp for Devstral, no harm to the
  leader (gemma4 Δ0, qwen3 +3.1), but "closes" overstates it: Devstral
  still trails gemma4 by 41pp with the prompt fix. The trigger was
  isolated by exact-replay A/B on a single trajectory before the suite
  rerun: with v1, Devstral answers with a plan and markdown pseudo-code
  and zero tool calls; same task with the one added sentence produces an
  immediate structured call.
- Caveat (motivates EXP-009): single seed. gemma4 failed *different*
  tasks in the two runs at the same 0.938 (data-duplicate-transaction +
  shell-top-error-sources in 001; code-interval-merge-fix +
  shell-fragment-reassembly in 002) — direct evidence of temp-0
  run-to-run variance, so per-task and small-Δ claims here are soft.
  qwen3's +3.1 is one task and within noise.

## Consequences

- Findings distilled: [F-012](../findings/F-012-agentic-tool-calling-failure-modes.md)
  (failure modes), [F-013](../findings/F-013-prompt-robustness-model-property.md)
  (prompt robustness).
- EXP-009 pre-registered: same suite/models/v2 prompt at N=8 seeds for
  CI-backed confirmation per ADR-004.
- Public writeup: `docs/writeups/local-coding-agent-benchmark.md`.
