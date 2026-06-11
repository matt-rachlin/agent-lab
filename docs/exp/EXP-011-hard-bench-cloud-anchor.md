---
doc_id: exp-011
title: 'EXP-011: HARD-BENCH-CLOUD-001 — frontier cloud anchor on the hard suite
  (pre-registered)'
zone: lab
kind: exp
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
depends_on:
- kind: doc
  target: exp-008
- kind: doc
  target: exp-009
- kind: doc
  target: f-012-agentic-tool-calling-failure-modes
tags:
- lab
- exp
- agentic
- tool-use
- cloud-anchor
- pbs-agent-hard-v0.1
---

# EXP-011: HARD-BENCH-CLOUD-001 — frontier cloud anchor on the hard suite

Date created: 2026-06-11
Status: planned
Pre-registered: <commit SHA filled by `lab exp register` at registration time>

## Question

EXP-008 ranked local models on pbs-agent-hard-v0.1 with gemma4-12b at
0.938. How far is that from frontier-class cloud models on the same
tasks, same scaffold, same v2 prompt? Two specific anchors: (a) the best
historical cloud arm (glm-5.1, EXP-005's top scorer), and (b)
qwen3-coder-480b — the ~16× larger sibling of the local
qwen3-coder-30b, making a clean within-family scale comparison.

## Hypothesis

- **H1 (frontier ceiling):** the best cloud arm scores ≥ 0.938, i.e.
  ties or beats gemma4-12b — the suite's top local score is not above
  frontier level.
- **H2 (within-family scale):** qwen3-coder-480b beats qwen3-coder-30b
  (0.812 in HARD-BENCH-002) by ≥ 2 tasks (≥ 6.2pp) — scale helps
  agentic coding within one training lineage.
- **H3 (failure-mode absence):** neither cloud arm exhibits the F-012
  failure modes — zero episodes with text-emitted tool calls and zero
  narration-instead-of-action episodes across all 64 trajectories.

## Method

### Models

| litellm_id | role |
|---|---|
| glm-5.1-cloud | frontier anchor (Ollama Cloud via local daemon) |
| qwen3-coder-480b-cloud | within-family scale anchor |

Local comparators (numbers from HARD-BENCH-002, not re-run):
gemma4-12b 0.938, qwen3-coder-30b 0.812, devstral-24b 0.531.

### Matrix

- suite: pbs-agent-hard-v0.1 (sealed; identical revision to EXP-009)
- prompt: tool_use_system_v2; config: react, temp 0.0, top_p 1.0,
  max_tokens 4096
- seeds: [1] — anchor pass, same role as HARD-BENCH-001/EXP-010
  validation runs; per ADR-004 not a reportable reliability claim.
  Comparisons against HARD-BENCH-002 (also seed 1) are like-for-like.
- 32 cells/model, 64 total; runs on the `default` pueue group (no GPU —
  inference is remote) concurrently with HARD-BENCH-003.
- Cost: Ollama Cloud subscription usage (~300–600 chat calls).

## Success / failure criteria

- H1: CONFIRMED iff max(cloud pass@1) ≥ 0.938; REFUTED below 0.906
  (≥ 1 task short); 0.906–0.937 → INCONCLUSIVE (within single-seed
  noise observed in EXP-008).
- H2: CONFIRMED iff qwen3-coder-480b pass@1 ≥ 0.874 (0.812 + 2 tasks);
  REFUTED at ≤ 0.812; between → INCONCLUSIVE.
- H3: CONFIRMED iff trajectory scan finds zero text-emitted-call and
  zero narration episodes for both arms; any single episode REFUTES.

## Kill criteria

- Kill if > 10% of cells (≥ 7/64) fail on transport (cloud
  availability, daemon proxy errors, timeouts) rather than model
  behavior — rerun when the cloud is healthy instead of reporting
  contaminated numbers.
- Kill if concurrent execution measurably disturbs the GPU sweep
  (HARD-BENCH-003 cell-rate drop > 25% sustained) — cloud anchor
  yields; the pre-registered N=8 experiment has priority.

## Analysis plan

One report: cloud arms vs the three local HARD-BENCH-002 rows (overall
+ per category), H1–H3 verdicts, and a trajectory-scan appendix for H3
(grep for content-embedded JSON calls and zero-tool-call episodes).
Feeds the public writeup's "how far is local from frontier?" section.
