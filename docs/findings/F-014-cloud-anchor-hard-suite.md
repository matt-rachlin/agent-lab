---
doc_id: f-014-cloud-anchor-hard-suite
title: 'F-014: EXP-011 — frontier cloud anchor on the hard suite. glm-5.1 scores
  1.000, qwen3-coder-480b 0.969; local gemma4-12b (0.938) sits 2 tasks off
  frontier; within the qwen3-coder family, 16x scale is worth +15.6pp and cures
  the narration failure mode. H1/H2/H3 all CONFIRMED.'
zone: lab
kind: finding
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
depends_on:
- kind: doc
  target: exp-011
- kind: doc
  target: f-012-agentic-tool-calling-failure-modes
- kind: doc
  target: f-013-prompt-robustness-model-property
- kind: code
  target: lab:scripts/analyze_exp011.py
- kind: artifact
  target: lab:analysis/EXP-011/SUMMARY.md
- kind: artifact
  target: lab:analysis/EXP-011/per_cell.csv
tags:
- lab
- finding
- findings
- agentic
- tool-use
- cloud-anchor
- pbs-agent-hard-v0.1
- confidence-medium
- importance-7
---

# F-014: Frontier cloud anchor on the hard suite — local is 2 tasks off frontier

## TL;DR

On pbs-agent-hard-v0.1 (32 tasks, react scaffold, v2 prompt, seed 1 —
like-for-like with HARD-BENCH-002):

| model | overall | code | data | multi | shell |
|---|---|---|---|---|---|
| glm-5.1-cloud | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| qwen3-coder-480b-cloud | 0.969 | 1.000 | 1.000 | 0.875 | 1.000 |
| gemma4-12b (local, 12 GB) | 0.938 | 0.875 | 1.000 | 1.000 | 0.875 |
| qwen3-coder-30b (local) | 0.812 | 0.500 | 1.000 | 0.875 | 0.875 |
| devstral-24b (local) | 0.531 | 0.625 | 0.625 | 0.375 | 0.500 |

All three pre-registered hypotheses **CONFIRMED**:

- **H1 (frontier ceiling):** best cloud arm 1.000 ≥ 0.938 — the local
  champion is not above frontier level; it is 2 tasks behind it on a
  suite a 12 GB card can serve.
- **H2 (within-family scale):** qwen3-coder-480b 0.969 vs the 30b's
  0.812 — 16× scale (475B-A35B MoE vs 30B-A3B) is worth **+15.6pp**
  here. Notably the 480b passes, with clean structured calls, all four
  `code` tasks the 30b fails *by narrating* (F-013/EXP-009-H5 watch
  item) — within one training lineage, scale cured the residual
  narration failure mode.
- **H3 (failure-mode absence):** trajectory scan of all 64 cloud
  episodes: zero text-emitted tool calls, zero narration episodes. The
  F-012 failure modes are a local/small-model phenomenon on this suite,
  not a scaffold artifact.

## Additional observations

- glm-5.1's perfect score means **pbs-agent-hard-v0.1 is saturated at
  the frontier** — it can rank local models but cannot rank
  frontier-class ones. The brutal tier (EXP-010, queued) exists for
  exactly this reason.
- The 480b's single miss is `multi-manifest-validate-compute` — the
  suite's hardest cell (also failed by devstral in both prompt arms and
  by qwen3-30b in v1). Worth a trajectory read during the EXP-010
  defect audit, since every non-glm model stumbles on it.
- The headline framing for the public writeup: *a 12B model running on
  a 12 GB consumer GPU is two tasks short of a frontier model on this
  agentic suite* — with the caveats below.

## Caveats

- Single seed (anchor pass, like HARD-BENCH-001); per-task and ±1-task
  comparisons are within observed run variance (~2 tasks for gemma4
  across reruns). The gemma4-vs-frontier gap statement is therefore
  directional; EXP-009 (N=8, running) tightens the local side.
- Cloud arms ran via Ollama Cloud through the local daemon; provider-
  side quantization/serving details are not observable from here.
