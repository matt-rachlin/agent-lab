---
doc_id: f-010-qwen3-30b-moe-re-anchored-not-promoted-h2-h4-fail
title: 'F-010: EXP-006b — Re-anchored MoE comparison. H2 REFUTED (lower-CI 0.750
  vs threshold 0.767, narrow by 0.017); H3 CONFIRMED (gap_closure = 0.552); H4
  REFUTED (tool_correctness lower-CI 0.865 vs 0.90 threshold). qwen3-30b-a3b-moe
  NOT promoted; lab default stays on qwen3-14b-q4.'
zone: lab
kind: finding
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
depends_on:
- kind: doc
  target: exp-006b
- kind: doc
  target: f-009-qwen3-30b-moe-refuted-h1-invalid
- kind: doc
  target: f-005-12gb-agent-v0-2-tool-use
- kind: code
  target: lab:scripts/analyze_exp006b.py
- kind: artifact
  target: lab:analysis/EXP-006b/SUMMARY.md
- kind: artifact
  target: lab:analysis/EXP-006b/verdicts.md
- kind: artifact
  target: lab:analysis/EXP-006b/per_task_endstate.csv
- kind: artifact
  target: lab:analysis/EXP-006b/per_cell.csv
- kind: artifact
  target: lab:analysis/EXP-006b/gap_closure.csv
- kind: artifact
  target: lab:analysis/EXP-006b/tokens_summary.csv
tags:
- lab
- finding
- findings
- moe
- qwen3
- 30b
- agent
- re-anchored
- not-promoted
- confidence-high
- importance-7
---

# F-010: EXP-006b — qwen3-30b-a3b-moe re-anchored; H2 narrow miss, H4 narrow miss; NOT promoted

## TL;DR

**qwen3-30b-a3b-moe does NOT promote to `lab-default-local` on the post-fix
PBS-Agent v0.1 surface.** EXP-006b ran the full 288-cell sweep at 288/288 done,
0 errors. The pre-registered promotion gate is `H2 AND H3`; the headline
result is a narrow but unambiguous fail on H2:

- **H1 — Baseline (no gate).** qwen3-14b-q4 end_state = **0.667**
  [0.573, 0.760] (n=96, percentile bootstrap). This is the new lab reference
  for the post-fix surface, superseding F-005's 0.750 anchor (now known to
  have been load-bearing on the broken http fixture + trivially-passable
  `code-find-and-fix-bug` predicate) and EXP-006's 0.583 (F-009's INVALID
  sweep number, on a partially-broken surface).
- **H2 — Headline lift. REFUTED (narrow).** MoE end_state = **0.833**
  [0.750, 0.906]. Lower-CI = **0.750**; threshold = dense_pe + 0.10 =
  **0.767**. Lower-CI is 0.017 below threshold. The MoE arm beats dense by
  +16.7pp at the point estimate, but the CI just barely fails to clear the
  pre-reg's +10pp lower-CI bar.
- **H3 — Gap closure. CONFIRMED.** gap_closure = (0.833 − 0.667) / (0.969 −
  0.667) = **0.552** ≥ 0.50. MoE closes 55.2% of the dense→cloud end_state
  gap on this surface.
- **H4 — Tool-correctness ceiling. REFUTED (narrow).** MoE tool_correctness =
  **0.917** [0.865, 0.969]. Lower-CI = 0.865 vs threshold 0.90. Lower-CI is
  0.035 below threshold; mean is comfortably above.

**Decision per pre-reg promotion rule (`H2 AND H3`)**: NOT PROMOTED. H3
passes, but H2 fails by the narrowest of margins. Lab default stays on
qwen3-14b-q4 (reasoning-OFF). H4 fails too — independently a quality caveat
were the model to promote on some other basis.

The promotion gate is not crossed, but **the MoE arm is no longer broken**:
the F-009 zero-tool-call shape (40/96 MoE cells with `tool_call_count = 0`)
is **fully gone — 0/96 in EXP-006b** — confirming Fix #69
(`--chat-template-kwargs enable_thinking=false`) closed the MoE tool-emission
gap as intended. The MoE arm is now a credible candidate that simply does
not clear the +10pp lower-CI bar on this 96-cell sample. A larger N (≥256
cells) might resolve the H2 verdict either way; that would be a separate
experiment.

## Setup

- **Experiment:** EXP-006b (pre-reg:
  [`docs/exp/EXP-006b-qwen3-30b-moe-re-anchored.md`](../exp/EXP-006b-qwen3-30b-moe-re-anchored.md),
  pre-reg commit `f4a1d48`).
- **Sweep config:**
  [`conf/sweep/EXP-006b.yaml`](../../conf/sweep/EXP-006b.yaml).
- **Models (3):** identical model list to EXP-006:
  - `qwen3-14b-q4` (local ollama, `think:false`) — baseline (H1).
  - `qwen3-30b-a3b-moe` (local llama.cpp Phase 19d CUDA build via llama-swap,
    `-ngl 99 -ot exps=CPU`, with chat-template fix at commit `1141dc1`:
    `--chat-template-kwargs enable_thinking=false`) — treatment (H2, H3, H4).
  - `gpt-oss-120b-cloud` (Ollama Cloud Pro) — ceiling reference (H3).
- **Tasks (12):** PBS-Agent v0.1, post-fix surface (success predicates
  tightened, fixtures repaired, prompt regressions reverted).
- **Config (1):** `greedy-1024` — `temperature=0.0`, `top_p=1.0`,
  `max_tokens=1024`.
- **Cells:** 12 tasks × 3 models × 1 config × 8 seeds = **288 runs.**
- **Pass rate:** **288/288 done, 0 errors.** No kill criterion fired.
- **CI method:** percentile bootstrap, n_resamples = 2000, seed = 42 (per
  `scripts/analyze_exp006b.py:_bootstrap_ci`). Same method as F-009.

### Pre-sweep follow-up commits (the reason for the re-anchor)

| commit | one-line |
|---|---|
| `cceaf62` | `multi-db-self-check` success predicate tightened (must now contain `"6.0"`). |
| `f48c517` | `multi-words-and-hash` prompt fix (task-local hashing). |
| `1141dc1` | MoE tool-emission gap closed via `--chat-template-kwargs enable_thinking=false`. |
| `971eb38` | Fix #70 — token-count passthrough for agent cells; `tokens_in / tokens_out` now populate. |
| `b0b0e96` | LiteLLM cold-load 502 retry policy for big-model lane. |

All five landed between F-009's sweep and EXP-006b's pre-reg commit `f4a1d48`.

## Dense-baseline comparison table — F-005 → F-009 → F-010

The dense (qwen3-14b-q4, reasoning-OFF) baseline has moved each time the
underlying surface has changed. This is the heart of the re-anchor.

| reference | dense end_state | 95% CI | n | surface |
|---|---|---|---|---|
| F-005 (EXP-002, May 2026) | 0.750 | (anchor, no published bootstrap) | 96 | pre-fix; broken http fixture + trivially-passable `code-find-and-fix-bug` predicate inflated this number |
| F-009 (EXP-006) | 0.583 | [0.479, 0.688] | 96 | partially-fixed surface; http fixture served but extraction failed; `code-find-and-fix-bug` predicate tightened (artefact gone); `multi-words-and-hash` regressed from task-global hashing prompt; sweep INVALID per H1 fail |
| **F-010 (EXP-006b, this finding)** | **0.667** | **[0.573, 0.760]** | 96 | post-fix surface; `multi-words-and-hash` prompt fix recovers `+8.3pp` of F-009's regression on that task; this is the new lab reference |

The F-010 dense baseline of 0.667 is the post-fix-surface anchor. It should
be the reference for any future EXP-NNN that wants to talk about local
agent quality on PBS-Agent v0.1.

## Per-hypothesis verdicts

### H1 — Baseline measurement (NOT a gate) · MEASURED

- **n** = 96 (12 tasks × 8 seeds, qwen3-14b-q4 only).
- **mean end_state** = **0.6667**.
- **95% bootstrap CI** = **[0.5729, 0.7604]**.

Per-task dense end_state (vs F-005 and F-009):

| task | F-005 | F-009 | F-010 | Δ F-009→F-010 |
|---|---|---|---|---|
| code-find-and-fix-bug | 1.000 | 0.000 | 0.000 | 0.000 (still 0; predicate now real) |
| code-read-and-explain | 1.000 | 1.000 | 1.000 | 0.000 |
| code-write-and-execute | 1.000 | 1.000 | 1.000 | 0.000 |
| fs-grep-extract-and-write | 1.000 | 1.000 | 1.000 | 0.000 |
| fs-read-and-copy | 1.000 | 1.000 | 1.000 | 0.000 |
| fs-write-csv-summary | 1.000 | 1.000 | 1.000 | 0.000 |
| http-fetch-and-count | 0.000 | 0.000 | 0.000 | 0.000 (dense still misses the extraction step) |
| http-fetch-and-extract | 0.000 | 0.000 | 0.000 | 0.000 (same) |
| multi-db-self-check | 1.000 | 1.000 | 1.000 | 0.000 |
| multi-words-and-hash | 1.000 | 0.000 | 1.000 | **+1.000** (Fix `f48c517` recovers the F-009 regression) |
| shell-count-lines | 0.000 | 0.000 | 0.000 | 0.000 |
| shell-pipeline-extract | 0.000 | 1.000 | 1.000 | 0.000 (carries the F-009 improvement) |

Net: F-009→F-010 delta is +0.083 (one task fully recovered:
`multi-words-and-hash`). Two http tasks remain at 0 — dense fires the tool
but cannot complete the downstream extraction; this is a real model
limitation, not a surface artefact, and is consistent across all three
sweeps.

### H2 — Headline lift (relative-delta, lower-CI bound) · REFUTED (narrow)

Rule: `lower_95_CI(MoE.end_state) ≥ mean(dense.end_state) + 0.10`.

- **n** = 96.
- **mean end_state** = **0.8333**.
- **95% bootstrap CI** = **[0.7500, 0.9062]**.
- **lower CI bound** = **0.7500**.
- **threshold** = dense_pe + 0.10 = 0.6667 + 0.10 = **0.7667**.

`0.7500 < 0.7667` by `0.0167`. **REFUTED** by 1.7 percentage points on the
lower CI bound.

Point estimate Δ vs dense: `+0.1667` (16.7pp). On a less conservative
gate — e.g. point-estimate-only at +10pp — H2 would have passed. The
pre-reg's choice of lower-CI bound at +10pp is the explicit guard against
sampling noise; it bites here.

Per-task: MoE matches dense on the 6 fs/code-pass tasks, beats dense on the
two stuck-at-zero tasks (`shell-count-lines`: 0→1; `http-fetch-and-extract`:
0→1), and ties dense everywhere else. There is no task where MoE
*regresses* vs dense in this sweep — the F-009 shell-pipeline-extract
trade is gone (both at 1.0 here).

### H3 — Gap closure (ratio, point estimate) · CONFIRMED

Rule: `gap_closure_pe := (moe_pe − dense_pe) / (cloud_pe − dense_pe) ≥ 0.50`.

| term | value |
|---|---|
| dense (qwen3-14b-q4) end_state | 0.6667 |
| moe (qwen3-30b-a3b-moe) end_state | 0.8333 |
| cloud (gpt-oss-120b-cloud) end_state | 0.9688 |
| denom (cloud − dense) | +0.3021 |
| numer (moe − dense) | +0.1667 |
| **gap_closure** | **0.5517** |

`0.552 ≥ 0.50`: **CONFIRMED** by ~5pp of headroom. MoE closes a majority
of the local-vs-cloud end_state gap on this surface.

Per-category gap closure (also in
[`analysis/EXP-006b/gap_closure.csv`](../../analysis/EXP-006b/gap_closure.csv)):

| category | dense | moe | cloud | gap_closure |
|---|---|---|---|---|
| code | 0.667 | 0.667 | 1.000 | 0.000 (MoE doesn't help on code) |
| fs | 1.000 | 1.000 | 1.000 | UNDEFINED (all 3 at ceiling) |
| http | 0.000 | 0.500 | 1.000 | 0.500 |
| multi | 1.000 | 1.000 | 0.812 | UNDEFINED (cloud below dense) |
| shell | 0.500 | 1.000 | 1.000 | 1.000 (MoE recovers shell-count-lines) |
| **overall** | **0.667** | **0.833** | **0.969** | **0.552** |

The gap-closure is carried by shell and http categories — exactly where
the F-009 MoE arm was failing for tool-emission reasons. Fix #69 turned
those zero-tool-call cells into wins (shell-count-lines) or partial wins
(http-fetch-and-extract). The code category is still stuck (both dense
and MoE fail `code-find-and-fix-bug`); MoE does not differentially close
the gap there.

### H4 — Tool-correctness ceiling (relaxed, lower-CI ≥ 0.90) · REFUTED (narrow)

Rule: `lower_95_CI(MoE.tool_correctness) ≥ 0.90`.

- **n** = 96.
- **mean tool_correctness** = **0.9167**.
- **95% bootstrap CI** = **[0.8646, 0.9688]**.
- **lower CI bound** = **0.8646**.
- **threshold** = **0.90**.

`0.8646 < 0.90` by `0.0354`. **REFUTED**. Mean is above threshold; only
the lower-CI bound trips the gate.

This is a dramatic improvement vs F-009's 0.500 — Fix #69 closed the
zero-tool-call gap entirely (see § Operational notes). The remaining
miss is at the +9pp mean level vs F-009; the gate just bites because the
pre-reg sets the lower-CI bar at 0.90 and the sample of N=96 doesn't
tighten the CI enough.

## Comparison vs F-009 (qwen3-30b-a3b-moe)

| metric | F-009 (EXP-006) | F-010 (EXP-006b) | Δ |
|---|---|---|---|
| MoE end_state mean | 0.583 | **0.833** | **+0.250** |
| MoE end_state lower-CI | 0.490 | **0.750** | **+0.260** |
| MoE tool_correctness mean | 0.500 | **0.917** | **+0.417** |
| MoE tool_correctness lower-CI | 0.406 | **0.865** | **+0.459** |
| MoE zero-tool-call cells | **40/96 (42%)** | **0/96 (0%)** | **−40 cells** |
| MoE mean tool calls / cell | 1.17 | **2.50** | **+1.33** |
| MoE mean turns / cell | 2.17 | (similar) | — |
| Sweep status | INVALID (H1 fail) | VALID (H1 is measurement) | — |

The Fix #69 chat-template change moves every metric in the right direction.
The MoE arm now behaves like a model that calls tools; it just doesn't
quite clear the +10pp lower-CI bar that the pre-reg requires for promotion.

## Operational notes

### MoE tool-emission post-template-fix

**Zero-tool-call cells: 40/96 (F-009) → 0/96 (F-010).** Fix #69
(`--chat-template-kwargs enable_thinking=false`) closed the F-009 mechanism
completely. MoE now fires `mean(tool_call_count) = 2.50` per cell, in line
with the cloud arm (2.51) and dense (2.50). The MoE arm's median
trajectory is no longer artificially-short; per-task tool_call_count
distributions are populated across all 12 tasks (see
`analysis/EXP-006b/per_cell.csv`).

This is the clearest single-commit win of the F-009 follow-up batch. The
H4 REFUTED verdict here is on a completely different shape than F-009's:
F-009 was "the model doesn't call the tool"; F-010 is "the model calls
the tool and occasionally gets the arguments wrong". The latter is a
normal model-quality finding, not a tool-wiring bug.

### Token capture verification (Fix #70)

`tokens_in` and `tokens_out` are now populated for **96/96 cells on all
three models** (288/288 total) — Fix #70 landed cleanly for both the
ollama route (dense) and the llama-swap+llama.cpp route (MoE) through
LiteLLM. Sample medians:

| model | tokens_in (p50 / mean) | tokens_out (p50 / mean) |
|---|---|---|
| qwen3-14b-q4 | 2074 / 2484 | 81 / 108 |
| qwen3-30b-a3b-moe | 2702 / 3500 | 77 / 160 |
| gpt-oss-120b-cloud | 2127 / 2544 | 194 / 209 |

No NULLs. Fix #70 is fully shipped; F-009's "tokens NULL for all 288
cells" caveat is closed.

### Latency notes

| model | latency mean (ms) | latency p50 | latency p95 |
|---|---|---|---|
| gpt-oss-120b-cloud | 10,102 | 9,601 | 15,206 |
| qwen3-14b-q4 (dense) | 18,728 | 16,612 | 32,617 |
| qwen3-30b-a3b-moe | 38,294 | 33,666 | 59,125 |

The MoE arm is ~2× the dense arm's latency at p50 and p95. On the lab's
12-task PBS-Agent v0.1 surface this isn't disqualifying, but for any
interactive loop a +2× latency cost for a +16.7pp end_state lift that
doesn't clear the lower-CI gate is a hard cost/benefit trade.

The cloud arm is fastest at p50 — Ollama Cloud Pro on a remote H100-class
GPU is unsurprisingly faster than llama-swap+llama.cpp on a 3080 Ti with
expert-offload-to-CPU.

### Other observations

- **`code-find-and-fix-bug`** is now a hard task: all three models fail it
  in F-010 except cloud (1.0). dense=0.000, MoE=0.000, cloud=1.000. The
  predicate was tightened (Fix `cceaf62`) and the change reveals the
  local models can't solve it. This is the dominant remaining drag on
  the local arms' end_state.
- **`multi-words-and-hash`** cloud arm regresses (1.000 → 0.625) vs F-009's
  surface, while both local arms hit 1.000. The hash check seems to be
  testing different things on different surfaces; worth a side-look but
  not load-bearing here.
- **No regressions** vs F-009 on any task for the MoE arm. Every
  task-level Δ is ≥ 0.

## Decision: does MoE promote?

**No.** Per the pre-registered promotion rule (`H2 AND H3` both pass):

- H2 REFUTED — narrow miss at lower-CI 0.750 vs threshold 0.767.
- H3 CONFIRMED — gap_closure 0.552 ≥ 0.50.
- H4 REFUTED — narrow miss at lower-CI 0.865 vs threshold 0.90 (would have
  been a "promote with quality caveat" only if H2 had passed).

Lab default stays on `qwen3-14b-q4` (reasoning-OFF). No changes to
`conf/litellm-config.yaml`, no new ADR for default-local.

The MoE arm is now a credible candidate. The pre-registered gate was
designed conservatively (lower-CI, not point-estimate, at +10pp). At
the point-estimate level, MoE beats dense by 16.7pp and closes 55% of
the local-vs-cloud gap. A future experiment with larger N (≥ 256 cells)
could resolve H2 either way without changing the surface.

## Follow-ups

- **H2 resolution.** The narrow H2 miss suggests a larger-N follow-up
  (≥ 256 cells, perhaps via more seeds per task or by expanding the
  task set) could either confirm MoE's promotion or definitively refute
  it. Filed as a follow-up exploration; not urgent.
- **`code-find-and-fix-bug` for local models.** Both local arms hit 0
  here. Either the task is now too hard for the 12B/30B local class,
  or the predicate is too tight. Worth a deeper look before any future
  promotion attempt — this single task represents 8.3% of the headline
  end_state metric and both local arms zero on it.
- **`multi-words-and-hash` on cloud.** Cloud regressed to 0.625 on this
  task while both local arms are at 1.0. The hash-check shape may be
  sensitive to which model wrote the words — worth verifying the
  predicate isn't relying on per-model determinism.

## Artifacts

- [`analysis/EXP-006b/SUMMARY.md`](../../analysis/EXP-006b/SUMMARY.md)
- [`analysis/EXP-006b/verdicts.md`](../../analysis/EXP-006b/verdicts.md)
- [`analysis/EXP-006b/per_task_endstate.csv`](../../analysis/EXP-006b/per_task_endstate.csv)
- [`analysis/EXP-006b/per_cell.csv`](../../analysis/EXP-006b/per_cell.csv)
- [`analysis/EXP-006b/gap_closure.csv`](../../analysis/EXP-006b/gap_closure.csv)
- [`analysis/EXP-006b/tokens_summary.csv`](../../analysis/EXP-006b/tokens_summary.csv)
- Analyzer: [`scripts/analyze_exp006b.py`](../../scripts/analyze_exp006b.py)
trust_level: unverified
