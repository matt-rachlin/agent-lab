---
doc_id: exp-002b
title: 'EXP-002b: qwen3-14b-q4 reasoning-ON ablation (matched comparison to EXP-002)'
zone: lab
kind: exp
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
tags:
- lab
- exp
- ablation
- qwen3
- reasoning
---

# EXP-002b: qwen3-14b-q4 reasoning-ON ablation (matched comparison to EXP-002)

Date created: 2026-05-27
Status: planned
Pre-registered: (commit SHA filled by `lab exp register` at registration time)
Source experiment for the baseline arm: [EXP-002](./EXP-002.md) /
[F-005](../findings/F-005-12gb-agent-v0.2-tool-use.md).

## Question

How much of qwen3-14b-q4's tool-use end_state accuracy is lost when
reasoning is left **on** (Ollama's default) versus the EXP-002 setting
of `think: false`? EXP-002 explicitly deferred this ablation to a
matched follow-up; this is that follow-up.

EXP-001b already showed `/no_think` in the system prompt does **not**
disable qwen3 reasoning — only the API-level `think: false` knob does.
EXP-002 used `think: false` for the qwen3 cells. EXP-002b reuses the
exact same 12-task suite, 8 seeds, scorers, and sandbox, but flips
`think: true` for the **only** model in the sweep.

## Hypothesis (single, pre-registered)

- **H1 — Reasoning-ON degrades end-state accuracy.** Mean `end_state`
  pass@1 for `qwen3-14b-q4` at `think: true` across the 12 PBS-Agent
  v0.1 tasks × 8 seeds is materially lower than EXP-002's measured
  baseline of **0.750** at `think: false`. Pre-registered decision
  rule:

  | think:true end_state mean | Verdict |
  | ------------------------- | ----------------------- |
  | ≥ 0.55                    | H1 **REFUTED**          |
  | 0.30 – 0.55 (exclusive)   | H1 **MIXED**            |
  | < 0.30                    | H1 **CONFIRMED**        |

  Equivalent operational reading: a CONFIRMED H1 means think-ON is
  worth ≥ 45 pp of end_state — strong evidence that reasoning mode is
  net-negative for this model class on agentic loops. A REFUTED H1
  means the reasoning trace's overhead is < 20 pp — within the
  variance band F-005 already documented. MIXED means there's a real
  signal but not strong enough to act on without further work.

  We pre-commit to reporting the verdict per the table above
  regardless of where the number lands; no peeking.

## Why this matters

F-004 already established that reasoning-ON is net-negative on
**single-turn** PBS-v0.1. EXP-002b extends that result to the
**agentic** regime. If the gap is even larger here (CONFIRMED), the
operational implication is "default qwen3-14b-q4 to `think: false`
everywhere we use it as a tool-use agent, not just on PBS-Agent". If
the gap is smaller (REFUTED), the agentic regime washes out reasoning
mode's noise — a useful negative result for routing decisions.

## Method

### Models (1)

| litellm_id      | Backend      | Per-model override |
| --------------- | ------------ | ------------------ |
| `qwen3-14b-q4`  | local Ollama | `extra: {think: true}` |

No other models. The compare-against baseline is EXP-002's
`qwen3-14b-q4` cells (already in the lab DB; identical task suite,
seeds, scorers).

### Config (1)

Identical to EXP-002:

```yaml
name: greedy-1024
temperature: 0.0
top_p: 1.0
max_tokens: 1024
scaffold: single_turn   # runner dispatches to agent path on max_turns>1
```

### Tasks (12)

Full PBS-Agent v0.1 suite (`pbs-agent-v0.1`), same 12 tasks as
EXP-002. No filter on slugs.

### Seeds (8)

`[1, 2, 3, 4, 5, 6, 7, 8]`, same as EXP-002.

### Total cells

12 tasks × 1 model × 1 config × 8 seeds = **96 runs**.

### Evaluators

Same deterministic scorers as EXP-002 (`end_state`, `tool_correctness`,
`budget_respected`). LLM-judge (`trajectory_judge`) on the same single
task `code-read-and-explain` × 8 seeds = **8 judge calls**.

### Statistics

- pass@1, pass⁸ per task on `end_state` (the H1 anchor).
- Bootstrap 95 % CI on the model-wide `end_state` mean (n_resamples=2000).
- Side-by-side comparison vs the EXP-002 `qwen3-14b-q4 / think:false`
  baseline pulled directly from the lab DB (`SLUG='EXP-002'`).
- Paired difference per task (think:false − think:true) with
  one-sided Wilcoxon p-value as a secondary signal (not in the
  decision rule, but reported).

## Success / failure criteria

Operational definition of the pre-registered verdict (no peeking at
data before the sweep completes):

- **H1 CONFIRMED** ⇔ think:true `end_state` mean across all 96 cells
  (12 tasks × 8 seeds) is **< 0.30**. Errored cells count as 0.0 in the
  denominator (same convention as EXP-002 / `analyze_exp002.py`).
- **H1 REFUTED** ⇔ think:true `end_state` mean **≥ 0.55**.
- **H1 MIXED** ⇔ think:true `end_state` mean in `[0.30, 0.55)`.

Side-channel quality checks (do not change the verdict, are reported
in `verdicts.md`):

- Bootstrap 95 % CI on the model-wide mean (n_resamples=2000).
- Per-task pass@1 / pass⁸ side-by-side vs the EXP-002 think:false rows.
- One-sided paired Wilcoxon p (think:false > think:true) over the 12
  per-task means.

## Kill criteria

If any of the following fire during the sweep, STOP the sweep, debug,
and only re-launch after the underlying issue is resolved AND the
pre-reg is re-validated:

- Cell error rate exceeds **5 %** (> 4 errored cells / 96).
- Sandbox failure rate exceeds **10 %** (a runner-level signal,
  distinct from cell errors).
- GPU lease contention causes > 3 cells to fail with
  `gpu_lease_timeout` (sibling agents may be using the GPU).
- The Ollama daemon crashes or has to be restarted mid-sweep.

If kill-criteria fire, the analysis script is still run on whatever
cells completed, but the H1 verdict is reported as `INVALID — sweep
killed` rather than CONFIRMED/MIXED/REFUTED.

## Cost ceiling

$0 — fully local GPU. The single judge call goes to
`gpt-oss-120b-cloud` (8 calls × ~1k tokens ≈ negligible spend).

## Confounders to control

- **Identical sandbox image** as EXP-002 (`manifest_sha` comparison
  in the analysis output).
- **Same Ollama daemon and model build** as EXP-002 (no quant change,
  no rebuild). If the Ollama daemon has been restarted/upgraded since
  EXP-002 ran, log it; do not abort.
- **Outer-loop-by-seed** sweep ordering (only one model, so no
  model-warmup variance).
- **No retries on errored cells.** Errors count as failures of the
  decision-rule denominator, not silent skips.

## Out of scope

- Sweeping over reasoning-mode strengths (low/medium/high). Ollama's
  `think: true` is the default-on toggle; we test default-on vs
  default-off and no finer granularity.
- Other locals. llama3.1-8b-q4 has no reasoning mode; gpt-oss-20b
  reasoning is on by default and not toggleable via this interface.
- Per-task vs aggregate verdict. The aggregate is the pre-registered
  rule; per-task breakdowns are reported but not in the decision.

## Reproduction

```bash
lab exp register docs/exp/EXP-002b-qwen3-reasoning-on-ablation.md
lab sweep run conf/sweep/EXP-002b.yaml --enforce-pre-registration
uv run python scripts/analyze_exp002b.py
```

## Expected output artifacts

- `analysis/EXP-002b/SUMMARY.md` — top-line H1 verdict + 1-line
  comparison vs EXP-002 baseline
- `analysis/EXP-002b/verdicts.md` — full decision-rule application
- `analysis/EXP-002b/per_task_endstate.csv` — per-task pass@1 / pass⁸
  for think:true and the matched think:false rows from EXP-002
- `analysis/EXP-002b/per_cell.csv` — per-cell scorer breakdown
- F-008 (or next free F-NNN) — finding doc linking back to F-005
