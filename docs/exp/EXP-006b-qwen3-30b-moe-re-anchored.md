---
doc_id: exp-006b
title: 'EXP-006b: Qwen3-30B-A3B (MoE) vs qwen3-14b-q4 (dense) on PBS-Agent v0.1 —
  re-anchored after F-009 follow-up fixes'
zone: lab
kind: exp
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
depends_on:
- kind: doc
  target: exp-006
- kind: doc
  target: f-009-qwen3-30b-moe-refuted-h1-invalid
- kind: doc
  target: f-005-12gb-agent-v0-2-tool-use
tags:
- lab
- exp
- moe
- qwen3
- 30b
- agent
- local-first
- re-anchored
---

# EXP-006b: Qwen3-30B-A3B (MoE) vs qwen3-14b-q4 (dense) — re-anchored

Date created: 2026-05-27
Status: planned
Pre-registered: (commit SHA filled by `lab exp register` at registration time)
Parent findings: [F-009](../findings/F-009-qwen3-30b-moe-refuted-H1-invalid.md)
(EXP-006 — sweep INVALID via the H1-fail path);
[F-005](../findings/F-005-12gb-agent-v0.2-tool-use.md) (original anchor at
`qwen3-14b-q4 / think:false`, `end_state = 0.750` — now superseded).

## Question

After the F-009 follow-up fixes landed (MoE chat-template gap closed, two
trivially-passable success predicates tightened, multi-words-and-hash
regression repaired, token passthrough wired, LiteLLM cold-load 502 retry
policy added), the lab measurement surface has moved materially in both
directions. The original EXP-006 H1 anchor of 0.750 is no longer the right
number to replicate against — F-009 explicitly identified it as having been
load-bearing on the broken http fixture and the trivially-passable
`code-find-and-fix-bug` predicate.

**EXP-006b re-poses the EXP-006 question against the post-fix surface:**
does **Qwen3-30B-A3B** (MoE, served by llama.cpp via llama-swap with the
template fix at `1141dc1`) close the **local-vs-cloud `end_state` gap** on
PBS-Agent v0.1, now that the MoE arm's tool-emission gap and the dense
arm's two artefactual lift sources have both been fixed?

This experiment makes **no replication claim**. H1 is reframed from a
pass/fail anchor into a **baseline measurement**: the EXP-006b dense
end_state number — with its bootstrap CI — becomes the new lab reference
for the post-fix surface. EXP-006's H2/H3/H4 thresholds are reused (the
underlying decision the lab needs to make has not changed), but H2 is
strengthened from an absolute threshold into a **relative-delta-vs-the-new-
baseline** rule so the experiment cannot be silently anchored on a moving
floor.

## Background: F-009 follow-up commits

These five commits landed between EXP-006's sweep (F-009) and EXP-006b's
pre-registration. Each is a known, named change to the measurement
surface; the re-anchor is necessary because of their cumulative effect.

| commit | one-line |
|---|---|
| `cceaf62` + `c345ab6` | `multi-db-self-check` success predicate tightened. The original predicate was satisfiable by sandbox-init state (F-009's "second F-005 Surprise 2"); now requires a `mean.txt` containing `"6.0"`, which forces the model to actually compute and write. |
| `f48c517` + `8503f5c` | `multi-words-and-hash` prompt fix. Hashing prompt was task-global; now task-local. Expected to recover ~8.3pp of the F-009 dense regression on this task. |
| `1141dc1` | MoE tool-emission gap closed via `--chat-template-kwargs enable_thinking=false`. F-009's H4 mechanism was a reasoning-budget overrun preventing tool emission on 40/96 MoE cells; this flag turns off the MoE's default reasoning prelude so the tool call fires within the budget. |
| `971eb38` | Token-count passthrough fixed. `experiment_runs.tokens_in / tokens_out` now populate for both ollama and llama-swap+llama.cpp routes through LiteLLM. F-009 reported these were NULL for all 288 cells. |
| `b0b0e96` | LiteLLM cold-load 502 retry policy added. Smooths over the first-call cold-load failures that periodically tripped big-model lanes; not expected to move the EXP-006b verdicts but does reduce the cell-error blast radius if anything misfires. |

A sixth commit landed in the same period — `67853f8` (ceiling-llm
eviction wrapper for the 70B lane) — but is not relevant for EXP-006b:
no 70B arm in this sweep.

## Hypothesis

Four pre-registered hypotheses follow. H1 is a baseline measurement (no
gate); H2, H3, and H4 are promotion gates with explicit pass/fail
thresholds.

### Pre-registered hypotheses

Four pre-registered hypotheses, all evaluated at greedy decoding
(`temperature=0.0`, `top_p=1.0`, `max_tokens=1024`), N=8 seeds per cell,
on PBS-Agent v0.1 — the same 12 hand-curated tool-use tasks F-005 and
EXP-006 used (3 fs + 3 code + 2 shell + 2 http + 2 multi-domain). Each
cell runs inside a Podman + gVisor sandbox. Outer-loop-by-model so each
model warms once per sweep.

### H1 — Baseline measurement (NOT a promotion gate)

Report `qwen3-14b-q4.end_state` (over all 96 dense cells) with a 95 %
bootstrap CI. This **establishes** the new lab baseline for the post-fix
PBS-Agent v0.1 surface. The number replaces F-005's 0.750 anchor for
future sweeps that need a local-dense reference. **There is no pass/fail
threshold on H1 in EXP-006b** — the number is what it is. The pre-reg
commits to reporting it before the H2/H3 verdicts so it cannot be
back-fit.

H1 *does* feed the H2 rule (see below), but the H1 number itself is
not gated.

### H2 — Headline (RELATIVE DELTA — promotion gate)

The MoE arm's lower 95 % CI bound on `end_state` exceeds the dense
arm's point estimate by ≥ **+0.10**:

```text
lower_95_CI(qwen3-30b-a3b-moe.end_state)
    ≥ point_estimate(qwen3-14b-q4.end_state) + 0.10
```

In words: MoE wins if it's at least 10 pp better than dense **after
accounting for the uncertainty in the MoE measurement**. Using the
lower CI bound on the MoE side (and the point estimate on the dense
side) is deliberately conservative — it requires the MoE arm's
*worst plausible* mean to still beat the dense *best estimate* by
10 pp. This is the rule that makes "MoE materially better than dense"
load-bearing rather than noise-driven.

### H3 — Gap closure (RATIO — promotion gate)

```text
gap_closure_pe := (point_estimate(MoE.end_state) − point_estimate(dense.end_state))
                / (point_estimate(cloud.end_state) − point_estimate(dense.end_state))

H3 confirmed ⇔ gap_closure_pe ≥ 0.50.
```

H3 uses point estimates on all three terms (the F-005 / EXP-006 H3
formulation; not changed). The denominator must be positive; if
`cloud ≤ dense`, H3 is reported as **UNDEFINED — gap denominator
non-positive**, and the gate **fails** for promotion purposes (UNDEFINED
≠ pass).

### H4 — Tool-correctness ceiling (promotion gate, slightly relaxed)

```text
lower_95_CI(qwen3-30b-a3b-moe.tool_correctness) ≥ 0.90
```

Relaxed from EXP-006's 0.95 to **0.90**. The MoE template fix
(`1141dc1`) may not perfectly close every tool-emission case; 0.90 is
still cloud-tier (F-005 measured cloud at 0.965 across five models,
none below 0.90). At a 96-cell binary metric, a lower-CI threshold of
0.90 corresponds roughly to a point estimate of ≥ 0.94 (the bootstrap
half-width is ~0.04 at this n on a near-ceiling proportion), so the
relaxation is real but small.

Like H2, the lower CI bound is used to guard against noise-driven
"wins" on a metric where ceiling effects compress variance.

These four hypotheses are independent; each is judged on its own
evidence. We pre-commit to reporting all four verdicts in F-010
regardless of which way they fall.

## Decision rule

**MoE promotes to `lab-default-local` if and only if H2 AND H3 both
pass.** H4 must additionally pass for the promotion to be quality-clean;
if H4 fails but H2 + H3 pass, the promotion still goes through but F-010
records H4 as an explicit quality caveat that follow-up work (template
audit, prompt-engineering on tool-call exemplars) must close before the
MoE arm is treated as fully trusted on tool-use workloads.

H1 is not gated and does not enter the promotion rule. It exists only
to establish the new lab baseline number — a measurement, not a
decision.

## Why this matters

F-009 left the lab here: the EXP-006 sweep ran cleanly (288/288, 0 errors)
but H1 fired the pre-reg's "INVALID" path, and the MoE arm's H4 surfaced
a real tool-emission gap on the MoE serving path. Five fixes have landed
since. The decision EXP-006 was supposed to make — promote MoE or stay
on dense — is still open and load-bearing for the lab's local-first
thesis (Phase 19a's investment in 12 GB-tier MoE inference).

The H2 reformulation (relative delta against the *measured* dense
baseline rather than an absolute threshold against a *legacy* anchor)
makes EXP-006b robust against future surface drift: even if the dense
baseline moves again, H2 still says "MoE must be 10 pp better than
*today's* dense, with uncertainty accounted for". That is the load-
bearing rule.

## Method

### Models (3, in this order)

| litellm_id            | Backend                                                | VRAM / tier      | Notes |
|-----------------------|--------------------------------------------------------|------------------|-------|
| `qwen3-14b-q4`        | local Ollama                                           | ~9.3 GB          | reasoning **disabled** via API-level `think: false` per F-005 / EXP-002 amendment. This is the H1 baseline arm. `keep_alive: 0` carried over from EXP-006 (non-semantic VRAM-residency knob; see EXP-006 § "Per-model overrides" for rationale). |
| `qwen3-30b-a3b-moe`   | local llama.cpp (Phase 19d build) via llama-swap       | ~5.6 GB peak, hybrid: `-ngl 99 -ot exps=CPU` | Treatment arm (H2, H3, H4). Routed through llama-swap on port 8080 per Phase 19b. **MoE template fix `1141dc1` applied** (`enable_thinking=false` via `--chat-template-kwargs` in `conf/llama-swap.yaml`). |
| `gpt-oss-120b-cloud`  | Ollama Cloud Pro                                       | cloud            | Ceiling reference for H3. F-009 measured at 0.969 end_state on this suite. |

### Config (1)

```yaml
name: greedy-1024
temperature: 0.0
top_p: 1.0
max_tokens: 1024
scaffold: single_turn  # runner dispatches to agent path on max_turns>1
```

Identical to EXP-002 / EXP-006 / F-005 by design.

### Per-model overrides

```yaml
model_defaults:
  qwen3-14b-q4:
    extra:
      think: false        # match F-005 / EXP-006 baseline (reasoning-OFF)
      keep_alive: 0       # plumbing — see EXP-006 § Per-model overrides
```

`qwen3-30b-a3b-moe` and `gpt-oss-120b-cloud` use default settings.
**The MoE template fix is applied at the llama-swap layer
(`conf/llama-swap.yaml`)**, not via per-call extras — the
`--chat-template-kwargs enable_thinking=false` flag is part of the
model's serve command and applies to every request the model sees.

### Tasks (12)

The full PBS-Agent v0.1 suite (`tasks/pbs-agent-v0.1/`, suite name
`pbs-agent-v0.1`), identical to EXP-006 / EXP-002:

| Category | Count | Slugs |
|----------|-------|-------|
| fs       | 3     | `fs-read-and-copy`, `fs-grep-extract-and-write`, `fs-write-csv-summary` |
| code     | 3     | `code-read-and-explain`, `code-write-and-execute`, `code-find-and-fix-bug` |
| shell    | 2     | `shell-count-lines`, `shell-pipeline-extract` |
| http     | 2     | `http-fetch-and-extract`, `http-fetch-and-count` |
| multi    | 2     | `multi-words-and-hash`, `multi-db-self-check` |

The two trivially-passable predicates flagged in F-005 / F-009 have
been tightened in the F-009 follow-ups:

- `code-find-and-fix-bug` predicate tightening landed pre-EXP-006 (per
  F-009 commentary).
- `multi-db-self-check` predicate tightening landed at `cceaf62` +
  `c345ab6` — now requires `mean.txt` containing `"6.0"`.

Both predicates now require model action; neither is satisfiable by
sandbox-init state.

The `multi-words-and-hash` prompt fix landed at `f48c517` + `8503f5c`
— hashing is now task-local rather than task-global, recovering the
dense regression F-009 reported (qwen3-14b-q4: 1.000 → 0.000 → expected
back to ~1.000).

### Seeds (8)

`[1, 2, 3, 4, 5, 6, 7, 8]` per `docs/protocols/reliability-sweep.md`.

### Total cells

12 tasks × 3 models × 1 config × 8 seeds = **288 runs.**

### Evaluators

Same set as EXP-006 (deterministic, applied to every done run):

- `end_state` — anchor metric for H1 (baseline), H2 (headline), H3 (gap).
- `tool_correctness` — anchor metric for H4.
- `budget_respected` — operational signal, reported but not in any
  decision rule.

LLM-judge (cloud-budgeted, selective):

- `trajectory_judge` — applied **only** to tasks with
  `success_predicate.include_judge: true`. In v0.1 this remains the
  single task `code-read-and-explain`. 1 task × 3 models × 8 seeds =
  **24 judge calls**.

### Statistics

Per protocol `docs/protocols/reliability-sweep.md`:

- `pass@1`, `pass^4`, `pass^8` per (model, task) cell on `end_state`
  and `tool_correctness`.
- **Bootstrap 95 % CI** on model-wide `end_state` and `tool_correctness`
  means, `n_resamples = 2000`, percentile method. Choice rationale:
  bootstrap is non-parametric (no normality assumption on a binary
  near-ceiling metric); 2000 resamples matches EXP-006's analyzer.
- Per-model `gap_closure_pe` defined as
  `(moe_pe − dense_pe) / (cloud_pe − dense_pe)` per H3.
- Paired-by-task differences (`moe − dense`) with a one-sided
  permutation test (1000 perms) as a secondary signal — not in the
  decision rule, reported for context.
- **New for EXP-006b** vs EXP-006: report tokens_in / tokens_out
  summaries per model (now that `971eb38` populates these columns).

## Success / failure criteria

Each hypothesis is judged by the pre-registered rule below, applied
AFTER the sweep + scoring complete. No peeking.

- **H1.** Report `mean(end_state | qwen3-14b-q4, n=96)` with 95 %
  bootstrap CI. No pass/fail. The number is recorded as the new lab
  reference.

- **H2 confirmed** ⇔
  `lower_95_CI(end_state | qwen3-30b-a3b-moe, n=96) ≥ mean(end_state | qwen3-14b-q4, n=96) + 0.10`.
  Otherwise REFUTED.

- **H3 confirmed** ⇔
  `gap_closure_pe ≥ 0.50`, where
  `gap_closure_pe = (mean(MoE) − mean(dense)) / (mean(cloud) − mean(dense))`,
  all three on the same 96-cell denominator (12 tasks × 8 seeds). If
  `mean(cloud) ≤ mean(dense)`, H3 is **UNDEFINED — denominator
  non-positive** (and the H3 gate does not pass for promotion).

- **H4 confirmed** ⇔
  `lower_95_CI(tool_correctness | qwen3-30b-a3b-moe, n=96) ≥ 0.90`.
  Otherwise REFUTED.

- **Promotion decision rule.** Promote MoE to `lab-default-local` if
  H2 AND H3 both pass. H4 must also pass; if H4 fails but H2 + H3 pass,
  promote and record H4 as a quality caveat in F-010 (follow-up: MoE
  template audit). If H2 OR H3 fails, do not promote.

## Kill criteria

The sweep aborts and writes an "INVALID — sweep killed" verdict block
if any of the following fire during the run:

- Cell error rate exceeds **5 %** (> 14 errored cells / 288).
- Sandbox failure rate exceeds **10 %** (runner-level signal).
- GPU lease contention causes > 3 cells to fail with `gpu_lease_timeout`.
- llama-swap stops responding to `/running` for ≥ 2 minutes mid-sweep.
- The Ollama daemon crashes.

If kill-criteria fire, the analysis script is still run on whatever
cells completed, but every hypothesis verdict is reported as
`INVALID — sweep killed`.

## Confounders to control

- **Identical sandbox image lineage** to EXP-006 (Containerfile at
  `5c364cc`). Analyzer logs the per-cell hash distribution and flags
  drift. F-009 reported zero drift on EXP-006; no reason to expect
  drift here either.
- **Same Ollama daemon and model build** as EXP-002 / EXP-006 for the
  qwen3-14b-q4 and gpt-oss-120b-cloud arms. The qwen3-30b-a3b-moe arm
  uses the same llama.cpp + llama-swap Phase 19a/b/c/d stack with the
  template-fix commit (`1141dc1`) applied.
- **Outer-loop-by-model** so each model warms once. Phase 19c
  `model_pool` handles llama-swap pre-flight + eviction.
- **Greedy decoding** to compress seed variance — matches EXP-002 /
  EXP-006 / F-005 by design.
- **N=8 seeds** matches EXP-002 / EXP-006 / F-005.
- The five F-009 follow-up commits are the **deliberate** changes to
  the measurement surface; they are the *purpose* of EXP-006b. They
  are not "confounders" — they are the treatment that EXP-006b is
  re-measuring against.

## Out of scope

- **Reasoning-mode ablation on `qwen3-30b-a3b-moe`.** The template fix
  at `1141dc1` already disables the MoE's reasoning prelude as the
  default chat-template behaviour for tool-use tasks. A separate
  reasoning-ON ablation is its own EXP-006c follow-up if needed.
- **Other Phase 19a locals** (gpt-oss-20b-local, phi-4-reasoning-14b,
  hermes-4.3-36b).
- **70B quality-ceiling reference** (`llama-3.3-70b-q4-local`,
  Phase 19e). Not part of this 288-cell sweep. EXP-006b uses the
  `--allow-slow-models` flag *not at all*; the three models in this
  sweep are all not-slow-tagged.
- **Temperature > 0.0**, **alternate seeds**, **alternate scaffolds**.
  Same scope discipline as EXP-002 / EXP-006.
- **Cost / latency comparison.** F-005 H4 settled the 20B-vs-120B
  cloud cost/latency frame; EXP-006b doesn't re-pull on that. Latency
  and tokens_in / tokens_out are reported as operational notes only.

## Reproduction

```bash
cd /data/lab/code

# 1. Register the plan
uv run lab exp register docs/exp/EXP-006b-qwen3-30b-moe-re-anchored.md

# 2. Sweep (~2 hr wall, 288 cells)
uv run lab sweep run conf/sweep/EXP-006b.yaml --enforce-pre-registration

# 3. Deterministic + judge evaluators (judge runs only on include_judge tasks)
uv run lab eval apply EXP-006b

# 4. Verdicts + analysis CSVs
uv run python scripts/analyze_exp006b.py
```

## Expected output artifacts

- `analysis/EXP-006b/SUMMARY.md` — top-line H1/H2/H3/H4 verdicts +
  1-line headline.
- `analysis/EXP-006b/verdicts.md` — full decision-rule application,
  per-hypothesis with point estimates and bootstrap CIs.
- `analysis/EXP-006b/per_task_endstate.csv` — per-task means for all
  three models.
- `analysis/EXP-006b/per_cell.csv` — per-cell scorer breakdown.
- `analysis/EXP-006b/gap_closure.csv` — `gap_closure` by category and
  overall.
- `docs/findings/F-010-*.md` — finding doc linking back to F-009 + F-005.

## Pre-mortem

Plausible failure modes for EXP-006b and their cheap mitigations:

- **Risk: MoE template fix is partial.** `1141dc1` closes the reasoning-
  budget overrun, but the MoE may still under-call tools on some task
  shapes. *Mitigation:* H4 is the diagnostic; relaxed to 0.90 lower-CI
  to allow a partial fix to still be promotable on H2+H3 if those pass.
  Per F-010, follow-up work fixes the residual tool-emission gap.
- **Risk: multi-words-and-hash prompt fix is incomplete.** `f48c517`
  fixes the task-global → task-local hashing prompt, but a downstream
  scorer expectation may still be wrong. *Mitigation:* per-task diff in
  the F-010 comparison table will surface this; doesn't break the
  overall decision rule because H2 is a delta-vs-dense, not an
  absolute.
- **Risk: MoE per-cell latency blows the wall budget.** F-009 measured
  MoE at 67 s p50 / 100 s p95 vs cloud's 9 s. 96 cells × 100 s ≈ 2.7 h
  on the MoE arm alone. *Mitigation:* 600 s per-cell timeout cushion,
  outer-loop-by-model + model_pool pre-flight, kill criterion on
  errored cells > 5 %. If wall exceeds 3 h, that's an operational
  signal but not a kill criterion on its own.
- **Risk: gpt-oss-120b-cloud regresses or rate-limits.** *Mitigation:*
  cloud arm runs last (after both locals); analyzer logs any non-done
  cells; if cloud arm completes with high error rate, H3 may be
  UNDEFINED, in which case the promotion rule fails as designed (no
  silent promotion on a missing ceiling).
- **Risk: token passthrough fix (`971eb38`) doesn't actually populate
  the columns.** *Mitigation:* this doesn't gate any verdict; F-010
  records the empirical result either way. If tokens_in/out are still
  NULL, file a follow-on; the finding is still load-bearing on
  end_state / tool_correctness.
- **Risk: H2's "lower CI ≥ dense + 0.10" rule is too strict for the
  realistic effect size.** *Mitigation:* this is *the point* of the
  rule. If MoE's measured lift is 10 pp but the CI is wide enough that
  the lower bound is < +10 pp over dense, the promotion is *not*
  warranted on the available evidence. The lab can re-run at N=16 if
  the verdict ends up borderline; the pre-reg commits to N=8 for this
  run and reports the verdict as written.

## Components NOT run end-to-end in EXP-006b

(Filled in after the sweep — placeholder.)

If the sweep completes cleanly, this section enumerates anything in
the plan above that was deferred. If kill criteria fire, this section
enumerates components downstream of the kill point.
