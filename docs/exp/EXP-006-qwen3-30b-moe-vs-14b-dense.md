---
doc_id: exp-006
title: 'EXP-006: Qwen3-30B-A3B (MoE) vs qwen3-14b-q4 (dense) on PBS-Agent v0.1 —
  does local MoE close the local-vs-cloud end_state gap?'
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
- moe
- qwen3
- 30b
- agent
- local-first
---

# EXP-006: Qwen3-30B-A3B (MoE) vs qwen3-14b-q4 (dense) on PBS-Agent v0.1

Date created: 2026-05-27
Status: planned
Pre-registered: (commit SHA filled by `lab exp register` at registration time)
Parent findings: [F-005](../findings/F-005-12gb-agent-v0.2-tool-use.md) (baseline at
`qwen3-14b-q4 / think:false`, `end_state = 0.750`).
Plan reference: `~/docs/plans/2026-05-27-lab-phase19-model-orchestration.md` §19d.

## Question

Does the new Phase 19a headline local model **Qwen3-30B-A3B** (MoE, 3B
active per token, GPU-attention + CPU-experts hybrid offload via the
Phase 19d CUDA llama.cpp build) close the **local-vs-cloud `end_state`
gap** that F-005 surfaced on PBS-Agent v0.1, while remaining fully
local and free?

F-005 left the lab with this picture: locals (qwen3-14b-q4 reasoning-OFF)
match cloud at `tool_correctness` (1.000) but lag cloud on `end_state`
(0.750 local vs 0.833 cloud — an ~8pp gap, with a 25pp tool→end-state
conversion penalty for locals). The hypothesis behind Phase 19a was that
**this gap is a planner-capacity gap, not a tool-wiring gap**, and that
a larger local model with the same tool-call quality should close it
substantially. EXP-006 is the falsifying test of that hypothesis.

## Hypothesis

Four pre-registered hypotheses, all evaluated at greedy decoding
(`temperature=0.0`, `top_p=1.0`, `max_tokens=1024`), N=8 seeds per cell,
on PBS-Agent v0.1 (the same 12 hand-curated tool-use tasks F-005 used:
3 fs + 3 code + 2 shell + 2 http + 2 multi-domain). Each cell runs
inside a Podman + gVisor sandbox.

- **H1 — Replication.** Mean `end_state` (over the 12 PBS-Agent v0.1
  tasks × 8 seeds) for `qwen3-14b-q4` reasoning-OFF lands within **±0.05
  pp** of F-005's measured baseline of **0.750**. This is a sanity
  check: the lab stack, the sandbox, the scorers and the task suite
  must reproduce the prior result before any claim is made about the
  new model.

- **H2 — Headline lift.** Mean `end_state` for `qwen3-30b-a3b-moe` is
  **≥ 0.850** — at least **+10pp** over the H1 anchor at 0.750. This
  is the bet that the MoE planner is materially better than the dense
  14B planner at the same VRAM budget.

- **H3 — Gap closure.** The fraction of the local-vs-cloud `end_state`
  gap that the new local model closes — measured relative to the same
  cloud ceiling — is **≥ 0.50**. Operational definition:

  ```text
  gap_closure = (end_state(qwen3-30b-a3b-moe) - end_state(qwen3-14b-q4))
              / (end_state(gpt-oss-120b-cloud)  - end_state(qwen3-14b-q4))
  ```

  This is the load-bearing decision rule. H2 says "the new local model
  is materially better than the baseline local model"; H3 says "in
  particular, it is good enough to plausibly replace the cloud
  reference model on this suite". A `gap_closure ≥ 0.50` says the new
  local closed at least half the distance to cloud; the lab-default
  routing recommendation can plausibly flip to local on H3 pass.

- **H4 — Tool-correctness ceiling.** Mean `tool_correctness` for
  `qwen3-30b-a3b-moe` (over all 12 tasks × 8 seeds) is **≥ 0.95**. F-005
  measured cloud at 0.965 and qwen3-14b-q4 at 1.000; the new model must
  not regress here for the gap-closure verdict to be load-bearing.
  A qwen3-30b-a3b-moe `tool_correctness < 0.95` would falsify the
  "MoE has the same tool wiring as the dense baseline" assumption that
  Phase 19a built on, and is grounds to investigate before any
  end-state numbers are quoted.

These hypotheses are independent; each is judged on its own evidence.
We pre-commit to reporting all four verdicts in F-009 regardless of
which way they fall — no peeking, no re-framing.

## Why this matters

F-005 said: "the binding constraint on local models is `end_state`,
not `tool_correctness`." Phase 19a's investment thesis was that this
ceiling is *liftable* with a bigger local planner. EXP-006 closes that
loop:

1. *Pass (H2 + H3 both):* the lab's default local model can be
   promoted from `qwen3-14b-q4` to `qwen3-30b-a3b-moe`; cloud becomes a
   ceiling reference, not a default. Phase 19a's hardware bet (12 GB
   GPU + CPU experts via `-ot exps=CPU`) pays out.
2. *Mixed (H2 pass, H3 fail):* the bigger local helps but doesn't
   close cloud; investigate whether the gap is task-class-specific
   (e.g. http, shell — F-005 surfaced both as zero categories for
   locals).
3. *Fail (H2 fail):* the MoE planner buys nothing measurable on
   PBS-Agent v0.1 at this VRAM tier. Phase 19a's local headline thesis
   is refuted on this suite; cloud routing stands.
4. *H1 fail:* sweep is invalid (lab plumbing drifted between EXP-002
   and EXP-006). Stop and root-cause before any new claims are made.

## Method

### Models (3, in this order)

| litellm_id            | Backend                           | VRAM / tier      | Notes |
|-----------------------|-----------------------------------|------------------|-------|
| `qwen3-14b-q4`        | local Ollama                      | ~9.3 GB          | reasoning **disabled** via API-level `think: false` per F-005 / EXP-002 amendment. This is the replication arm (H1). |
| `qwen3-30b-a3b-moe`   | local llama.cpp (Phase 19d build) via llama-swap | ~5.6 GB peak, hybrid: `-ngl 99 -ot exps=CPU` | This is the treatment arm (H2, H3, H4). Routed through llama-swap on port 8080 per Phase 19b. |
| `gpt-oss-120b-cloud`  | Ollama Cloud Pro                  | cloud            | Ceiling reference for H3. F-005 measured `end_state = 0.833`. |

Outer-loop-by-model so each model warms once per sweep; the per-cell
`model_pool.declare(plan)` (Phase 19c) handles pre-flight + eviction
for the llama-swap-served `qwen3-30b-a3b-moe`.

### Config (1)

```yaml
name: greedy-1024
temperature: 0.0
top_p: 1.0
max_tokens: 1024
scaffold: single_turn  # runner dispatches to agent path on max_turns>1
```

Identical to EXP-002 / F-005 by design — the H1 replication check
requires the configuration to match.

### Per-model overrides

```yaml
model_defaults:
  qwen3-14b-q4:
    extra:
      think: false        # match F-005 baseline (reasoning-OFF)
```

`qwen3-30b-a3b-moe` and `gpt-oss-120b-cloud` use default settings —
qwen3-30b-a3b-moe is a reasoning-mode-on-by-default model in its
default chat template, and Phase 19a's llama-swap config already
exposes it with `-ngl 99 -ot exps=CPU`; no per-call knobs are pulled
in this sweep. Reasoning-on/off ablation on the MoE arm is explicitly
deferred to a follow-up (see "Out of scope").

### Tasks (12)

The full PBS-Agent v0.1 suite (`tasks/pbs-agent-v0.1/`, suite name
`pbs-agent-v0.1`), identical to EXP-002 / F-005:

| Category | Count | Slugs |
|----------|-------|-------|
| fs       | 3     | `fs-read-and-copy`, `fs-grep-extract-and-write`, `fs-write-csv-summary` |
| code     | 3     | `code-read-and-explain`, `code-write-and-execute`, `code-find-and-fix-bug` |
| shell    | 2     | `shell-count-lines`, `shell-pipeline-extract` |
| http     | 2     | `http-fetch-and-extract`, `http-fetch-and-count` |
| multi    | 2     | `multi-words-and-hash`, `multi-db-self-check` |

Known plumbing constraints from F-005 carry forward:

- **http fixture loading** still requires the `LAB_HTTP_FIXTURE_DIR`
  wiring (F-005 Surprise 1). If still broken, the 2 http tasks remain
  uniformly 0 across all 3 models — the verdict computations are
  resilient to this because all three models are penalised identically;
  the gap-closure ratio is well-defined as long as the cloud reference
  also fails uniformly. We **report `gap_closure` both with and without
  http cells** for transparency; the pre-registered headline is the
  with-http version (matches F-005's denominator).
- **`code-find-and-fix-bug`** still has the trivially-passable success
  predicate (F-005 Surprise 2). Same handling: locals and cloud both
  benefit from it equally; the gap-closure ratio is unaffected at
  first order.

### Seeds (8)

`[1, 2, 3, 4, 5, 6, 7, 8]` per `docs/protocols/reliability-sweep.md`.

### Total cells

12 tasks × 3 models × 1 config × 8 seeds = **288 runs**.

### Evaluators (pre-registered)

Deterministic, applied to every done run (same set F-005 used):

- `end_state` — task `success_predicate` over post-run `/workspace`
  snapshot and/or DB state. Anchor metric for H1, H2, H3.
- `tool_correctness` — model called `target_tool` declared in
  `task.rubric.tool_call`. Anchor metric for H4.
- `budget_respected` — `actual_turns ≤ max_turns AND tool_call_count ≤
  tool_budget` (operational signal, reported but not in any decision
  rule).

LLM-judge (cloud-budgeted, selective):

- `trajectory_judge` — applied **only** to tasks with
  `success_predicate.include_judge: true`. In v0.1 this is the single
  task `code-read-and-explain` (judge model `gpt-oss-120b-cloud` as set
  on the task). 1 task × 3 models × 8 seeds = **24 judge calls**.

### Statistics

Per protocol `docs/protocols/reliability-sweep.md`:

- `pass@1`, `pass^4`, `pass^8` per (model, task) cell on `end_state`
  and `tool_correctness`.
- Bootstrap 95 % CI on model-wide `end_state` and `tool_correctness`
  means (n_resamples=2000).
- Per-model `gap_closure` defined as
  `(local_new - local_old) / (cloud_ref - local_old)` for each cell
  category and overall.
- Paired-by-task differences (`local_new - local_old`) with a
  one-sided permutation test (1000 perms) as a secondary signal —
  not in the decision rule, reported for context.

## Success / failure criteria

Each hypothesis is judged by the pre-registered rule below, applied
AFTER the sweep + scoring complete. No peeking.

- **H1 confirmed** ⇔
  `|mean(end_state | qwen3-14b-q4, all 96 cells) − 0.750| ≤ 0.05`.
  H1 is reported as **REFUTED — sweep invalid** if outside the band,
  and the H2/H3/H4 verdicts are recorded as **INVALID — H1 replication
  failed**. The 0.05 pp band matches the F-005 bootstrap CI half-width
  for qwen3-14b-q4 end_state (CI reported in F-005 § H2).

- **H2 confirmed** ⇔
  `mean(end_state | qwen3-30b-a3b-moe, all 96 cells) ≥ 0.850`.
  Anchor: F-005's qwen3-14b-q4 baseline at 0.750. The threshold is a
  **+10pp absolute lift** on the same anchor.

- **H3 confirmed** ⇔
  `gap_closure ≥ 0.50`, where
  `gap_closure = (end_state(qwen3-30b-a3b-moe) − end_state(qwen3-14b-q4))
              / (end_state(gpt-oss-120b-cloud) − end_state(qwen3-14b-q4))`,
  all three terms computed on the same 96-cell denominator (12 tasks
  × 8 seeds) within this sweep. Denominator must be `> 0`; if cloud
  ≤ local-old on this sweep (i.e. the cloud reference itself
  regresses), H3 is reported as **UNDEFINED — gap denominator
  non-positive**.

- **H4 confirmed** ⇔
  `mean(tool_correctness | qwen3-30b-a3b-moe, all 96 cells) ≥ 0.95`.
  F-005 anchor: cloud was 0.965, qwen3-14b-q4 was 1.000.

Any failure modes (>5 % cells errored, sandbox failure rate >10 %,
judge agreement implausible) are escalated in F-009 rather than swept
under the rug.

## Kill criteria

The sweep aborts and writes an "INVALID — sweep killed" verdict block
if any of the following fire during the run:

- Cell error rate exceeds **5 %** (> 14 errored cells / 288).
- Sandbox failure rate exceeds **10 %** (runner-level signal, distinct
  from cell errors).
- GPU lease contention causes > 3 cells to fail with
  `gpu_lease_timeout` (sibling agents may be using the GPU).
- llama-swap stops responding to `/running` for ≥ 2 minutes mid-sweep
  (Phase 19b operational signal; without it the `qwen3-30b-a3b-moe`
  arm cannot make progress).
- The Ollama daemon crashes (kills the qwen3-14b-q4 and
  gpt-oss-120b-cloud arms simultaneously).

If kill-criteria fire, the analysis script is still run on whatever
cells completed, but every hypothesis verdict is reported as
`INVALID — sweep killed` rather than CONFIRMED / MIXED / REFUTED.

## Confounders to control

- **Identical sandbox image** to EXP-002 where feasible
  (`manifest_sha` comparison in the analysis output). New images are
  allowed (the Containerfile lineage at `5c364cc` is still the
  source), but the analyzer logs the per-cell hash distribution and
  flags drift.
- **Same Ollama daemon and model build** as EXP-002 for the qwen3-14b
  arm and the cloud arm. The qwen3-30b-a3b-moe arm is necessarily a
  new code path (llama.cpp CUDA build + llama-swap, both Phase 19a/b/c/d).
- **Outer-loop-by-model** so each model warms once. The model_pool
  predictive-load (Phase 19c) handles the cold-cell cost.
- **Greedy decoding** to compress seed variance — matches EXP-002 /
  F-005 by design. F-005's reliability ratio == 1.000 observation
  (H3-old) carries forward as an assumption that we're measuring
  capability, not seed luck.
- **N=8 seeds** matches EXP-002 / F-005. We're not chasing the
  reliability cliff here — that's the EXP-003 lineage — so N=8 with
  greedy decoding suffices for the four decision rules in this
  experiment.
- **Sandbox image hash drift** mid-sweep: F-005 surfaced that the
  guard fires only at launch. We accept that limitation for EXP-006
  rather than block on a guard hardening; the analyzer logs the
  distribution.

## Out of scope

- **Reasoning-mode ablation on `qwen3-30b-a3b-moe`.** F-004 / EXP-002b
  established that reasoning-OFF is the right default for the dense
  14B; the analogous question for the MoE is its own EXP-006c
  follow-up if H2 passes. Here we use the MoE's stock default chat
  template and report the result.
- **Other Phase 19a locals** (gpt-oss-20b-local, phi-4-reasoning-14b,
  hermes-4.3-36b). Each gets its own focused experiment if and when
  there's a hypothesis worth testing. EXP-006 is specifically the
  MoE-vs-dense bake-off.
- **70B quality-ceiling reference** (`llama-3.3-70b-q4-local`,
  Phase 19e). The plan calls EXP-006b a follow-on that adds the 70B
  arm at N=4. Not part of this 288-cell sweep.
- **Temperature > 0.0**, **alternate seeds**, **alternate scaffolds**.
  Same scope discipline as EXP-002.
- **Cost / latency comparison.** F-005 H4 already settled the 20B-vs-120B
  cloud cost/latency frame; EXP-006 doesn't pull on that.

## Reproduction

```bash
cd /data/lab/code

# 1. Register the plan
uv run lab exp register docs/exp/EXP-006-qwen3-30b-moe-vs-14b-dense.md

# 2. Sweep (~2 hr wall, 288 cells)
uv run lab sweep run conf/sweep/EXP-006.yaml --enforce-pre-registration

# 3. Deterministic + judge evaluators (judge runs only on include_judge tasks)
uv run lab eval apply EXP-006

# 4. Verdicts + analysis CSVs
uv run python scripts/analyze_exp006.py
```

## Expected output artifacts

- `analysis/EXP-006/SUMMARY.md` — top-line H1/H2/H3/H4 verdicts +
  1-line headline.
- `analysis/EXP-006/verdicts.md` — full decision-rule application,
  per-hypothesis with point estimates and bootstrap CIs.
- `analysis/EXP-006/per_task_endstate.csv` — per-task means for all
  three models.
- `analysis/EXP-006/per_cell.csv` — per-cell scorer breakdown.
- `analysis/EXP-006/gap_closure.csv` — `gap_closure` by category and
  overall.
- `docs/findings/F-009-qwen3-30b-moe-vs-14b-dense.md` — finding doc
  linking back to F-005.

## Pre-mortem

If by EOD 2026-05-27 this experiment has failed badly, plausible
causes:

- **Risk: llama-swap eviction misfires during outer-loop-by-model
  transitions.** The qwen3-30b-a3b-moe model is loaded once per cell
  (well, once per stretch of consecutive qwen3-30b-a3b-moe cells);
  llama-swap should evict any prior model first. *Mitigation:* Phase 19c
  `model_pool` already declares the per-cell plan and asks llama-swap
  for an explicit unload between models. If it doesn't fire cleanly,
  the analyzer surfaces the cold-cell distribution and we re-run with
  a longer per-cell timeout.
- **Risk: `-ot exps=CPU` is slow enough that the sweep blows the wall
  budget.** Phase 19a smoke showed ~5.6 GB peak VRAM and "smoke clean"
  but didn't measure throughput end-to-end on 12 tasks × 8 seeds.
  *Mitigation:* the 600 s request timeout cushions long cells; if a
  task hits the timeout repeatedly the sweep keeps going and the
  errored cells count toward kill criteria.
- **Risk: H1 fails (lab plumbing drift since EXP-002).** Image rebuild
  changed the sandbox; the tool surface drifted; the http fixture
  silently regressed further. *Mitigation:* the analyzer reports H1
  first and STOPS if outside the band — no "but H2 looks good" creep.
- **Risk: H4 fails (MoE tool wiring is worse than dense).** Some MoE
  models in the wild emit malformed tool-call JSON under llama.cpp's
  `--jinja` template. *Mitigation:* the smoke pass before the sweep
  catches "0 tool calls fired across smoke cells"; if smoke is fine
  but H4 fails on the real sweep, that's a real finding (write it up,
  don't paper over it).
- **Risk: Cloud reference (`gpt-oss-120b-cloud`) regresses or
  rate-limits.** Ollama Cloud Pro tier has caps; F-005 burnt ~120
  cloud cells. *Mitigation:* the runner's per-cell timeout is 600 s,
  the cloud arm sweeps last (outer-loop-by-model with cloud at the
  bottom), and the analyzer logs any non-done cells in the cloud arm
  separately so an "H3 undefined — denominator non-positive" verdict
  is computed correctly.
