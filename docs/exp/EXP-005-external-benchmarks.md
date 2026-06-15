---
doc_id: exp-005
title: 'EXP-005: External benchmark BFCL v3 — local vs cloud on the Berkeley Function
  Calling Leaderboard'
zone: lab
kind: exp
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
depends_on:
- kind: doc
  target: lab-master-roadmap-2026-05-26
- kind: doc
  target: f-010-qwen3-30b-moe-re-anchored-not-promoted-h2-h4-fail
tags:
- lab
- exp
- bfcl
- external-benchmark
- tool-use
- function-calling
- phase-17
---

# EXP-005: BFCL v3 external benchmark — local vs cloud on tool-use

Date created: 2026-05-27
Status: planned
Pre-registered: a32b449  (registered by `lab exp register` at file-creation time; backfilled 2026-06-14)
Parent docs: [lab master roadmap §17.5](/home/m/docs/plans/2026-05-26-lab-master-roadmap.md);
[F-010](../findings/F-010-qwen3-30b-moe-re-anchored-not-promoted-H2-H4-fail.md)
(established post-fix lab baselines on PBS-Agent v0.1: dense local
0.667 end_state, cloud reference 0.969).

## Question

How do the lab's registered local models compare to cloud models on the
**Berkeley Function Calling Leaderboard v3** (BFCL v3) tool-use
benchmark? Specifically: where does our local-default
(`qwen3-14b-q4`, reasoning-off) land on the published BFCL AST
categories (`simple`, `multiple`, `parallel`, `parallel_multiple`)
relative to cloud-class models on the same suite?

F-010 settled the **PBS-Agent v0.1** picture (a 12-task lab-internal
suite). EXP-005 asks the inverse: **on a community standard with
external comparators**, how do we score? The point is not to beat the
leaderboard — it is to anchor the lab's measurement surface against an
external one, so future findings can quote both a lab-internal
(PBS-Agent) and a community number.

## Adoption decision

**Option B (vendor a Python-only subset of BFCL's AST grader).** We
ship the four non-live AST categories (`simple`, `multiple`,
`parallel`, `parallel_multiple` — 1000 examples total) as a lab task
suite `bfcl-v3-ast`. Each example becomes one `Task` row with a
`bfcl_ast` rubric; a new sweep code path (`_execute_bfcl_cell`) issues
a single tool-calling LiteLLM request, captures the response's
`tool_calls`, and grades inline via the vendored AST checker.

The vendored checker is Python-only (skips Java / JavaScript / SQL /
REST / `live_*` / `multi_turn_*` categories). License: upstream Apache
2.0 (NOTICE preserved). Vendored on 2026-05-27 against HEAD-of-main of
`https://github.com/ShishirPatil/gorilla`.

### Why not Option A (Inspect's task registry)?

BFCL examples don't use our MCP/sandbox tool surface — the model is
asked to emit a function call against a *published* schema, not invoke
any of our tool servers. Going through Inspect's full task harness
would force a synthetic Sandbox + MCP tool wiring layer for code that
ends up doing a single LiteLLM `chat.completions` call. The dedicated
runner path keeps the surface load-bearing instead of decorative.

### Why not Option C (BFCL's own runner binary)?

Their runner is a process-spawning + result-aggregation pipeline that
expects to own the model-handler layer. Plumbing it into the lab's
LiteLLM proxy + sweep tables would require shimming both directions;
vendoring the AST checker against the on-disk format is strictly
less surface.

## τ²-bench: deferred (Phase 17.5b follow-up)

τ²-bench (`HuggingFaceH4/tau2-bench-data` + `sierra-research/tau2-bench`)
ships as a turnkey CLI that runs a **user-simulator subprocess** plus
a **stateful tool environment** alongside the agent. Plumbing this
into the lab's sweep harness requires:

1. Vendoring or installing the upstream `tau2` package
2. Wiring its user-simulator LLM to our LiteLLM proxy
3. Capturing per-domain DB-state-diff results into our `eval_results` table
4. Honouring the lab's per-cell idempotency invariant across the
   user-simulator round-trips

Each of those is its own integration project. Phase 17.5 ships BFCL v3
as the headline; τ²-bench is queued as a follow-on (track as
`Phase 17.5b — τ²-bench adapter` in the master roadmap).

## Setup

### Models

| litellm_id          | backend       | role                     | notes |
|---------------------|---------------|--------------------------|-------|
| `qwen3-14b-q4`      | ollama-local  | local-default            | reasoning **disabled** via `think: false` per F-005 / EXP-006 amendment |
| `gpt-oss-20b-cloud` | ollama-cloud  | cloud-small reference    | community-tier comparison |
| `glm-5.1-cloud`     | ollama-cloud  | cloud-medium reference   | |
| `gpt-oss-120b-cloud`| ollama-cloud  | cloud-ceiling reference  | F-010 measured at 0.969 on PBS-Agent v0.1 |

### Models NOT in this sweep (with reason)

| litellm_id            | reason for exclusion |
|-----------------------|----------------------|
| `qwen3-30b-a3b-moe`   | llama-swap is currently misconfigured for this model (502 on every request as of 2026-05-27 17:00); operational issue, not in EXP-005 scope. Track as `Phase 19b operational follow-up`. F-010 already established MoE does not outperform dense on PBS-Agent v0.1; deferring MoE on BFCL is a low-cost decision. |
| `phi-4-reasoning-14b` | Same llama-swap issue. |
| `hermes-4.3-36b`      | LiteLLM proxy reports "Invalid model name" — model is in `lab.models` and `conf/llama-swap.yaml` but not surfaced through the proxy; operational issue, not in EXP-005 scope. |
| `xlam-2-7b-fc-r`      | GGUF still unavailable per Phase 19a status. |
| `llama-3.3-70b-q4-local` | Quality-ceiling lane; tagged `slow_mode`; not promoted into BFCL sweep without `--allow-slow-models`. |

### Matrix

- Models: 4
- Tasks: 1000 (BFCL v3 AST categories: 400 simple + 200 multiple + 200 parallel + 200 parallel_multiple)
- Configs: 1 (greedy decoding, max_tokens=1024)
- Seeds: 1 (BFCL is a deterministic single-call benchmark — there is no inter-trial variance to estimate with multi-seed; the upstream leaderboard quotes single-pass accuracy)
- **Total cells: 4000**

This is a substantial increase over the 500-cell target in the master
spec, but it is the natural scale of the published BFCL AST suite:
every example becomes one cell per model. The cells are fast
(qwen3-14b-q4 smoke at ~1-2 s per cell; cloud at <1 s), so wall
estimate is in single hours, not tens of hours.

### Config

```yaml
name: greedy-1024
temperature: 0.0
top_p: 1.0
max_tokens: 1024
scaffold: single_turn
```

`scaffold: single_turn` is nominal — the runner dispatches on the
rubric type to `_execute_bfcl_cell`, which calls LiteLLM once with
`tool_choice: "auto"` and a `tools=[...]` body per the example schema.

### Per-model overrides

```yaml
model_defaults:
  qwen3-14b-q4:
    extra:
      think: false        # reasoning OFF, matches F-005 / EXP-006 baseline
```

The three cloud arms use defaults.

### Estimated cost / wall time

- Cloud: ~3000 cells × 0.5-1 s each = ~1 hour combined (subscription, $0)
- Local (qwen3-14b-q4): 1000 cells × ~1.5 s = ~25 min
- Total wall estimate: **2-3 hours** including warmup and serialisation overhead.

Cells run outer-loop-by-model so qwen3-14b-q4 warms once and the
cloud arms drain serially (`max_concurrency=1` per existing
discipline — the lab has not validated cloud parallelism for this
endpoint family).

## Method

See § Setup above for matrix + config + per-model overrides + statistics.

## Hypothesis

All four hypotheses are evaluated at greedy decoding on the same 1000
cells per model. No multi-seed averaging (single-pass benchmark).

### H1 — Cloud beats local on BFCL by ≥ 10pp (promotion gate)

```text
point_estimate(cloud_best.bfcl_ast_match.mean)
   ≥ point_estimate(qwen3-14b-q4.bfcl_ast_match.mean) + 0.10
```

where `cloud_best` is whichever of `gpt-oss-20b-cloud`,
`glm-5.1-cloud`, `gpt-oss-120b-cloud` scores highest. The community
leaderboard (April 2026 snapshot) shows ~10-30 pp gaps between
mid-tier locals and frontier clouds on the BFCL AST suite; we expect
to replicate that ordering.

### H2 — Local lands in 35-65% range on BFCL AST (measurement gate)

```text
0.35 ≤ mean(qwen3-14b-q4.bfcl_ast_match) ≤ 0.65
```

The published BFCL leaderboard places GPT-4-class models at ~75-85%
overall and 14B-class locals at ~40-55% on the non-live AST suite
(community numbers from April 2026). A score outside [0.35, 0.65] for
qwen3-14b-q4 would be evidence we are either grading much more
strictly than upstream (lower) or that something in our schema /
tool-call passthrough is silently boosting accuracy (higher) — either
case is informative.

### H3 — Cloud-local gap ranks: `gpt-oss-120b-cloud` ≥ `glm-5.1-cloud` ≥ `gpt-oss-20b-cloud` ≥ `qwen3-14b-q4`

We expect the model-size ordering to hold on a community benchmark
because BFCL is the kind of structured-output task where parameter
count + training-data quality dominate. Refutation (ordering inversion)
would be a *finding* about our particular cloud-model line-up that
warrants a follow-up.

### H4 — Per-category profile holds: `simple ≥ multiple ≥ parallel ≥ parallel_multiple`

The categories are ordered by upstream-published difficulty;
specifically, `parallel_multiple` is the hardest (requires multiple
correctly-ordered parallel calls with the right function selection).
We expect each model's per-category accuracy to monotonically
decrease across this ordering. Refutation would suggest the lab's
schema-translation step (Python-type → JSON-schema in
`_to_litellm_tool_spec`) is biasing one category specifically.

## Success / failure criteria

Each hypothesis is judged independently. EXP-005 is **measurement, not
promotion** — F-011 reports verdicts but does not promote or demote any
model.

- **H1 confirmed** ⇔
  `max(point_estimate(<cloud>.bfcl_ast_match)) ≥ point_estimate(qwen3-14b-q4.bfcl_ast_match) + 0.10`.
  Otherwise REFUTED.
- **H2 confirmed** ⇔
  `0.35 ≤ point_estimate(qwen3-14b-q4.bfcl_ast_match) ≤ 0.65`.
  Otherwise REFUTED (and the direction — too low or too high — is the
  finding).
- **H3 confirmed** ⇔ the model ordering on overall accuracy is
  `gpt-oss-120b-cloud ≥ glm-5.1-cloud ≥ gpt-oss-20b-cloud ≥ qwen3-14b-q4`.
  Otherwise REFUTED.
- **H4 confirmed** ⇔ for every model, per-category means satisfy
  `simple ≥ multiple ≥ parallel ≥ parallel_multiple` (ties allowed).
  Otherwise REFUTED (and the per-model breakdown is the finding).

## Decision rule

EXP-005 is **measurement, not promotion** — it produces F-011 (the
finding) rather than promoting any model. The hypotheses are evidence
about how our local stack compares to cloud on a community standard.
No model is promoted or demoted as a function of EXP-005 outcomes;
the goal is to put a community number next to the lab-internal
number from F-010 and report the delta.

If H3 is refuted (ordering inversion) AND the inversion is for a
model the lab currently treats as default-for-tool-use, F-011 will
recommend an ADR (e.g., ADR-009 "default tool-use model") for the
next phase. Otherwise no ADR.

## Kill criteria

The sweep aborts if any of these fire:

- Cell error rate exceeds **5 %** (> 200 errored cells / 4000).
- LiteLLM proxy 5xx rate exceeds 10 % for any 60-second window
  (runner-level signal).
- `qwen3-14b-q4` (ollama-local) wall time on a single category
  exceeds 60 minutes (~6 cells/s, expected ~10× faster).

If kill criteria fire, the analysis script is still run on whatever
cells completed, but every hypothesis verdict is reported as
`INVALID — sweep killed`.

## Confounders to control

- **Greedy decoding** (`temperature: 0.0`, `top_p: 1.0`) on all arms
  matches EXP-002 / EXP-006 / EXP-006b. BFCL community results are
  also greedy.
- **Reasoning OFF on qwen3-14b-q4** matches F-005 / EXP-006 / EXP-006b
  (`think: false`). Without this knob the model produces a long
  reasoning prelude and fires zero tool calls on most examples
  (smoke-confirmed 2026-05-27).
- **Outer-loop-by-model** so each model warms once.
- **`tool_choice: "auto"`** — not `"required"`; Ollama rejects `required`
  on some model families. `"auto"` lets each model decide whether to
  fire a tool call (and the AST grader penalises empty tool_calls
  appropriately via `model_output:no_tool_call`).
- **Sweep-level idempotency**: re-running the sweep does not re-bill
  cloud cells — the runner's `ON CONFLICT (run_id) DO UPDATE` clause
  is the same as PBS-Agent sweeps.
- **No KB query, no sandbox** — BFCL doesn't involve our RAG or
  sandboxing stack. The new `_execute_bfcl_cell` path bypasses both.

## Statistics

Per (model, category) pair:

- Point estimate: `mean(bfcl_ast_match.score)` over all cells.
- Bootstrap 95 % CI: `n_resamples = 2000`, percentile method, same as
  EXP-006b.
- Per-model overall: weighted mean across categories (cells, not
  categories, are the unit) plus bootstrap CI.
- Per-category cross-model comparison: paired-by-task differences
  (model_a − model_b) with a one-sided permutation test (1000
  perms) on the overall vector — reported as context, not in any
  decision rule.

The unit of repeated measurement is **the example**, not the seed —
single-pass community benchmark, no inter-seed averaging.

## Reproduction

```bash
cd /data/lab/code

# 1. Download + register the benchmark (idempotent — ~1.2 MB total)
uv run lab data add-benchmark bfcl-v3

# 2. Register the plan (this file)
uv run lab exp register docs/exp/EXP-005-external-benchmarks.md

# 3. Sweep (~2-3 hr wall, 4000 cells)
uv run lab sweep run conf/sweep/EXP-005.yaml --enforce-pre-registration

# 4. The sweep runner writes the AST grade inline; no second
#    `lab eval apply` pass is required — but if you want to re-grade
#    from traces (e.g. after vendored-checker fixes), run:
uv run lab eval apply EXP-005 --only bfcl_ast_match

# 5. Verdicts + analysis CSVs
uv run python scripts/analyze_exp005.py
```

## Expected output artifacts

- `analysis/EXP-005/SUMMARY.md` — top-line H1/H2/H3/H4 verdicts +
  per-(model, category) table + 1-line headline.
- `analysis/EXP-005/per_model_overall.csv` — overall accuracy with CIs.
- `analysis/EXP-005/per_category.csv` — per-(model, category) means.
- `analysis/EXP-005/per_cell.csv` — per-cell scorer breakdown.
- `docs/findings/F-011-bfcl-v3-external-benchmark.md` — finding doc
  linking back to F-010.

## Pre-mortem

Plausible failure modes for EXP-005 and their cheap mitigations:

- **Risk: schema translation drops a parameter or mistypes a field.**
  Effect: artificially-low scores on `multiple` or `parallel_multiple`
  where the model can't find the right argument. *Mitigation:* per-
  category breakdown surfaces this immediately (a specific category
  dropping while others are normal); follow-up fixes
  `_to_litellm_tool_spec` and re-runs only the affected suite.
- **Risk: cloud rate-limit blast (Ollama Cloud Pro tier).** *Mitigation:*
  `max_concurrency: 1` (existing default). If the sweep hits 429s,
  retry through LiteLLM's existing 502 retry policy (`b0b0e96`).
- **Risk: `think: false` doesn't fully suppress reasoning on
  qwen3-14b-q4 in a tool-calling context.** Smoke-confirmed working
  on 2026-05-27 for `simple_0` and `parallel_0`. If the larger sweep
  surfaces unexpected `no_tool_call` failures, F-011 reports the
  per-category emission rate as a diagnostic, and the experiment
  remains scientifically valid (the metric is what the model emits).
- **Risk: the cloud reference doesn't beat the local arm by 10 pp,
  refuting H1.** *Mitigation:* H1 is a falsifiable hypothesis, not a
  promotion gate. Refutation is a *finding*, not a failure.
- **Risk: the AST grader has a vendor-side bug.** *Mitigation:* 22
  unit tests pin the grader's category dispatch + pass/fail behaviour
  against hand-built fixtures. The vendored code is a literal subset
  of upstream's HEAD-of-main checker.

## Components NOT run end-to-end in EXP-005

Filled in after the sweep — placeholder.

If the sweep completes cleanly, this section enumerates anything in
the plan above that was deferred. The pre-committed defers (also
listed in the master roadmap §17.5) are:

- **τ²-bench**: deferred entirely to Phase 17.5b. The lab's BFCL
  finding ships as F-011; τ²-bench will produce its own F-NNN.
- **`qwen3-30b-a3b-moe`, `phi-4-reasoning-14b`, `hermes-4.3-36b`**:
  not run because llama-swap is misconfigured for these models as
  of pre-reg time. EXP-005 covers `qwen3-14b-q4` as the dense
  local; the operational follow-up to restore llama-swap models
  is tracked separately.
- **`live_*`, `multi_turn_*`, `sql`, `rest`, `java`, `javascript`
  BFCL categories**: not included; the AST grader only covers the
  four non-live Python categories.
