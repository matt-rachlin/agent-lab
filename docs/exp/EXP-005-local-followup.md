---
doc_id: exp-005-local-followup
title: 'EXP-005-local-followup: BFCL v3 external benchmark — follow-up adding the 3 local models the original sweep dropped'
zone: lab
kind: exp
status: active
owner: m
created: '2026-05-28'
last_updated: '2026-05-28'
last_verified: '2026-05-28'
depends_on:
- kind: doc
  target: exp-005
- kind: doc
  target: f-011-bfcl-v3-external-benchmark
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
- follow-up
---

# EXP-005-local-followup: BFCL v3 — adding 3 dropped local models

Date created: 2026-05-28
Status: planned
Pre-registered: 5c1d3c6  (registered by `lab exp register` at file-creation time; backfilled 2026-06-14)
Parent docs: [EXP-005 pre-reg](/data/lab/code/docs/exp/EXP-005-external-benchmarks.md);
[F-011 (cloud + dense local) finding](/data/lab/code/docs/findings/F-011-bfcl-v3-external-benchmark.md);
[F-010 PBS-Agent baselines](/data/lab/code/docs/findings/F-010-qwen3-30b-moe-re-anchored-not-promoted-H2-H4-fail.md).

## Why a follow-up

The original EXP-005 pre-reg explicitly excluded three local models —
`qwen3-30b-a3b-moe`, `phi-4-reasoning-14b`, and `hermes-4.3-36b` —
citing a llama-swap operational issue as of 2026-05-27 17:00 EDT.
Subsequent diagnostic work (Phase 19b/19d/19e) verified all three
models load and run cleanly through llama-swap (smoke-confirmed
2026-05-28 against llama-swap CUDA build at HEAD `4d29d34`):

| litellm_id | llama-swap smoke | LiteLLM proxy smoke | note |
|---|---|---|---|
| `qwen3-30b-a3b-moe` | OK | OK | already in LiteLLM config |
| `phi-4-reasoning-14b` | OK | OK | already in LiteLLM config |
| `hermes-4.3-36b` | OK | needed proxy wiring | added in this commit |

The `hermes-4.3-36b` LiteLLM proxy entry was missing from
`conf/litellm-config.yaml`; that operational gap is fixed in the same
commit that lands this pre-reg. The Phase 17.5 original-agent claim
that all three were "operationally misconfigured" was correct only
for hermes (and trivially fixable); the other two worked through
llama-swap already.

EXP-005 itself is **complete** (4000/4000 cells, zero errors;
analysis at `analysis/EXP-005/SUMMARY.md`). The cloud-vs-dense-local
finding F-011 stays exactly as committed. This follow-up adds three
local arms so the F-011-supplement can table a 7-arm comparison
(4 original + 3 new) covering the full registered local matrix.

## Question

Where do `qwen3-30b-a3b-moe`, `phi-4-reasoning-14b`, and
`hermes-4.3-36b` land on BFCL v3 AST relative to the existing
EXP-005 arms? Specifically:

1. Does any local model dethrone `qwen3-14b-q4` (0.910 on EXP-005)
   as the dense-local default for tool use on this suite?
2. How does the MoE arm `qwen3-30b-a3b-moe` (3B active) compare to
   the dense `qwen3-14b-q4` from the same family? F-010 showed MoE
   under-performed dense on PBS-Agent v0.1; we expect the same on
   BFCL but the per-category profile may differ.
3. Does `phi-4-reasoning-14b`, with reasoning ON, beat the
   reasoning-OFF dense baseline on the harder categories
   (`parallel`, `parallel_multiple`)? If yes, this is the first lab
   evidence that reasoning helps on this specific surface.
4. Is `hermes-4.3-36b`'s advertised "reliable JSON tool calling"
   actually delivered? The model card claims 512K context + first-
   class tool use; BFCL is the cleanest test for that claim.

## Setup

### Models

| litellm_id | backend | role | notes |
|---|---|---|---|
| `phi-4-reasoning-14b` | llama.cpp (via llama-swap) | reasoning specialist | reasoning ON (model default); 14B Q4_K_M |
| `qwen3-30b-a3b-moe` | llama.cpp (via llama-swap) | MoE-default candidate | `think: false` (F-009/F-010 convention); A3B with experts on CPU |
| `hermes-4.3-36b` | llama.cpp (via llama-swap) | tool-use claimant | 36B Q4_K_M hybrid offload; vendor positions as tool-call-first |

All three route llama-swap (port 8080) -> LiteLLM proxy (port 4000)
-> sweep runner. `keep_alive: 0` semantics are honoured by llama-swap
per Phase 19b conventions.

### Matrix

- Models: 3 (new local)
- Tasks: 1000 (identical BFCL v3 AST suite to EXP-005)
- Configs: 1 (greedy decoding, max_tokens=1024)
- Seeds: 1 (single-pass community benchmark)
- **Total cells: 3000**

### Config

Identical to EXP-005:

```yaml
name: greedy-1024
temperature: 0.0
top_p: 1.0
max_tokens: 1024
scaffold: single_turn
```

### Per-model overrides

```yaml
model_defaults:
  qwen3-30b-a3b-moe:
    extra:
      think: false
```

- `phi-4-reasoning-14b`: no override — reasoning is the model's
  designed mode of operation; toggling it off would defeat the
  purpose of running it.
- `hermes-4.3-36b`: no override — model defaults to reasoning OFF
  for tool-call output (per Hermes 4 card).

### Estimated wall time

Local-only sweep, all through llama-swap:

- `phi-4-reasoning-14b`: 14B Q4_K_M, ~17 tok/s -> 1024 tok ~= 60s × 1000 = ~17 hr **worst case** (this is reasoning ON, so output tokens will spike on hard examples). More realistic: average reasoning trace ~400 tok -> ~25 s/cell × 1000 = ~7 hr.
- `qwen3-30b-a3b-moe`: A3B MoE, ~36 tok/s -> 1024 tok ~= 28s × 1000 = ~8 hr worst case. With `think: false` typical output << 1024 (tool-call envelope is ~50-200 tok), so realistic: ~5 s/cell × 1000 = ~1.5 hr.
- `hermes-4.3-36b`: 36B Q4_K_M hybrid offload, ~8 tok/s -> 1024 tok ~= 130s × 1000 = ~36 hr worst case. With tool-call output << 1024 typical: ~10 s/cell × 1000 = ~3 hr realistic.

Total realistic estimate: **~11 hr**. Worst case (every cell hits
max_tokens): substantially more. Cost: $0 (local only).

## Method

See § Setup above for matrix + config + per-model overrides. The
sweep dispatches to ``_execute_bfcl_cell`` in the lab sweep runner
(same code path as EXP-005), which issues a single tool-calling
LiteLLM request per cell and grades inline via the vendored AST
checker. No second ``lab eval apply`` pass is required. Statistics
are computed by ``scripts/analyze_exp005_followup.py`` against the
``EXP-005-local-followup`` slug.

## Hypothesis

All four hypotheses are evaluated at greedy decoding on the same
1000 cells per model. No multi-seed averaging.

### H1 — Cloud-best (from EXP-005) beats every new local by ≥ 10pp

```text
0.9250 (glm-5.1-cloud point estimate from EXP-005)
   >= mean(<new_model>.bfcl_ast_match) + 0.10
   for each of: qwen3-30b-a3b-moe, phi-4-reasoning-14b, hermes-4.3-36b
```

This is the cloud-promotion-gap restatement of EXP-005-H1 against
the three new local arms. Refutation per-model is informative —
specifically, a refutation by `hermes-4.3-36b` would validate its
vendor positioning.

### H2 — `qwen3-30b-a3b-moe` and `phi-4-reasoning-14b` land in [0.50, 0.95]

```text
0.50 <= mean(qwen3-30b-a3b-moe.bfcl_ast_match) <= 0.95
0.50 <= mean(phi-4-reasoning-14b.bfcl_ast_match) <= 0.95
```

Loose measurement gate. The community-tier expectation for
14B-30B-class locals on BFCL v3 AST (April 2026 snapshot) is
roughly [0.50, 0.85]; the upper bound is widened to 0.95 because
the lab's dense-local already scored 0.91 on EXP-005, suggesting
our schema-translation path is on the favourable side of the
upstream grader. A score outside [0.50, 0.95] for either model is
evidence of either grader-side surprise or a model-side regression.

`hermes-4.3-36b` is excluded from H2 because there is no public
community number for hermes-4.3 on BFCL v3 AST as of this pre-reg.

### H3 — Per-category profile holds for every new model

```text
simple >= multiple >= parallel >= parallel_multiple  (ties allowed)
   for each of the 3 new models.
```

Matches the H4 from the original EXP-005. Refutation would suggest
the model's tool-call envelope is biased toward one category, or
the schema-translation path is biasing one category specifically.

### H4 — At least one new local beats `qwen3-14b-q4` (0.910) on overall

```text
max(mean(<new_model>.bfcl_ast_match)) >= 0.910
   over: qwen3-30b-a3b-moe, phi-4-reasoning-14b, hermes-4.3-36b
```

**This is the headline decision rule for the follow-up.** If true,
the dense-local default for tool use may need to change (subject
to PBS-Agent re-confirmation — BFCL is a single benchmark and
ADR-promotion would require a second confirming surface). If false,
F-011-supplement reports "dense-local default
(qwen3-14b-q4) is unchanged on BFCL; no ADR triggered."

## Success / failure criteria

Each hypothesis is judged independently. EXP-005-local-followup is
**measurement, not promotion** — it produces an F-011 supplement
(F-011-supplement-local-arms) rather than promoting or demoting any
model.

- **H1 confirmed** ⇔ for every new local M in
  {qwen3-30b-a3b-moe, phi-4-reasoning-14b, hermes-4.3-36b},
  `0.9250 (glm-5.1-cloud) - mean(M.bfcl_ast_match) >= 0.10`.
  Otherwise REFUTED (per-model details reported).
- **H2 confirmed** ⇔
  `0.50 <= mean(qwen3-30b-a3b-moe.bfcl_ast_match) <= 0.95` AND
  `0.50 <= mean(phi-4-reasoning-14b.bfcl_ast_match) <= 0.95`.
  Otherwise REFUTED.
- **H3 confirmed** ⇔ for every new model, per-category means
  satisfy `simple >= multiple >= parallel >= parallel_multiple`
  (ties allowed). Otherwise REFUTED (per-model breakdown is the
  finding).
- **H4 confirmed** ⇔
  `max(mean(new_local.bfcl_ast_match)) >= 0.910 (qwen3-14b-q4
  baseline from EXP-005)`. Otherwise REFUTED.

## Decision rule

EXP-005-local-followup is **measurement, not promotion** — it
produces a supplement to F-011 (cited as F-011-supplement) rather
than a standalone finding or any model promotion. No ADR is
triggered as a direct function of this experiment's verdicts:

- If H4 is confirmed (a new local beats qwen3-14b-q4), F-011-
  supplement flags a follow-up: re-run that model on PBS-Agent
  v0.1 to see whether the BFCL improvement transfers. ADR
  on the default tool-use model is gated on that second result,
  not on EXP-005-local-followup alone.
- If every H is refuted, the supplement reports the per-model
  numbers and closes the local-arm coverage gap F-011 originally
  flagged.

## Kill criteria

Same as EXP-005:

- Cell error rate exceeds **5 %** (> 150 errored cells / 3000).
- LiteLLM proxy 5xx rate exceeds 10 % for any 60-second window.
- Any single model's wall time on a single category exceeds 6 hr
  (4× the realistic per-category budget at ~25 s/cell).

If kill criteria fire, the per-model analyzer still emits
whatever cells completed; the supplement reports those numbers as
"INVALID — sweep killed for <reason>" with no verdict.

## Confounders to control

Same as EXP-005, with the following arm-specific notes:

- **Reasoning ON for phi-4-reasoning-14b** is intentional — the
  model is designed to reason and toggling it off mis-represents
  the arm. The cost is wall time: a reasoning model emits
  hundreds of tokens of trace before the tool call, which the
  AST grader ignores but llama.cpp still has to generate.
- **Reasoning OFF for qwen3-30b-a3b-moe** matches the F-009/F-010
  amendment (`think: false`). Without it, MoE fires zero tool
  calls on most examples.
- **Hermes 4.3 reasoning default**: model defaults to reasoning OFF
  in tool-calling contexts per vendor card; no override needed.
- **Outer-loop-by-model** so each llama.cpp process warms once
  per arm and llama-swap evicts cleanly between arms.
- **`tool_choice: "auto"`** identical to EXP-005.
- **Sweep-level idempotency**: separate experiment slug
  (`EXP-005-local-followup`) -> no collision with EXP-005's
  experiment_runs rows. Re-running this sweep is idempotent per
  `ON CONFLICT (run_id) DO UPDATE`.

## Statistics

Per (model, category) pair:

- Point estimate: `mean(bfcl_ast_match.score)` over all cells.
- Bootstrap 95% CI: `n_resamples = 2000`, percentile method.
- Per-model overall: weighted mean across categories (cells, not
  categories, are the unit) plus bootstrap CI.

The unit of repeated measurement is **the example**, identical to
EXP-005.

## Reproduction

```bash
cd /data/lab/code

# 1. The BFCL benchmark is already on disk from EXP-005; no
#    additional data download is required.

# 2. Register the plan (this file)
uv run lab exp register docs/exp/EXP-005-local-followup.md

# 3. Sweep (~11 hr realistic wall, 3000 cells)
uv run lab sweep run conf/sweep/EXP-005-local-followup.yaml \
    --enforce-pre-registration

# 4. Analyzer (mirrors analyze_exp005.py against the new slug)
uv run python scripts/analyze_exp005_followup.py

# 5. Supplement finding
#    docs/findings/F-011-supplement-local-arms.md
```

## Expected output artifacts

- `analysis/EXP-005-local-followup/SUMMARY.md` — top-line H1/H2/H3/H4 verdicts + per-(model, category) table + 1-line headline.
- `analysis/EXP-005-local-followup/per_model_overall.csv`
- `analysis/EXP-005-local-followup/per_category.csv`
- `analysis/EXP-005-local-followup/per_cell.csv`
- `docs/findings/F-011-supplement-local-arms.md` — supplement to
  F-011 with the combined 7-arm comparison table.

## Components NOT run end-to-end in EXP-005-local-followup

Filled in after the sweep — placeholder.

Pre-committed defers:

- **τ²-bench**: still deferred (Phase 17.5b).
- **`xlam-2-7b-fc-r`**: still no GGUF (Phase 19a status unchanged).
- **`llama-3.3-70b-q4-local`**: quality-ceiling lane, gated by
  `--allow-slow-models`; not in this follow-up's scope (a 70B Q4
  reasoning-eligible run on 3000 cells is a >24hr commitment that
  warrants its own pre-reg).
- **`live_*`, `multi_turn_*`, SQL, REST, Java, JS BFCL categories**:
  same as EXP-005 — out of scope for the vendored AST grader.
