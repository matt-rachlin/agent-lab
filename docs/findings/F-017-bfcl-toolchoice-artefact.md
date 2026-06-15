---
doc_id: f-017-bfcl-tool-choice-artefact
title: 'F-017: BFCL tool_choice artefact — phi-4-reasoning re-eval'
zone: lab
kind: finding
status: active
owner: m
created: '2026-06-13'
last_updated: '2026-06-13'
last_verified: '2026-06-13'
tags:
- lab
- finding
- bfcl
- phi
- trust-lifecycle
- tool-choice
depends_on:
- kind: doc
  target: adr-008-trust-lifecycle
- kind: doc
  target: exp-phi-toolchoice-001-bfcl-toolchoice-rerun
---

# F-017: A reasoning model scored ~1% on BFCL because of a `tool_choice` harness artefact — true function-calling accuracy is ~45%

Date: 2026-06-13
Confidence: high
Source: PHI-TOOLCHOICE-001 (A/B); EXP-005 / OLLAMA-BFCL-FULL-001 (historical corpus)
trust_level: unverified

## Claim

Our BFCL v3 AST evaluator (`bfcl_ast_match`) reported `phi-4-reasoning-plus` at
~1% — bottom of the leaderboard. That number was a **measurement artefact, not a
capability**. BFCL measures *function-calling*: the task is defined as emitting a
tool call. Under the harness's historical default `tool_choice="auto"`, this
reasoning model answered in fluent prose and emitted **no tool call**, so the AST
checker correctly recorded `model_output:no_tool_call` and scored it 0. The score
conflated *non-emission* (a decode/format outcome) with *incorrectness* (a
capability outcome). Isolating `tool_choice` in an A/B raises emission from
**1.7% → 100%** and reveals a true accuracy of **~45%**.

## Evidence

**A/B — same 60 tasks (15 per category), only `tool_choice` varied:**

| arm | emission | pass | accuracy-given-emission | avg output tokens |
|---|---|---|---|---|
| `auto` (old default) | 1.7% | 0.0% | 0.0% | 332 (prose) |
| `required` (fix) | 100.0% | 45.0% | 45.0% | 37 (direct call) |

The token counts are the tell: under `auto` the model spends ~332 tokens
reasoning in prose and never calls; under `required` it emits a 37-token tool
call directly. Example recovered call: `triangle_properties.get({"side1":5,
"side2":4,"side3":3})`, graded `valid:true`.

**This is a per-model pathology, not a broken harness.** Across ~10,600 historical
BFCL evals only **13%** were `no_tool_call` (87% of runs emit a call); the eight
non-reasoning models emit 85–100%. The artefact concentrates in reasoning models
under permissive `tool_choice`, compounded by a backend limitation: Ollama rejects
`tool_choice="required"`, so the harness had defaulted everything to `"auto"`.

**A second, separate defect found en route (cosmetic):** the dominant historical
`error_type`, `simple_function_checker:unclear` (47.8% of evals), was *not* a
failure mode — a crosstab showed **all** such rows had `passed=true`. The checker
initialised `error_type` to `"...unclear"` and the success path returned that dict
unchanged. Scores were unaffected, but the telemetry column was misleading. Fixed.

## Fix

Three code changes (`fix/bfcl-harness-toolchoice`):

1. **Backend-aware `tool_choice`.** `"required"` is the semantically correct
   setting for a function-calling benchmark; fall back to `"auto"` only on
   backends that reject it (Ollama).
   ```python
   backend_l = (cell.model_backend or "").lower()
   default_tool_choice = "auto" if "ollama" in backend_l else "required"
   merged_extra.setdefault("tool_choice", default_tool_choice)
   ```
2. **Emission/accuracy decomposition in the report** — never report a
   non-emission as a wrong answer:
   ```
   emit_rate_pct      = runs that emitted any tool call
   acc_given_emit_pct = pass rate among runs that emitted a call
   ```
3. **Trace fidelity** — log `request_tools` + `tool_choice` into the trace, so
   emission failures are auditable from the trace rather than inferred. (Before
   this, the request side was not recorded, which is what let the artefact hide.)

Plus the cosmetic checker fix (init `error_type=None`, set explicitly on failure).

## Caveats / limits

- The A/B is a 60-task sample (proof of the effect), not the full 1000-task suite;
  ~45% is the sample estimate, not yet a citable headline figure.
- `tool_choice="required"` forces a call, so it cannot measure *abstention* — fine
  for BFCL (every task expects a call) but not a general tool-use metric.
- BFCL AST match scores call-format + argument correctness, **not** task-solving.
  A model that explains the correct answer in prose still scores 0 — by design.
- The eight non-reasoning models' `acc_given_emit` should be cross-checked against
  the published BFCL leaderboard before any number is quoted publicly.

## Implications

General lessons for LLM eval harnesses, beyond this one model:
- **Decompose emission from correctness.** Conflating them silently penalises any
  model whose output format differs from what the scorer expects.
- **Decode settings and backend capabilities bias leaderboards.** `tool_choice`
  and "Ollama can't do `required`" turned a capability benchmark into a
  format-compliance one for one model family.
- **Log the full request, not just the response.** Auditing required the tools and
  `tool_choice` that were sent; their absence is what hid the bug.
- **Don't trust telemetry fields without checking them** (the `unclear` label).

## Open questions

- Full 1000-task phi number under `required`.
- Do other reasoning models (e.g. `qwen3-30b-a3b`, which already emits ~100%)
  shift materially under `required` vs `auto`?
- Best treatment of abstention in a general tool-use eval.

## Status
- [x] Logged
- [x] Replicated (PHI-TOOLCHOICE-001 A/B)
- [ ] Published
