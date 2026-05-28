---
doc_id: f-011-supplement-local-arms
title: 'F-011 supplement: EXP-005-local-followup — 3 dropped local models (qwen3-30b-a3b-moe, phi-4-reasoning-14b, hermes-4.3-36b) on BFCL v3 AST'
zone: lab
kind: finding
status: draft
owner: m
created: '2026-05-28'
last_updated: '2026-05-28'
last_verified: '2026-05-28'
depends_on:
- kind: doc
  target: f-011-bfcl-v3-external-benchmark
- kind: doc
  target: exp-005-local-followup
- kind: doc
  target: exp-005
- kind: code
  target: lab:scripts/analyze_exp005_followup.py
- kind: artifact
  target: lab:analysis/EXP-005-local-followup/SUMMARY.md
- kind: artifact
  target: lab:analysis/EXP-005-local-followup/per_model_overall.csv
- kind: artifact
  target: lab:analysis/EXP-005-local-followup/per_category.csv
- kind: artifact
  target: lab:analysis/EXP-005-local-followup/per_cell.csv
tags:
- lab
- finding
- findings
- bfcl
- external-benchmark
- tool-use
- function-calling
- phase-17
- follow-up
- confidence-high
---

# F-011 supplement: BFCL v3 — 3 dropped local models added

This supplement extends F-011 (cloud + dense local on BFCL v3 AST) with
the three local models the original Phase 17.5 sweep dropped. F-011
stays as the cloud finding; this document is the local-arm follow-on.

Parent: [F-011](./F-011-bfcl-v3-external-benchmark.md).
Pre-reg: [EXP-005-local-followup](../exp/EXP-005-local-followup.md).

## TL;DR (auto-filled from analysis/EXP-005-local-followup/SUMMARY.md after sweep finishes — placeholder)

<!-- HEADLINE_PLACEHOLDER -->

## Combined 7-arm comparison (cloud + dense local + 3 new locals)

```
analysis/EXP-005/per_model_overall.csv          (cloud + dense)
analysis/EXP-005-local-followup/per_model_overall.csv  (3 new locals)
```

<!-- TABLE_PLACEHOLDER

Expected shape (sorted by overall accuracy descending):

| model               | n    | mean accuracy | 95% CI                                            | source experiment |
|---------------------|------|---------------|---------------------------------------------------|-------------------|
| glm-5.1-cloud       | 1000 | <fill>        | <fill>                                            | EXP-005           |
| qwen3-14b-q4        | 1000 | <fill>        | <fill>                                            | EXP-005           |
| <new local 1>       | 1000 | <fill>        | <fill>                                            | EXP-005-local-followup |
| <new local 2>       | 1000 | <fill>        | <fill>                                            | EXP-005-local-followup |
| <new local 3>       | 1000 | <fill>        | <fill>                                            | EXP-005-local-followup |
| gpt-oss-120b-cloud  | 1000 | 0.522         | [0.491, 0.555]                                    | EXP-005           |
| gpt-oss-20b-cloud   | 1000 | 0.520         | [0.489, 0.552]                                    | EXP-005           |

-->

## Per-(model, category) accuracy — new arms only

<!-- CATEGORY_TABLE_PLACEHOLDER -->

## Hypothesis verdicts (this supplement)

The four hypotheses in the follow-up pre-reg (note: these are different
H1-H4 from the parent F-011 — see EXP-005-local-followup.md for the
exact statements):

<!-- VERDICTS_PLACEHOLDER

  H1 — cloud_best (glm-5.1-cloud @ 0.925) beats each new local by >= 10pp:
       <CONFIRMED | REFUTED> (per-model breakdown).

  H2 — qwen3-30b-a3b-moe AND phi-4-reasoning-14b each in [0.50, 0.95]:
       <CONFIRMED | REFUTED>.

  H3 — per-category profile simple >= multiple >= parallel >= parallel_multiple
       for every new model:
       <CONFIRMED | REFUTED> (per-model breakdown).

  H4 — at least one new local beats qwen3-14b-q4 @ 0.910:
       <CONFIRMED | REFUTED>.

-->

## Reconciliation with F-011

F-011's headline (cloud and dense local at ~92% on BFCL v3 AST AST,
gpt-oss line collapses on parallel*) is unaffected by this supplement.
The 3 new arms are inserted into the leaderboard at whatever rank they
score, but the cloud-best (glm-5.1) and dense-local (qwen3-14b-q4)
numbers in F-011's TL;DR table are unchanged.

### Effect on F-011 H3 (model ordering)

F-011 reports H3 REFUTED because the gpt-oss line collapsed on
parallel categories, breaking the expected ordering. The 3 new
locals will re-rank H3 per the combined table — the supplement
reports the updated ordering but does not revise the parent
finding's verdict.

## Effect on tool-use default

<!-- DEFAULT_DECISION_PLACEHOLDER

Two cases:

(a) If max(new_local.overall) >= qwen3-14b-q4 (0.910):
    Flag for PBS-Agent re-confirmation. Do NOT change the lab default
    on BFCL alone — that would be a one-benchmark promotion. ADR on the
    default tool-use model is gated on a second confirming surface
    (PBS-Agent v0.1 with the same model, the F-010 lane).

(b) If max(new_local.overall) < 0.910:
    Dense-local default (qwen3-14b-q4) is unchanged on BFCL. The
    follow-up closes the local-arm coverage gap with no operational
    change. No ADR.

-->

## What didn't run end-to-end

Filled in after sweep completes — placeholder. Pre-committed defers
(per EXP-005-local-followup pre-reg):

- τ²-bench (Phase 17.5b).
- xlam-2-7b-fc-r (no GGUF).
- llama-3.3-70b-q4-local (ceiling-class; separate pre-reg required).
- live_*, multi_turn_*, SQL, REST, Java, JS BFCL categories (out of
  vendored AST grader scope).

## Cost

$0 (local-only sweep).

## Wall time

<!-- WALL_TIME_PLACEHOLDER -->

## Token capture

<!-- TOKEN_CAPTURE_PLACEHOLDER -->
