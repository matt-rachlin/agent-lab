---
doc_id: postmortem-exp-013-narration
title: 'EXP-013 narration postmortem — H4 followup: cap probe + greedy degeneration diagnosis'
zone: lab
kind: postmortem
status: active
owner: m
created: '2026-06-13'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
depends_on:
- kind: doc
  target: exp-013
- kind: doc
  target: f-012-agentic-tool-calling-failure-modes
tags:
- lab
- postmortem
- fine-tuning
- tool-use
- exp-013-followup
---

# EXP-013 narration postmortem — H4 followup

Date created: 2026-06-13
Reclassified as postmortem: 2026-06-14 (was EXP-013b)
Status: complete

## Reclassification notice (2026-06-14)

This document originated on 2026-06-13 as `EXP-013b-ft-toolcall-narration.md`
with the claim "pre-registered: yes (this doc, before the probe results were
read)." Wave-2 contamination + research-rigor audit (2026-06-14) found
that the file was never committed to git before the Stage 1 cap-probe
results were observed — there is no SHA-prior-to-result evidence to
support the pre-reg claim. Per the audit's recommendation:

> Convert EXP-013b's untracked 'pre-reg' into an EXP-013-postmortem doc
> — it's a post-hoc analysis of a refuted hypothesis, that's fine, but
> it's not pre-registration.

Action taken: renamed and reclassified as a postmortem (kind: postmortem)
with full disclosure. The Stage 1 finding (H4a refuted, H4b root-caused
as greedy-decoding degeneration) is preserved as a substantive H4
follow-up record. Stage 2 (conditional act-first retrain) is no longer
pre-registered; reformulating it requires a fresh EXP-NNN with a
committed pre-reg before any further training-data work.

Going forward, `lab exp register` (per the P1.G1 guardrails commit
on this branch) fills in the registration SHA automatically and the
pre-push hook refuses push if any tracked EXP doc still carries the
placeholder, so this discipline gap should be closed at the tooling
level.

## Background

EXP-013 confirmed H1/H2/H3 decisively but REFUTED H4 as written: the
fine-tuned arm had 10 zero-tool-call ("narration") episodes (vs threshold
0), concentrated on 4 hard multi-step tasks (spec-invoice-window,
spec-tournament-podium [brutal]; data-log-parse-p95-latency,
data-payment-reconciliation [hard]), firing on every seed.

## Diagnosis (from the trajectories + turn metadata)

Every flagged episode is **one turn**, with `tokens_out == 4096` (the exact
per-turn output cap), `budget_exhausted == false`, and assistant content
that is a verbose natural-language plan truncated mid-sentence ("Okay,
let's tackle this step by step... I'll start by reading both files" — cut
off). So the model is NOT tool-averse; it exhausts the output-token budget
**narrating a plan before emitting its first tool call**, the harness sees
a no-tool-call message and ends the episode.

Two facts reframe the "fix":
1. **The 4096 cap is an eval-config choice** (`react-4096` in the eval
   YAMLs), set during the gemma4-12b-era campaign — a light-thinking model
   that fits its action well under 4k. It is tight for a verbose Qwen3-4B.
   `think:false` is already set for both arms, so this is plain-content
   verbosity, not Qwen3 thinking-block tokens.
2. **2 of the 4 tasks were already in training** (data-payment-reconciliation
   x8, data-log-parse-p95-latency x8 own-trajectory episodes). The model
   trained on successful runs of these and STILL over-narrates on some
   seeds — so "more of the same task" is not the lever. (The two spec-*
   tasks are brutal-only -> clean held-out, must never enter training.)

Both arms ran under the same 4096 cap, so **H1/H2/H3 remain valid** (fair,
apples-to-apples). Only H4 is confounded.

## Stage 1 — cap probe (queued: tasks 113/114)

Re-run the **ft arm only** on brutal + hard at `max_tokens: 16384`
(configs ft-eval-brutal-cap16k / ft-eval-hard-cap16k; slugs
FT-EVAL-*-CAP16K-001), everything else identical. Re-audit narration.

- **H4a (artifact):** CONFIRMED iff ft narration episodes drop from 10 to
  <= 2 at the 16384 cap. -> H4's refutation was largely a stingy-cap
  artifact; record it, no retrain needed, and flag that the campaign's
  4096 cap under-serves verbose/reasoning models (eval-harness finding).
- **H4b (behavior):** if narration persists (> 2) WITH room to think, the
  verbosity is a genuine learned behavior -> proceed to Stage 2.

Note pass@1 may also move (episodes that previously truncated may now
complete) — recorded, but the probe's primary readout is the narration
count, not the score (the score arms of EXP-013 stand at 4096).

## Stage 2 — conditional act-first retrain (only if H4b)

Intervention: an **act-first curriculum** — (a) trim verbose pre-first-tool
preambles in the existing tool trajectories to a short cap, and (b) add
procedurally-generated terse-preamble data-pipeline exemplars (CSV ordered
pipelines, dedup, percentile, reconciliation), **generated independently of
the eval suites** (no brutal/hard task instances; spec-* never rendered).
Re-train QLoRA with the same hyperparameters; re-run the full EXP-013 eval
trio + audit.

- Success: ft narration -> <= 2 AND no regression on H1 (BFCL >= prior
  -2pp), H2 (brutal pass@1 >= prior -2pp), H3 (hard >= prior).

## Guardrails

- The brutal suite stays clean: NO brutal task instance, and specifically
  no spec-invoice-window / spec-tournament-podium content, enters any
  training data. Stage-2 exemplars are synthetic + structurally distinct.
- One variable per stage; cap change (Stage 1) and data change (Stage 2)
  are not mixed.

---

## Stage 1 results (cap probe) — recorded 2026-06-13

Re-ran the ft arm on brutal+hard at `max_tokens: 16384` (FT-EVAL-*-CAP16K-001),
re-audited narration.

| metric | 4096 cap (EXP-013) | 16384 cap (probe) |
|---|---|---|
| narration episodes (ft, brutal+hard) | 10 | **9** |

**H4a (cap artifact): REFUTED.** 4x the output budget did not reduce
zero-tool-call episodes. Every residual episode hits `tokens_out == cap`
exactly (4096 -> 16384) with empty content, null tool_calls, empty
content_preview — i.e. the whole budget goes to `<think>` reasoning that is
stripped and never resolves into a tool call.

**H4b (genuine behavior): CONFIRMED but narrow + fragile.** Runaway /
non-terminating generation on a few specific hard prompts (spec-invoice-window,
spec-tournament-podium, code-lru-cache-trace, data-payment-reconciliation),
not tool-aversion. Evidence it is NOT fundamental:
- Direct ollama, `think:false` honored: tool_call emitted, 807 tokens. OK.
- Proxy, paraphrased prompt: tool_call emitted, 1714 tokens. OK.
- Only the exact eval prompt + full react scaffold + greedy decoding triggers it.
- Modelfile has `repeat_penalty 1` (disabled); eval is temp 0 (greedy) —
  textbook greedy-degeneration conditions. `/no_think` made it WORSE.

**Revised root cause:** greedy-decoding degeneration on a handful of hard
prompts, not a training defect (the ft model wins everywhere else).

**Revised Stage 2 (cheap fix first, before any retrain):** re-eval the ft arm
with a small repetition penalty (`repeat_penalty ~1.1`) and/or tiny temperature;
if the runaways break, H4 is resolved with no retrain. Retrain (act-first
curriculum) only if decoding fixes fail. The 4096 eval cap stays (raising it
just burns compute).
