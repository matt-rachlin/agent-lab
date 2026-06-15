---
doc_id: exp-013
title: 'EXP-013: FT-TOOLCALL-001 — close the eval→train→eval loop: QLoRA fine-tune
  Qwen3-4B for agentic tool-calling on the lab''s own verified trajectories
  (pre-registered)'
zone: lab
kind: exp
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-13'
last_verified: '2026-06-13'
depends_on:
- kind: doc
  target: lab-roadmap-2026-06
- kind: doc
  target: f-012-agentic-tool-calling-failure-modes
- kind: doc
  target: f-014-cloud-anchor-hard-suite
tags:
- lab
- exp
- fine-tuning
- qlora
- tool-use
- eval-train-eval
---

# EXP-013: FT-TOOLCALL-001 — the eval→train→eval loop

Date created: 2026-06-11
Status: complete — 3/4 hypotheses confirmed (H1/H2/H3), H4 refuted-but-improved
Pre-registered: bec8b99  (registered by `lab exp register` at file-creation time; backfilled 2026-06-14)

## Question

F-012 established that agentic tool-calling fidelity — not size or
coding skill — gates local agent performance, and attributed the gap to
training. Can a 12 GB lab *change* that training? Concretely: does QLoRA
SFT of Qwen3-4B on a 60/40 mix of tool-calling data — including 659 of
the lab's own verified-successful agent trajectories (STaR/RFT-style,
with 62 frontier-teacher episodes from HARD-BENCH-CLOUD-001) — measurably
improve agentic performance on held-out suites?

## Hypothesis

- **H1 (held-out format generalization):** fine-tuned ≥ base + 5pp on
  BFCL v3 AST overall (1000 tasks, never in training data).
- **H2 (held-out task generalization):** fine-tuned > base on
  pbs-agent-brutal-v0.1 pass@1 by ≥ 2 tasks-equivalent (≥ 8.3pp at
  n=24). The brutal suite postdates the dataset build — zero episodes
  of it exist in training data.
- **H3 (seen-task gain):** fine-tuned ≥ base + 10pp on
  pbs-agent-hard-v0.1. DISCLOSED CONTAMINATION: training data contains
  successful hard-suite trajectories (its own + cloud teachers), so H3
  measures memorization-inclusive gain and is reported separately from
  H1/H2 — never headline.
- **H4 (no protocol regression):** fine-tuned shows zero F-012 failure
  modes (narration / text-emitted calls) on the audited suites —
  fine-tuning must not break what works (trajectory_audit.py is the
  check).

## Method

### Training (prepared by the ft pipeline, /data/lab/ft/)

- base: unsloth/Qwen3-4B, QLoRA 4-bit, r=16 α=32, lr 2e-4, ≤2 epochs,
  batch 2 × grad-accum 8, bf16, responses-only masking via the Qwen3
  chat template, seed 1, MLflow experiment FT-TOOLCALL-001.
- data: train_mix.jsonl — 20,000 samples, 60% tool (659 lab
  trajectories + 3,780 xLAM + 3,780 Hermes-FC + 3,781 ToolACE) / 40%
  general (ultrachat). Lab trajectories: end_state==1.0 only, 14/1,295
  rejected by faithfulness cross-checks (alignment, truncation,
  recovered-call filters); rendered against the real MCP tool schemas.
- PRE-RUN AMENDMENT 2026-06-11: the original registration excluded
  xLAM-60k (gate unaccepted). The gate was accepted and the mix rebuilt
  with xLAM included BEFORE training started (job still queued at
  amendment time). Hypotheses, thresholds, and eval design unchanged.

### Evaluation (both arms identical; q4_k_m GGUF parity)

| eval | status vs training data | seeds |
|---|---|---|
| BFCL v3 AST (vendored, 1000) | clean | 1 (deterministic single-turn) |
| pbs-agent-brutal-v0.1 (24) | clean | 3 |
| pbs-agent-hard-v0.1 (32) | CONTAMINATED (disclosed) | 3 |

Arms: `qwen3-4b-base-q4` vs `qwen3-4b-ft-toolcall-q4`, both served via
ollama from q4_k_m GGUFs, same litellm lane config, v2 prompt, react
scaffold, temp 0. Baseline arm runs are part of this experiment (the
base 4B has never been benchmarked in the lab).

## Success / failure criteria

- H1: CONFIRMED iff ft_overall − base_overall ≥ 0.05 on BFCL AST;
  REFUTED if ≤ 0; between → INCONCLUSIVE.
- H2: CONFIRMED iff ft − base ≥ +2/24 tasks mean pass@1 on brutal;
  REFUTED if ft < base; between → INCONCLUSIVE.
- H3: CONFIRMED iff ft − base ≥ 0.10 on hard suite (reported with the
  contamination label regardless).
- H4: CONFIRMED iff trajectory_audit narration+text_emitted == 0 for
  the ft arm across both agent suites; any episode REFUTES.

## Kill criteria

- Kill training on OOM unrecoverable at --max-seq-length 4096 fallback,
  or loss divergence (train loss not decreasing over first 200 steps).
- Kill eval if GGUF export produces template-broken serving (tool calls
  unparseable at inference — the documented Ollama Modelfile/template
  risk); fix the template, re-export, restart eval (training stands).
- Train loss < 0.2 sustained ⇒ overfitting per unsloth guidance — stop
  at the checkpoint before it.

## Analysis plan

Per-eval before/after table with deltas + CIs where seeded; H1–H4
verdicts regardless of direction; trajectory_audit report on the ft
arm; if H1 or H2 confirm, a public writeup follows ("closing the
eval→train→eval loop on a 12 GB GPU") — the lab's flagship artifact.

---

## Results (recorded 2026-06-13)

Runs: BFCL `FT-EVAL-BFCL-001` (92), brutal `FT-EVAL-BRUTAL-001` (93),
hard `FT-EVAL-HARD-001` (94). All n match pre-registration (BFCL 1000x1,
brutal 24x3=72, hard 32x3=96), 0 execution errors on either arm, both
served from q4_k_m GGUF via the canonical litellm lanes `qwen3-4b` (base)
and `qwen3-4b-ft-toolcall-q4-latest` (ft).

### Headline before/after

| eval | metric | base | ft | delta | verdict |
|---|---|---|---|---|---|
| BFCL v3 AST (clean) | AST pass-rate | 0.647 | **0.837** | **+19.0pp** | **H1 CONFIRMED** (>=+5pp) |
| brutal v0.1 (clean held-out) | pass@1 | 0.028 | **0.250** | **+22.2pp** (+5.3 task-equiv/24) | **H2 CONFIRMED** (>=+2 tasks) |
| hard v0.1 (disclosed contam.) | pass@1 | 0.177 | **0.448** | **+27.1pp** | **H3 CONFIRMED** (>=+10pp) |

H1/H2 are the headline (clean, no contamination). H2 is the strongest
signal: the brutal suite postdates the dataset freeze, so the ~9x pass@1
gain (2->18 of 72 runs) cannot be memorization.

### Efficiency (secondary, not pre-registered — striking enough to record)

| eval | latency base->ft | out-tokens base->ft |
|---|---|---|
| BFCL | 4246->1993 ms (-53%) | 708->314 (-56%) |
| brutal | 40.3->33.2 s (-18%) | 5501->3331 (-39%) |
| hard | 45.9->18.0 s (-61%) | 6418->1890 (-71%) |

The ft model is simultaneously more accurate AND 2-2.5x faster / 40-70%
less verbose — it stops the reasoning-ramble and calls the tool.

### H4 — trajectory audit (REFUTED as pre-registered)

`trajectory_audit.py` mechanical classifiers on the ft arm, both agent
suites:

| classifier | base (brutal+hard) | ft (brutal+hard) |
|---|---|---|
| **text_emitted** (F-012 malformed JSON-as-text call) | — | **0** |
| **narration** (0 structured calls in episode) | 28 (7+21) | **10** (5+5) |

H4 was pre-registered as narration+text_emitted == 0, "any episode
REFUTES." The ft arm has **10 narration episodes**, so **H4 is REFUTED
as stated.** Honest characterization of the miss:

- The specific F-012 *malformed-call* mode (`text_emitted`) is fully
  **eliminated** (0 episodes) — the worst-of-F-012 is gone.
- Pure narration dropped **64%** vs base (28->10); fine-tuning introduced
  **no new** failure mode, it reduced the existing one — just not to zero.
- The residual is **concentrated and deterministic**: only 4 tasks
  (spec-invoice-window, spec-tournament-podium, data-log-parse-p95-latency,
  data-payment-reconciliation), and on those it narrates on *every* seed.
  That is a targetable per-task weakness (candidate for a focused data
  top-up), not a broad protocol regression.

### Overall

The eval->train->eval loop closes positively: **3/4 hypotheses confirmed
decisively** (H1 +19pp, H2 +22pp/9x on clean held-out, H3 +27pp), H4
refuted-but-improved. QLoRA SFT on 659 of the lab's own verified
trajectories (+ public tool-call data) turned a 12 GB-trainable 4B model
into a materially better — and far more efficient — local agent, with the
generalization confirmed on an uncontaminated suite. Public-writeup
trigger (H1 or H2 confirmed) is **met**; the residual-narration nuance is
part of the honest story, not a blocker.

### Open limitations (added 2026-06-14, post-audit)

The 2026-06-14 perfect-order audit (wave-2 contamination + research-rigor
review) surfaced three caveats that the writeup must carry:

1. **BFCL N=1, brutal+hard N=3 — below ADR-004's N≥8 floor.** The +19pp,
   +22.2pp, +27.1pp deltas are point estimates without bootstrap CIs.
   Pre-registration permitted "single deterministic pass" for BFCL only;
   brutal/hard were not justified at N=3. Re-run at N>=8 before any
   formal H1/H2/H3 claim that would gate downstream work (e.g., scoreboard
   tier-1-deployable, per ADR-009).

2. **BFCL "clean" applies to rows, not to distribution.** No BFCL row
   appears verbatim in training. However, **11,341 of the 20,000-record
   training mix (56.7%) use the same OpenAI tool-call envelope BFCL
   grades**: xLAM (3,780) + Hermes-FC (3,780) + ToolACE (3,781). The
   contamination-check protocol section 5 (perturbed-twin) was not run
   against BFCL; "held-out format" is supported only at the task level,
   not the distribution level. Soften writeup language accordingly.

3. **Hard suite is 100% task-overlap, not partial.** The "disclosed
   contamination" label is correct but qualitative. Quantitatively:
   **32/32 hard-suite tasks have at least one trajectory in training**,
   with 4-11 trajectories per task (**220 trajectories total = 33.4% of
   the lab-trajectories slice**). The +27.1pp gain is bounded by
   memorization. Writeup must add the explicit 100% / 220-trajectories
   figure beside the +27.1pp row.

Wave-2 verified one finding survives the audit cleanly: **brutal H2
is the headline that stands.** 0/24 brutal task slugs appear in either
training file; 66 "brutal" string hits in train_mix.jsonl are all
ultrachat prose. The +22.2pp / 9× claim is defensible at N=3;
re-running at N>=8 is the only thing that lifts it to ADR-004 standing.
