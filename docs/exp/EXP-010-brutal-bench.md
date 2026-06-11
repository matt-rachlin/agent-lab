---
doc_id: exp-010
title: 'EXP-010: BRUTAL-BENCH-001 — validation + first ranking on pbs-agent-brutal-v0.1
  (pre-registered)'
zone: lab
kind: exp
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
depends_on:
- kind: doc
  target: exp-009
- kind: doc
  target: pbs-agent-brutal-v0-1-card
- kind: doc
  target: f-012-agentic-tool-calling-failure-modes
- kind: doc
  target: f-013-prompt-robustness-model-property
tags:
- lab
- exp
- agentic
- tool-use
- pbs-agent-brutal-v0.1
---

# EXP-010: BRUTAL-BENCH-001 — validation + first ranking on the brutal suite

Date created: 2026-06-11
Status: planned
Pre-registered: <commit SHA filled by `lab exp register` at registration time>

## Question

pbs-agent-hard-v0.1 has ~2 tasks of headroom over gemma4-12b (0.938 in
EXP-008). Does the new brutal tier (24 tasks: debug loops, broken-path
recovery, long-horizon chains, adversarial-precision specs) restore
discriminating range at the top — and do the failure-mode predictions
from F-012/F-013 hold on task types the hard suite didn't contain
(reading failing test output; recovering from a wrong intermediate
step)? This is also the suite's validation gate: tasks were
machine-verified at authoring time, but turn-budget fairness and
spec clarity are only provable against live models.

## Hypothesis

- **H1 (headroom):** gemma4-12b pass@1 ≤ 0.85 on the brutal suite —
  i.e. the tier creates ≥ 3.5 tasks of discriminating range that the
  hard suite lacks.
- **H2 (ranking):** the EXP-008 ordering holds: gemma4-12b >
  qwen3-coder-30b > devstral-24b in pass@1.
- **H3 (solvability):** ≥ 90% of tasks (≥ 22/24) are passed by at
  least one of the three models. Tasks failed by all models are
  presumed suite defects until a trajectory audit clears them.
- **H4 (failure-mode prediction):** devstral-24b's worst category is
  `longhaul` (long-horizon weakness, F-012/F-013: its hard-suite multi
  score was 3/8), and qwen3-coder-30b underperforms its own overall
  mean in `debug` (its hard-suite weakness concentrated in `code`,
  4/8).

## Method

### Models

| litellm_id | role |
|---|---|
| gemma4-12b | subject — incumbent local coding agent |
| qwen3-coder-30b | subject |
| devstral-24b | subject |

### Matrix

- suite: pbs-agent-brutal-v0.1 (24 tasks, sealed at registration SHA)
- prompt: tool_use_system_v2 (all tasks reference it)
- config: react, temperature 0.0, top_p 1.0, max_tokens 4096
- seeds: [1] — **deliberate single-seed validation pass**, mirroring
  HARD-BENCH-001's role. Per ADR-004 these numbers are not reportable
  reliability claims; the N=8 run is gated on this experiment's H3
  verdict (no point burning 8 seeds on a defective suite). This
  staging (validate at N=1, confirm at N=8) is the explicit lab
  pattern going forward.
- 72 cells; est. 2–3 h on the gpu queue (queued behind HARD-BENCH-003).

### Metrics

pass@1 per model (overall + per category); per-task pass map across
models (for H3 and the defect audit list).

## Success / failure criteria

- H1: CONFIRMED iff gemma4-12b pass@1 ≤ 0.85; 0.85–0.92 →
  INCONCLUSIVE (headroom thin); > 0.92 REFUTED (tier failed, design a
  harder one).
- H2: CONFIRMED iff both inequalities hold strictly; any tie within
  1 task → INCONCLUSIVE for that pair.
- H3: CONFIRMED iff ≥ 22/24 tasks passed by ≥ 1 model. 20–21 →
  INCONCLUSIVE pending trajectory audit of the all-fail tasks; < 20
  REFUTED (suite revision required before any N=8 run).
- H4: CONFIRMED iff devstral's lowest category pass rate is `longhaul`
  (ties count) AND qwen3-coder's `debug` rate < its overall mean.

## Kill criteria

- Kill if > 10% of cells (≥ 8/72) terminate on harness error (sandbox,
  tool-server, infra timeout) rather than model behavior.
- Kill if a fixture/predicate defect is found mid-run (NXDOMAIN-class);
  fix the suite, bump the suite revision, restart the experiment —
  EXP-008's mid-run cell patching is explicitly not repeated under
  pre-registration.
- Abort if wall clock exceeds 3× estimate (queue contention).

## Analysis plan

One report: per-model overall + per-category table, per-task pass map,
H1–H4 verdicts regardless of direction, and the defect-audit list for
any all-models-fail task (trajectory inspection → either a finding
about a real shared weakness, or a task fix + suite revision bump).
N=8 follow-up (BRUTAL-BENCH-002) only if H3 holds.
