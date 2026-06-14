---
doc_id: adr-008-result-trust-lifecycle
title: 'ADR-008: Result-trust lifecycle (and why action-control is a separate axis)'
zone: lab
kind: adr
status: active
owner: m
created: '2026-06-13'
last_updated: '2026-06-13'
last_verified: '2026-06-13'
tags:
- lab
- adr
- research-agent
- trust
---
# ADR-008: Result-trust lifecycle

Status: proposed  (front-matter `status: active` tracks the doc; the decision is not yet accepted)
Date: 2026-06-13
Deciders: Matt Rachlin

## Context

We are building a gated-copilot research agent (see
[research-agent-stage0](../protocols/research-agent-stage0.md)). It will propose,
run, analyse, and write up experiments at machine speed, toward building capable,
reliable, safe/controlled agents.

Two distinct risks must be controlled, and conflating them is itself a design
error:

- **Epistemic risk** — confidently wrong *results* propagating into findings and
  decisions. [F-017](../findings/F-017-bfcl-toolchoice-artefact.md) is canonical:
  BFCL scored `phi-4-reasoning-plus` at ~1% vs ~45% true, a `tool_choice`
  artefact. It was **reproducible** — every seed agreed — so
  [ADR-004](ADR-004-reliability-discipline.md) (N≥8/pass^k) would have certified
  it. **Reliability is not validity.**
- **Operational risk** — the agent taking unsafe *actions* (git push, DB writes,
  cloud spend, network calls). This session a coding agent pushed to a *public*
  repo under a wrong visibility assumption — an irreversible outward action that
  no amount of result-trust would have prevented.

This ADR governs the first (result-trust). It establishes one load-bearing
principle about the second and defers its full design to the autonomy ADR.

## Decision

### 1. Results advance through an ordered, evidence-bearing ladder

```
raw -> validity_passed -> reliability_confirmed -> verification_attempted -> verified -> finding
```

| level | gate that admits it | meaning |
|---|---|---|
| `raw` | run completed | numbers exist; trust nothing |
| `validity_passed` | eval-validity gate contract (per eval *path*) | the measurement is of the model, not the harness |
| `reliability_confirmed` | ADR-004 (N≥8, pass^k, CI) **+ held-out split reproduces the effect** | stable across seeds *and* not an artifact of item selection |
| `verification_attempted` | the verifier ran but its minimum battery is not yet met | refutation attempted; not yet decision-grade |
| `verified` | the minimum refutation battery failed to break it | an independent, bounded attempt to refute it (perturbed seeds/prompts, independent re-grade, class-spanning anchor) did not |
| `finding` | **human approval** via a channel the agent identity cannot reach | promoted into the registry; may inform decisions and agent-building |

Rules:
- **No ladder level may be skipped.** Two **flags** gate the upper rungs,
  independent of the ladder: `pre_registered` (the run's experiment was
  registered via `lab exp register` *before* running) and FDR/multiplicity
  correction. **A result that is not `pre_registered` can reach at most
  `validity_passed`** — exploratory, usable to generate hypotheses, never to
  confirm them. Legacy/backfilled runs carry `legacy=true` and are likewise
  capped at `validity_passed`.
- **`finding` (trust_level) vs the findings table are distinct.**
  `trust_level='finding'` on a run means its result was promoted into the
  findings registry; the `findings` row carries its own `status`/`confidence`,
  linked by `source_run_id`. Two linked records, not one field.
- **`verified` is meaningless without a defined battery.** Until the
  adversarial-verifier minimum battery (Stage-0 D3) is specified and tested, the
  top mintable level is `verification_attempted`, not `verified`. Power of the
  refutation attempt is part of the definition.
- **Only `verified` may be promoted to `finding`,** human-approved, and the
  approval actor must be structurally distinct from the agent (see §3).
- **Consumers declare the minimum `trust_level` they rely on.** Default for any
  decision, leaderboard headline, or agent-building input is `finding`. Reports
  never show a sub-`finding` number without its level.
- **Multiplicity is accounted at the campaign level.** `verified` for an effect
  discovered across many comparisons requires false-discovery-rate correction
  for the number of comparisons the loop examined (tracked per experiment/
  campaign, not per result).
- **Trust is a property of the cell, not the evaluator.** `trust_level` lives on
  `experiment_runs` (every run has one; ~22% of runs have no `eval_results` row —
  e.g. the agent path). Validity *evidence* may be decomposed per-evaluator.
- **Every transition is append-only and provenanced** (level, actor, timestamp,
  evidence pointer); see §3.
- **Downgrade quarantines.** New evidence re-opens a result as a logged downgrade;
  a downgraded/refuted result is **quarantined** and cannot be re-promoted except
  by human review that references the downgrade reason (no launder-by-re-run). A
  downgrade must enumerate downstream consumers that declared dependence on it.

### 2. Result-trust is necessary but NOT sufficient for action

A `verified` (or even `finding`) result confers **zero** action permission. "An
agent may act on trust level X" is rejected as a model: action authorization is a
**separate axis** (capability + blast-radius), owned by the forthcoming autonomy
ADR. A perfectly verified finding still does not make `git push` to a public repo
safe. Stage 0 therefore ships *action-control primitives* (deny-by-default
capability gating, append-only audit, kill switch, least-privilege identity) even
though it ships no agent — see Stage-0 D5.

### 3. Authorization & integrity

- Promotion to `verified` and `finding` requires a credential the agent's runtime
  identity **does not hold** (operated via the Bridge human-action channel).
- The transitions log is **append-only and tamper-evident** (no UPDATE/DELETE
  grant to the agent role; rows hash-chained). The agent may *propose* transitions
  up to `reliability_confirmed`; `verified`/`finding` are human-minted.
- The `baselines` registry (expected ranges) is human-gated and sourced; the
  agent cannot widen its own pass ranges.

## Consequences

- **Easier**: no silent bad finding reaches a decision; autonomy can be tiered
  later against an explicit, auditable state rather than vibes.
- **Harder**: real process + compute per result (held-out splits, refutation
  battery, anchors). Stage 0 is weeks, not days (see spec effort note).
- **Risks & mitigations**:
  - *Rubber-stamp / Goodhart on gates.* Each gate is a runnable check with a
    regression test, and the test corpus must include ≥1 artefact class the gate
    authors did **not** design against — a gate with zero positive test cases for
    a class is "unproven," not "passing."
  - *Construct validity.* Gates assert the measurement is of the model (internal
    validity); whether the metric measures the *capability we claim* (BFCL scores
    call-format, not task-solving — F-017's deeper lesson) is a named, separate
    question on every suite.
  - *Verifier incompleteness.* It catches known artefact classes; the battery is a
    living document and every wild artefact adds a check.

## Considered alternatives

- **Reliability alone (ADR-004).** Rejected — F-017 was reliable and wrong.
- **Human review as the only gate.** Doesn't scale to agent throughput; humans
  rubber-stamp plausible writeups. Gates do the catching; humans adjudicate.
- **Trust_level as the autonomy axis.** Rejected (§2) — conflates epistemic trust
  with action permission; the public-repo-push incident is the counterexample.
- **Full autonomy + post-hoc audit.** Rejected for now — that is the BFCL problem
  at machine speed.
