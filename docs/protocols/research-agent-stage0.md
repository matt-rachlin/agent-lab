---
doc_id: research-agent-stage0
title: 'Spec: Research-agent program — Stage 0 (trust + control substrate)'
zone: lab
kind: guide
status: active
owner: m
created: '2026-06-13'
last_updated: '2026-06-13'
last_verified: '2026-06-13'
tags:
- lab
- guide
- research-agent
- spec
---
# Spec: Research-agent program — Stage 0 (trust + control substrate)

## Program in one paragraph

Build a **gated-copilot research agent** that performs ongoing research and
experimentation to grow the lab, toward building **capable, reliable,
safe/controlled agents for all manner of tasks**. The agent proposes and
validates; building is done by a human with Claude. Autonomy is **earned**: the
agent starts fully human-gated and widens its envelope per experiment-class as
its verifier proves reliable. Compute is **local-first** (llama-swap, single
serialized GPU); cloud is used minimally as anchors/judges/verifiers only.

Stages: **(0) trust + control substrate** [this spec] -> (1) the objective
scoreboard (capable/reliable/safe suites) -> (2) the gated propose->run->verify->
writeup loop -> (3) autonomy widening under trust tiers + budgets.

## Why Stage 0 first

You cannot let an agent optimise toward a goal it cannot measure, act on results
it cannot trust, or take actions you cannot bound. Stage 0 builds the substrate
for all three **before** any automation. It ships no agent.

## Scope correction (from review)

Two facts shape this spec and override the naive version:
1. **Trust is per-cell, not per-evaluator.** ~22% of `experiment_runs` have no
   `eval_results` row (the whole agent path writes runs+`agent_logs` only), and
   ~1.7k runs carry multiple evaluators. So `trust_level` lives on
   `experiment_runs`; validity evidence is decomposed per-evaluator.
2. **There is no evaluator base class to hang a contract on.** Evaluators are
   bare decorated `fn(RunRow, TaskRow)`; BFCL is graded *inline in the runner*
   (`_persist_bfcl_eval_result`), bypassing the registry; the agent path scores
   from the Inspect trajectory. So validity gates live in the **eval execution
   paths** (`_execute_single_turn`, `_execute_bfcl_cell`, `_execute_agent_cell`)
   and operate on the **trace**, not on an evaluator method.

## Deliverables

### D1 — Eval-validity gate contract (per eval *path*)
Generalises the [F-017](../findings/F-017-bfcl-toolchoice-artefact.md) fix to all
three eval paths. A run reaching `validity_passed` must satisfy:
- **Request fidelity** — full request (messages + tools + tool_choice + sampling
  params) persisted in every trace. BFCL done; single-turn and agent paths are
  separate edits + a round-trip test each.
- **Preconditions, fail-loud** — declared per suite/path (e.g. "tools were
  actually passed"); violation marks the run `error`, never a silent 0.
- **Emission/correctness decomposition** — "valid-form response" vs "correct"
  reported separately; a decode/format failure is never a capability miss.
- **Contamination check** — wire the existing
  [contamination-check](contamination-check.md) protocol in as a gate, not a
  side-doc.
- **Judge integrity** — for judged suites, require agreement-vs-gold per
  [judge-calibration](judge-calibration.md) before the judge's score counts.
- **Template/tokenizer round-trip** — assert the chat template + stop tokens the
  model needs were actually applied (a dropped template silently tanks scores).
- **Baseline sanity** — each suite declares expected ranges; aggregates deviating
  implausibly (in *either* direction) are flagged, never auto-passed.
- **Telemetry integrity** — diagnostic fields mean what they say (the F-017
  `unclear` label); covered by a test.
- **Construct-validity note** — each suite states what capability the metric
  actually measures (BFCL = call-format + args, not task-solving), so a valid
  number is not over-claimed.

### D2 — Result-trust lifecycle (implements ADR-008)
- `trust_level` on **`experiment_runs`** (enum: raw/validity_passed/
  reliability_confirmed/verification_attempted/verified/finding) + boolean flags
  `pre_registered` and `legacy`
  + an **append-only** `trust_transitions` table keyed on `run_id`
  (level, actor, ts, evidence pointer; hash-chained; no UPDATE/DELETE to agent role).
- **Migration + backfill (explicit):** new column NULL-default; backfill all
  ~17.9k existing runs to `raw` with `legacy=true` (NOT `validity_passed` — none
  cleared a gate that did not exist). Runs with no `experiment_id` (ad-hoc/agent
  runs) are inherently exploratory, `pre_registered=false`, capped at `validity_passed`. Existing F-NNN findings keep `min_trust_seen = legacy`; the
  ladder is not claimed retroactively.
- **Finding<->run link:** add `source_run_id` + `min_trust_seen` to `findings`;
  `lab finding new` refuses unless the referenced run is `verified`. (Today
  `finding new` is prose-only with no run link — this is the enforcement point.)
- `lab analyze report` becomes trust-aware: surfaces results by level, refuses to
  print an unlevelled headline number. (Depends on the schema landing first;
  extends the emission/correctness table already in `fix/bfcl-harness-toolchoice`.)
- **Promotion authz:** `verified`/`finding` transitions require the Bridge
  human-action channel; the agent runtime identity cannot invoke them.

### D3 — Adversarial verifier (skeleton, with a defined battery)
Tries to **refute** a `reliability_confirmed` candidate. The **minimum battery**
(below which a result is only `verification_attempted`):
- re-run with >=N perturbed seeds AND >=M prompt variants (reuse F-013 robustness),
- independent re-grade (second checker impl or independent output extraction) to
  catch harness-specific artefacts,
- **class-spanning anchor**: the anchor set must include >=1 model of each class
  under test (e.g. >=1 reasoning + >=1 non-reasoning). A *class-correlated* local
  outlier with the anchor in range is `artefact-suspected`, NOT a model deficit —
  this is exactly the F-017 pattern a single-anchor check would have missed.
- Emits `holds | artefact-suspected | inconclusive` + evidence; human reviews.

### D4 — Cloud-minimal policy + budgets
- Cloud is never system-under-test — only anchor / judge / verifier, at
  checkpoints, out of the routine sweep hot-loop.
- **Budgets are global and enforced at the gateway (litellm), with a hard stop:**
  daily $ + token + GPU-hour caps, not merely a per-experiment call count (a loop
  spawning many small experiments must not bypass the cap). Breach halts the
  agent (ties to D5 kill switch). Per-experiment `cloud_calls` is a sub-limit.
  **A gateway-only cap is bypassable** by direct llama-swap/network access, so it
  depends on D5's egress restriction to actually bind.

### D5 — Action-control substrate (NEW; the safety floor)
Result-trust governs beliefs; D5 governs *actions*. Present from Stage 0 even
though no agent ships, so nothing built later inherits an uncontrolled actor:
- **Least-privilege identity:** the agent runs under an identity with **no** git
  push/publish credentials, no cloud-write secrets, no prod-DB write grant, and **no network egress except via the litellm gateway**
  (so the D4 budget hard-stop cannot be bypassed) in its environment.
- **Deny-by-default capability allowlist:** outward/irreversible actions (git
  push, publish, external network, schema/prod writes) require explicit human
  approval each time. (This session's public-repo push is the canonical thing
  this must stop.)
- **Append-only audit log** of every agent-initiated action (intent, args,
  approver, outcome), write-restricted from the agent role (same integrity as
  `trust_transitions`).
- **Kill switch:** a single control halts the agent and releases its GPU lease
  (the existing `gpu_lease`) and budget holds; tested.

## Pre-registration gate (cross-cutting)
The lab already has `lab exp register` (plan + git SHA). Make it the floor of the
ladder: a run with no pre-registered hypothesis/analysis-plan/planned-N can reach
at most `validity_passed` and is marked exploratory. Multiplicity (FDR) is tracked
per campaign for any `verified` promotion.

## Acceptance criteria (Stage 0 done when)

*Result-trust*
1. All three eval paths persist the full request; each has a round-trip test.
2. `trust_level` enforced on `experiment_runs`; a `finding` can only be minted
   from a `verified` run, via the human-only channel.
3. **Substrate self-test (no Goodhart):** re-introduce the F-017 `tool_choice=auto`
   artefact behind a flag AND include >=1 *blind* artefact class the gate authors
   did not design against; the validity gate refuses `validity_passed` and the
   verifier returns `artefact-suspected` for both. A class with zero positive test
   cases is reported as "unproven."
4a. Held-out split is enforced as part of `reliability_confirmed`.
4b. The minimum refutation battery is defined; below it the top mintable level is
   `verification_attempted`, not `verified`.

*Control*
5. All agent-initiated actions are append-only audit-logged.
6. Kill switch halts a (stub) agent and releases leases; tested.
7. **Outward-action regression:** an attempted git push / publish by the agent
   identity is denied without approval (the public-repo-push scenario), as a test.
8. Global cloud/GPU budget enforced with a hard stop; tested.

## Non-goals (later stages)
The proposer/research loop (Stage 2); the capable/reliable/safe eval suites
(Stage 1); any autonomy widening (Stage 3); the full action-authorization model
(autonomy ADR). Stage 0 ships primitives, not policy.

## Effort & sequencing (honest)
Bigger than "add a column." Touches lab-eval, lab-sweep, lab-inspect, lab-cli, DB.
Split:
- **0a — substrate:** `trust_level` on `experiment_runs` + transitions table +
  migration/backfill; request-fidelity in all 3 paths; trust-aware `make_report`;
  finding<->run link.
- **0b — gates, verifier, control:** validity-gate enforcement in the 3 paths;
  verifier skeleton + battery; D5 action-control + audit + kill switch; budgets;
  the self-test (criterion 3). Estimate ~1.5-3 weeks total, dominated by
  multi-path enforcement and the framework/identity work, not the verifier.

## Open questions (resolved 2026-06-13 — see Resolved decisions)
- Minimum battery sizes (N seeds, M prompt variants, anchor count) for a
  meaningful `verified`.
- Hash-chaining vs signed transitions for tamper-evidence — which is worth it now.
- Exact least-privilege identity mechanism on m-box (separate user vs scoped
  token vs container).

## Resolved decisions (v0, 2026-06-13) — supersede the Open questions above

1. **Minimum refutation battery (D3), conservative v0 (tunable via verifier config):**
   `verified` requires UNANIMOUS survival of all of: >=16 seeds; >=5
   semantic-preserving prompt variants (F-013); >=2 independent re-grade paths
   that must agree; >=2 anchors per model-class under test. Any single failure ->
   `artefact-suspected`. Cost stays bounded — the battery runs only on candidate
   findings, not every cell.
2. **Tamper-evidence (D2 / ADR-008 §3):** hash-chained append-only
   `trust_transitions` (each row binds `prev_hash`) PLUS Ed25519-signed promotion
   rows for `verified`/`finding`, signing key held outside the agent's
   environment. Integrity + authenticity together enforce "the agent cannot mint
   its own findings".
3. **Agent identity isolation (D5), maximal v0:** rootless-podman container; a
   dedicated `lab_agent` DB role (SELECT broad; INSERT only up to
   `reliability_confirmed`; promotions via a SECURITY DEFINER function the role
   cannot execute); no git/cloud/signing credentials mounted; network egress
   allowlisted to the litellm gateway only.
