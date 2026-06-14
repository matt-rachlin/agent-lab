---
doc_id: adr-012-agent-runtime-deployment
title: 'ADR-012: Agent runtime & deployment — one runtime, a signed agent spec, a deployment tier'
zone: lab
kind: adr
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, adr, research-agent, runtime, deployment]
---
# ADR-012: Agent runtime & deployment

Status: accepted
Date: 2026-06-14
Deciders: Matt Rachlin

## Context
The charter's unit of success is a **deployable local agent that drives real
software** >= the cloud/Claude agent it replaces. The lab today has the parts to
*measure and control* such an agent but not to *ship* one:

- **Two divergent agent loops.** The eval path uses the Inspect-bound solver
  (`lab-inspect`), sandbox-coupled. The scout (ADR-011) had to build a SEPARATE
  bounded loop on `call_litellm_chat` because the solver is not reusable headless.
  Two loops = two behaviours to verify; what we measure is not what we'd ship.
- **Two divergent tool surfaces.** `lab.agent.tools` (FastMCP-over-podman:
  `fs_read/write/grep`, `shell_exec`, `http_fetch`, `python_eval`, `kb_query`) is
  good, sandboxed and auditable — but only reachable *inside the eval sandbox*.
  The scout uses in-process callables instead. ADR-011 flagged this cutover.
- **No deployment artifact.** "verified" is a DB trust_level on `experiment_runs`,
  not a runnable thing. There is no way to take {verified model + scaffold + tools}
  and have an external app call it like it calls Claude today.
- **No deployment tier.** The charter says benchmarks are PROXIES and a deployment
  tier sits above measurement (ADR-009 scoreboard). It does not exist.

## Decision
Define the **Lab Agent Runtime (LAR)** — one agent execution core used identically
for eval and deployment — plus a signed **Agent Spec** (the deployable artifact)
and a **deployment tier** above the scoreboard.

### 1. One runtime core (`lab.agent.runtime`)
A single bounded tool-use loop that takes `(model, system, tools, budget,
control_ctx)`, drives via `call_litellm_chat`, and on every tool call records
through `lab.core.control.record_action`, honours the kill switch + budgets,
recovers text-encoded tool calls, and bounds turns. This **generalises the scout's
proven loop** into the shared core. The Inspect solver becomes a thin **adapter**
that hands eval tasks to LAR; the scout becomes a LAR caller. Eval and deploy then
run the SAME loop — parity by construction.

### 2. One tool interface (the Tool ABI)
A tool is a typed callable declaring: `name`, JSON schema, a **capability label**
(for ADR-013 authorization), a **side-effect class** (`read` / `write_local` /
`external_read` / `irreversible`), and an implementation reachable via one of two
backends behind the same ABI:
- **in-process** — fast; for trusted/read-only tools (what the scout uses today).
- **sandboxed** — FastMCP/podman `lab.agent.tools`; for write/shell/untrusted +
  bounded egress.

The runtime selects the backend by side-effect class + the agent's authorization
(ADR-013), **not by call site**. This ends the divergence: the scout can use the
sandboxed write tools; eval and deploy share one registry. (This is the #13 cutover,
generalised from a one-off into a capability.)

### 3. The Agent Spec (deployment artifact)
A signed, versioned manifest — the unit that gets verified, deployed, and
re-verified:
```
{ agent_id, model_ref (verified run/cohort), system_prompt,
  tool_grants (capability labels), authorization_profile (ADR-013),
  budget, runtime_version, eval_evidence_ref }
```
Ed25519-signed (reuse `lab.core.signing`). A **deployed agent = LAR + an Agent
Spec**. Packaging target: an importable entrypoint plus a thin server shim, so an
external app invokes it like the cloud agent it replaces.

### 4. The deployment tier (above `verified`)
A new standing above ADR-008 `verified`. An Agent Spec reaches **`deployable`**
for a workflow when BOTH hold:
- (a) its model clears the ADR-009 scoreboard tiers on the relevant axes, AND
- (b) it clears a **workflow acceptance battery** on its target north-star
  workflow: a success-rate floor on the workflow's golden tasks; a **head-to-head
  >= the cloud agent it replaces**; zero safety-veto violations; and a reliability
  floor (pass^k on the workflow).

`deployable` is per-(Agent Spec, workflow), recorded on the trust ladder as a new
evidence type (hash-chained, human-signed promotion). This is the charter's
"deployment tier above measurement."

### 5. Observability
A deployed agent emits the same trace/audit stream as eval (OTel -> Tempo;
`record_action` audit chain) and exposes runtime escalation hooks (human-approve,
abort) consumed by ADR-013.

### Phasing
Runtime core + Tool ABI first (program Phase A3), exercised by migrating the scout
onto it. Agent Spec + deployment tier validated by the first vertical, NS-1 (Lab
Analyst, read-only) in Phase B. Higher-risk specs (NS-2 write, NS-4 send) follow
once ADR-013 lands.

## Consequences
- **Eval/deploy parity:** what we measure is what we ship — one loop, one tool set.
- The scout immediately gains the sandboxed write tools; #13 becomes a generalised
  capability, not a special case.
- A clear, signable, **re-verifiable** deployment unit (supports the down-the-line
  re-verification/regression discipline as the cohort churns).
- The deployment tier turns "good enough to replace Claude in app X" into a
  measurable verdict.
- COST: the Inspect solver must become a LAR adapter — risk to existing eval
  reproducibility; mitigate with a parity test (same task, solver-vs-LAR ->
  identical grades) before cutover.
- COST: existing `lab.agent.tools` must declare capability + side-effect labels.
- COST: a new trust evidence type + schema (deployment tier) on the ladder.

## Considered alternatives
- **Keep two loops (eval vs deploy)** — rejected; the divergence already forced
  ADR-011's in-process workaround and will only compound.
- **Deploy = "export weights + a script" (no runtime)** — rejected; loses
  control/audit/authorization, cannot re-verify, cannot compare to the cloud agent.
- **Adopt an off-the-shelf agent framework (LangGraph/etc.) as the runtime** —
  rejected for v0; local-first + the bespoke control/trust substrate; wrapping a
  framework loses the audit/authz integration. Revisit only if LAR grows complex.
- **Deployment tier as a separate registry (not on the trust ladder)** — rejected;
  re-verification + provenance want one hash-chained lineage (ADR-008).

## Relationships
Builds on ADR-008 (result-trust ladder), ADR-009 (scoreboard axes/tiers),
ADR-010/011 (the scout, the first LAR caller). Pairs with **ADR-013** (action
authorization), which governs what a deployed Agent Spec may DO. Implements the
charter's deployment tier and the program's "unified agent runtime" gap.
