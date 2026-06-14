---
doc_id: adr-013-action-authorization
title: 'ADR-013: Action authorization — capabilities, gating tiers, earned autonomy'
zone: lab
kind: adr
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, adr, research-agent, authorization, autonomy, safety]
---
# ADR-013: Action authorization

Status: proposed
Date: 2026-06-14
Deciders: Matt Rachlin

## Context
ADR-008 established the result-trust ladder and stated explicitly: **result-trust
is necessary-but-NOT-sufficient for ACTION** — *what an agent may DO* is a separate
axis, deferred to a future autonomy ADR. This is that ADR.

The mechanism already exists (migration 007, `lab.core.control`): a `lab_agent`
role, deny-by-default capability guards, a promotion guard (`lab_agent` may propose
up to `reliability_confirmed`, never mint `verified`), an append-only hash-chained
audit, a kill switch, budgets, and Ed25519 signing. What is missing is the
**policy model** that uses them.

The north stars need escalating stakes: NS-1 read-only; NS-2 mutates code
(reversible via git); NS-4 sends email (irreversible, external). Without an
authorization model we cannot safely run NS-2/NS-4. The motivating incident this
program began with — an agent that pushed to a public repo — is precisely an
unauthorized, irreversible, external action.

## Decision
Define **action authorization** as an axis distinct from result-trust: *what an
Agent Spec (ADR-012) is permitted to DO*, independent of how good its model is.

### 1. Capabilities & side-effect classes
Every tool (Tool ABI, ADR-012) carries a capability label and a **side-effect
class**:
- `read` — no mutation, no egress.
- `external_read` — network egress, no side effect (bounded by the egress
  allowlist / SSRF guard / #13 sandbox).
- `write_local` — reversible local mutation (e.g. `fs_write` inside a git-tracked
  workspace).
- `irreversible` — send / publish / delete / spend; no clean undo.

Capabilities are granted to an Agent Spec **explicitly, deny-by-default**.

### 2. Authorization tiers (the gate per capability)
For each granted capability the Spec declares a gate:
- **auto** — runtime executes without asking. Allowed for `read` / `external_read`,
  and for `write_local` only once *earned* (see 3).
- **human-approve** — runtime pauses, emits {proposed call, args, diff/preview} to
  a human channel (ntfy / Bridge), executes only on a signed approval; timeout ->
  deny. Default for `irreversible` and for un-earned `write_local`.
- **dry-run** — execute against a shadow (git stash/worktree, draft folder,
  no-send) and surface the would-be effect; never touches the real target.
- **deny** — refuse.

Default gate by class: `read`=auto, `external_read`=auto (within allowlist),
`write_local`=human-approve (-> earn auto), `irreversible`=human-approve
(**never auto in v0**).

### 3. Earned autonomy (the gated-copilot principle, made concrete)
A Spec's gate for a class may ratchet `human-approve -> auto` ONLY by a clean track
record: N consecutive approved actions of that class on its deployed workflow with
zero overrides/incidents, recorded on the audit chain, plus a human-signed
(Ed25519) promotion. Ratchets are per-(Agent Spec, capability class, workflow).
**Any incident — human override, kill, or safety veto — resets the ratchet.**
`irreversible` never auto-ratchets in v0.

### 4. Enforcement lives in the runtime, not the tool
Before dispatching any tool call, LAR (ADR-012) resolves: side-effect class -> gate
-> `{execute | approve-flow | dry-run | deny}`, and records the decision on the
audit chain regardless of outcome. The existing deny-by-default guard + kill switch
+ budget are the **floor**: a global stop overrides any grant. The egress allowlist
bounds `external_read`.

### 5. Provenance & reversibility
`write_local` runs in a git-tracked workspace/worktree, so every mutation is
diff-able and revertible. `irreversible` requires a recorded preview + human
signature. The audit chain links **action -> Agent Spec -> approving human** (or
`auto` + the ratchet evidence that earned it).

### 6. Relationship to result-trust (ADR-008)
Orthogonal axes; BOTH required to act. A `verified` model with no grant does
nothing; a granted capability on an unverified model is refused at deploy (the
deployment tier requires `verified`). Every real action is gated by
`(result-trust >= deployable) AND (authorization gate satisfied)`.

## Consequences
- NS-2 / NS-4 become safe to build: writes are reversible + previewed, sends are
  human-gated, everything audited.
- "Earn autonomy via trust tiers" (the locked program principle) gets a concrete,
  resettable mechanism — the path from copilot to wider autonomy.
- The repo-push incident class is structurally prevented (`irreversible` external
  is always human-approve in v0).
- COST: human-approve adds latency + a human-channel dependency; the approval loop
  (ntfy/Bridge -> signed approve) must be built — Phase E (NS-4) exercises it.
- COST: ratchet bookkeeping + reset rules add state + audit surface.
- RISK: approval fatigue -> mitigated by dry-run defaults and sensible `auto` for
  read-class capabilities.

## Considered alternatives
- **Fold authorization into result-trust (one axis)** — rejected; ADR-008
  deliberately separated them. A great model is not a license to act.
- **Static allowlists only (no earned autonomy)** — rejected; the copilot would
  stay fully manual forever, never reaching the program's autonomy-widening goal.
- **Per-call human approval for everything** — rejected; approval fatigue defeats
  the purpose. Tiers + ratchet are the balance.
- **Rely on OS/container permissions alone (the sandbox)** — rejected; the sandbox
  bounds blast radius but cannot express "send is human-gated, read is auto" —
  authorization is semantic and per-capability, above the sandbox.

## Relationships
Realizes the action axis deferred by ADR-008; enforced by the runtime defined in
**ADR-012**; consumes the existing control substrate (migration 007 `lab_agent`
role, guards, kill switch, budgets, `lab.core.signing`). Gates the high-stakes
north stars (NS-2 write, NS-4 send) and the deferred NS-5 ops agent.
