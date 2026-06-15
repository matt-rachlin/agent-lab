---
doc_id: adr-019-trust-lifecycle-db
title: 'ADR-019: Trust-lifecycle DB schema and migration ownership'
zone: lab
kind: adr
status: draft
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, adr, trust, database, schema, migrations]
---
# ADR-019: Trust-lifecycle DB schema and migration ownership

Status: proposed
Date: 2026-06-14
Deciders: Matt Rachlin

## Context

ADR-008 (result-trust lifecycle) defines a four-level trust ladder:
`raw → validity_passed → reliability_confirmed → verified`. It names the concept
but does not specify which module owns the DB schema, which migration introduced
it, or how future schema changes are gated.

Wave-1 architecture review (2026-06-14) found that `migration 006_result_trust.sql`
implements the schema and `lab.core.trust` + `lab.core.verifier` are the only
code surfaces that touch it, but this ownership is implicit — no ADR records it.
Without a formal ownership record, a future developer might add trust columns
via a migration that bypasses `lab.core.trust`, breaking the ladder invariants.

Current schema artifacts:
- `packages/lab-core/src/lab/migrations/006_result_trust.sql` — creates
  `trust_transitions` table, adds `trust_level` column to `experiment_runs`,
  adds `min_trust_seen` to `findings`.
- `packages/lab-core/src/lab/migrations/007_action_control.sql` — adds
  `agent_action_log`; grants `INSERT` on `trust_transitions` to `lab_agent` role;
  explicitly `REVOKE UPDATE, DELETE` on `trust_transitions` to enforce append-only.

## Decision

### Canonical surfaces

- **Schema**: `packages/lab-core/src/lab/migrations/006_result_trust.sql` (trust
  tables) + `007_action_control.sql` (append-only grants). These two migrations
  are the ground truth for the DB shape.
- **Write path**: `lab.core.trust` is the sole module permitted to insert rows into
  `trust_transitions`. No other module writes to this table directly.
- **Read/verify path**: `lab.core.verifier` is the sole module that evaluates
  whether a run meets the criteria to advance on the ladder. External callers
  invoke `verifier.promote(run_id, target_level)` only.
- **Agent write scope**: the `lab_agent` DB role may `INSERT` on
  `trust_transitions` and `agent_action_log` (granted by 007); it may not
  `UPDATE` or `DELETE` either table. This is the append-only audit guarantee.

### Gate for schema changes

Any change to `trust_transitions`, the `trust_level` column, or the `findings`
trust columns requires:
1. A new numbered migration SQL file in `packages/lab-core/src/lab/migrations/`.
2. An amendment to this ADR in the same PR, explaining the schema delta.
3. Reviewer sign-off confirming the change does not weaken the ladder invariants
   (specifically: no `UPDATE` grant on `trust_transitions` to any role; no
   removal of the `CHECK` constraint on `trust_level`).

### Trust level values

Canonical set (enforced by DB constraint in 006): `raw`, `validity_passed`,
`reliability_confirmed`, `verified`. Adding a new level requires amending ADR-008
AND this ADR in the same PR.

## Consequences

- Easier: a developer looking for where trust is stored or written has two clear
  entry points (`lab.core.trust`, `lab.core.verifier`) and two migration files.
- Harder: any future trust-adjacent schema change (e.g. per-finding trust audit)
  requires an ADR amendment, which adds process overhead.
- Risks: if `lab_agent`'s `INSERT` grant is accidentally broadened to `UPDATE` in
  a future migration, the append-only audit property is silently broken. The
  amendment gate is the only safeguard.

## Considered alternatives

- **Trust schema owned by `lab-eval`** — rejected: trust transitions are needed
  by the verifier before eval results exist; `lab-core` is the correct layer.
- **Application-level trust table in Valkey** — rejected: Valkey has no durable
  write log, violating ADR-008's auditability requirement.
