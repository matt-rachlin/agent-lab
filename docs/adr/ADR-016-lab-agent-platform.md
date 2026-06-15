---
doc_id: adr-016-lab-agent-platform
title: 'ADR-016: Lab Agent Platform — body kinds, planes, packaging'
zone: lab
kind: adr
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, adr, agent-platform, runtime, deployment, body-kinds]
---
# ADR-016: Lab Agent Platform

Status: accepted
Date: 2026-06-14
Deciders: Matt Rachlin (drafted in workspace session; second-decider review 2026-06-14 in lab session, see daily log)

## Context
ADR-012 names a "packaging target: an importable entrypoint plus a thin server
shim, so an external app invokes [a verified Agent Spec] like the cloud agent
it replaces." The packaging is named but not designed. ADR-014 adds Composition
Specs and an orchestration runtime; ADR-013 supplies authorization gates. None
of these are a runnable application — they are the pieces an application
composes. The charter's deployment tier requires that composition to exist.

A single "container as the deployable unit" framing was considered first and
rejected on examination: it collapses three concerns (inference, agent loop,
interface) that the lab already separates, presumes pattern-B (agent-as-brain)
when other interaction patterns are equally valid for the lab's intended
workloads, and requires GPU passthrough into every deployed unit when the
inference plane already multiplexes one GPU across many callers.

## Decision
Define the **Lab Agent Platform (LAP)**: a shared layer of platform primitives
and per-pattern **body templates** that compose ADR-012/013/014 into runnable
applications. Specifically:

### 1. Three planes + a body layer
LAP arranges existing lab infrastructure as three planes (inference, control,
observability) plus an agent plane (LAR + Tool ABI + orchestration runtime).
Each deployed application is a **body** on top of these planes. Bodies are
I/O processes; the GPU lives only in the inference plane. Multi-instance is
solved at the agent plane (bodies are light); GPU contention is solved at the
inference plane (existing queue + litellm).

### 2. Five interaction patterns (body kinds)
Every body declares a `body_kind`, one of:

- **service** — worker invoked by an app; per-request LAR
- **brain** — long-lived process whose main loop IS the LAR / orchestration
- **library** — importable Python package; no process; restricted to
  `read`/`external_read` tools
- **companion** — long-lived subscriber to a stream; emits notifications
- **environment** — REPL/CLI/TUI owning the loop AND the toolspace

`body_kind` is a new field on the Agent Spec / Composition Spec. The platform
refuses Specs whose `body_kind` is incompatible with declared tools or
authorization profile (e.g. a `library` Spec granting `irreversible` tools).

### 3. Bodies are not containers
Container packaging is a per-body hygiene choice, not architecture. Default
packaging is a systemd user unit. A body may also be a `podman compose`
service, a pip-installable library, or a `python -m`. The platform does not
require Docker/Podman; it does require every body to honour the lifecycle
contract (start / health / kill / budget update / audit emission).

### 4. Platform-provided primitives
LAP provides: Spec loader + Ed25519 verify, body registry (sqlite + Redis
pub/sub), CLI (`lap`), body templates (one per pattern), state-directory
convention (`~/db/agents/<body-name>/`), and the integration glue between
existing planes and bodies. LAP does NOT re-implement LAR (ADR-012), the
Tool ABI (ADR-012), the orchestration runtime (ADR-014), authorization
(ADR-013), or audit (`lab.core.control`) — it consumes them.

### 5. Body principal on the audit chain
Every `record_action` row carries BOTH the body process principal (which body
executed this) AND the Spec principal (which authorization profile permitted
it). The Spec is what is authorized; the body is what executed.

### 6. Packaging as a Python package
LAP lives in a new lab package, `lab-platform`, sibling to **both** `lab-core`
and `lab-agent` (not a child of either). Rationale (corrected from the draft):
LAR + Tool ABI + composition primitives **already live in `lab-core`**
(`lab.core.agent_runtime`, `lab.core.authz`, `lab.core.composition`); the
existing `lab-agent` package is the sandboxed tool backend (FastMCP-over-podman
`fs_read/write/grep`, `shell_exec`, `http_fetch`, `python_eval`, `kb_query`,
plus `ToolPool`/`Sandbox`). LAP's scope — body registry, `lap` CLI, Spec loader,
body templates, body-principal audit — is orthogonal to both. The "fold into
`lab-agent`" fallback is dropped: folding bodies into the sandboxed-tools
package would confuse layering. Body templates live under `lab-bodies/<pattern>/`.
The CLI binary is `lap`.

### 7. Phasing matches the plan
Phase A: LAR + Tool ABI consolidation. **Status as of 2026-06-14:** A1
(LAR core), A2 (Tool ABI in-process backend + ADR-013 authz tiers), and A4a
(scout migrated to LAR) **already shipped in `lab.core.agent_runtime`/
`lab.core.authz`** — multiple thin LAR-caller bodies already exist in
`lab-cli` (scout, synthesizer v0, analyst, maintainer, comms, ops). What
remains for Phase A: **A2b** (sandboxed Tool ABI dispatcher — the #13 seam
into `lab-agent`'s FastMCP backend) and **A3** (Inspect-solver → LAR adapter
+ parity test, pre-registered EXP-017 on `pbs-agent-hard-v0.1`). Phase B: LAP
core + CLI + a hello body. Phase C: pipeline orchestration runtime — a v0
linear pipeline already ships in `lab.core.composition`, Phase C is the
typed-edge/per-edge-schema/fan-out upgrade. Phase D: NS-3 Research Synthesiser
as the first real `brain` body, migrating from the existing v0 in
`lab-cli/synthesizer.py`. Phase E: prove kind-agnostic by lifting a `library`
body. Patterns D and E (companion, environment) deferred until a workflow
pulls them — same rule as ADR-014.

## Consequences
- A `deployable` agent has an unambiguous form: `(verified Spec) + (body kind)
  + (state directory) + (lifecycle hooks)`. External apps invoke it through
  its declared interface, audit + observability come for free.
- The platform is body-kind-agnostic: adding a new pattern is "add a template
  + a Spec field validator," not a re-architecture.
- GPU passthrough collapses to a single concern (the inference plane); bodies
  scale horizontally without GPU fights.
- Cost: a new package, a registry service, a CLI, five templates (three v0,
  two deferred). Each is small individually; the integration work is the
  bulk of effort.
- Cost: the Spec schema grows `body_kind`; the audit row grows a body
  principal. Both are additive, but every existing consumer must accept
  the wider rows.
- Risk: pattern-B's whole-app failure modes (loops, hallucination, budget
  blowouts). Mitigation: ADR-012 bounded LAR + ADR-013 authorization gates
  + global kill switch + per-Spec budgets — the same stack the steward
  already runs on, exercised by the synthesiser as the first vertical.

## Considered alternatives
- **Container-per-app as the deployable unit** — rejected; conflates inference
  / agent / interface; assumes pattern-B; requires GPU passthrough per body;
  blocks library/environment patterns.
- **One body kind (brain only) for v0** — rejected; the lab's intended
  workloads (service-style summarisers, library-style classifiers, companion-
  style observers) do not fit the brain shape and would each need a bespoke
  packaging path.
- **Fold LAP into `lab-agent` from day one** — rejected for v0 (boundary
  clarity while the abstraction is new); revisit after Phase B.
- **Adopt an off-the-shelf agent-deployment framework** — same reasoning as
  ADR-012/014 rejecting framework adoption (LangGraph / AutoGen / CrewAI):
  lab-specific control / trust / authz substrate; framework wrapping forfeits
  audit integration. Revisit only if LAP grows beyond its current scope.

## Relationships
Realises the packaging target named by **ADR-012** and the deployment tier
named by the charter. Consumes **ADR-013** (authorization), **ADR-014**
(composition runtime), **ADR-008/009** (trust ladder + scoreboard), the
existing `lab.core.control` substrate, and the litellm + llama-swap inference
plane (ADR-002).

The long-form design + implementation plan live in:
- `~/docs/specs/2026-06-14-lab-agent-platform-spec.md`
- `~/docs/plans/2026-06-14-lab-agent-platform-design.md`
- `~/docs/plans/2026-06-14-lab-agent-platform-plan.md`

First vertical (the Research Synthesiser, ADR-014's NS-3) lands in Phase D
of the plan as the platform's workflow-acceptance proof.
