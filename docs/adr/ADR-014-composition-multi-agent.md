---
doc_id: adr-014-composition-multi-agent
title: 'ADR-014: Compositions — multi-agent architectures & LLM pipelines'
zone: lab
kind: adr
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, adr, research-agent, multi-agent, pipeline, composition, runtime]
---
# ADR-014: Compositions — multi-agent & LLM pipelines

Status: accepted
Date: 2026-06-14
Deciders: Matt Rachlin

## Context
The lab reasons about a **model** (scoreboard model-grain) or a **model+scaffold**
(agent-config grain), and ADR-012 defines a single-agent runtime (LAR) + a signed
Agent Spec. We are expanding the domain to **multi-agent architectures and LLM
pipelines** and the harness that comes with them.

A multi-agent system or an LLM pipeline is a **composition**: a topology of nodes
(agents / LLM-calls / tools / routers / reducers) joined by typed edges (data +
control flow). Almost everything already built lifts from "single node" to "graph
of nodes"; the genuinely new work is the harness for the **edges** and the **whole**
— attribution, inter-node fidelity, compounding reliability/cost, multiplied safety
surface, termination, and emergent failure modes that a single agent cannot exhibit.

Two existing paths are already proto-compositions: the `lab-rag` retrieve->generate
flow and the `plan_execute` scaffold.

## Decision
Add **composition** as a first-class grain alongside model and agent-config, built
ON ADR-012 (each agent node IS a LAR) and ADR-013 (authorization per node + over
the composition). Reuse the trust ladder, scoreboard axes, control substrate, and
Tempo span tree; add the composition spec, an orchestration runtime, and the new
harness below.

### 1. The agent-as-tool primitive (keeps most of this cheap)
**An agent is a tool whose implementation is another agent.** Under the ADR-012
Tool ABI, an orchestrator calling a worker is a tool call whose backend is another
LAR. Consequences:
- **Orchestrator-worker multi-agent** ≈ LAR + agent-as-tool — reuses runtime,
  authorization, audit, budgets with minimal new machinery.
- **LLM pipelines** ≈ a typed DAG of LAR/LLM nodes with schema contracts on edges.
- Only **peer / cyclic** topologies (debate, reflect-revise, blackboard) need a new
  orchestrator (message-passing, shared state, termination).

### 2. Three composition classes (scoped by harness cost)
- **Pipeline** — acyclic, mostly deterministic (retrieve->rank->generate->verify).
  New: typed edges, per-edge schema contracts, per-node retry/fallback.
- **Orchestrator-worker** — dynamic tree (planner spawns specialists). New:
  agent-as-tool, hierarchical budgets.
- **Peer / cyclic** — graph with emergent dynamics (debate, multi-critic). New: a
  real orchestrator + termination/convergence control.

### 3. The Composition Spec (the deployable artifact)
A signed, versioned manifest = a **graph of Agent Specs** (ADR-012) + typed edges +
an orchestration policy:
```
{ composition_id, nodes[: Agent Spec ref | llm-call | tool | router | reducer],
  edges[: from, to, payload_schema], orchestration (class + routing + termination),
  budget (composition-level, split across nodes), authorization (union policy),
  runtime_version, eval_evidence_ref }
```
Ed25519-signed. A deployed composition = the **orchestration runtime** + a
Composition Spec. The `deployable` tier (ADR-012) applies to compositions on a
per-(Composition Spec, workflow) basis.

### 4. The orchestration runtime
Executes the graph over LAR nodes: routing, fan-out/fan-in, message passing,
shared state/blackboard, and termination. Records the full call graph on Tempo
spans (span tree == composition structure) and every node action on the
`record_action` audit chain. Hierarchical: the composition budget + kill switch
bound all children. Phased — pipeline executor + agent-as-tool first; the
peer/cyclic orchestrator only when a workflow needs it.

### 5. The new harness (what compositions require that single agents do not)
- **Attribution / credit assignment** — which node caused an end-to-end
  pass/fail. Per-node tracing PLUS counterfactual ablation (swap a node for an
  oracle/stub; does the score recover?). Without it a composition score is
  uninterpretable. Highest-value new capability.
- **Inter-node protocol fidelity + error propagation** — an edge validity gate
  (the request-fidelity analogue for edges): did node A hand node B what its
  contract promised? And does a bad upstream output get CAUGHT vs. AMPLIFIED
  downstream.
- **Compounding reliability** — k nodes at 0.9 each => 0.9^k. Extend the ADR-004
  pass^k discipline to compositions; the reliability axis reports composition-level
  + per-node.
- **Hierarchical cost/latency budgets** — multi-agent multiplies calls; loops can
  detonate budgets. **Promote cost to a gated scoreboard axis** (not just
  reported), attributable per node, enforced as a composition budget split.
- **Multiplied safety surface + inter-agent injection** — more nodes touching
  tools; one agent's output is another's input (an injection vector between
  agents); a write-capable sub-agent inside a read-only composition. Authorization
  is per-node AND composition-aware (gate on the UNION of node capabilities).
- **Determinism & replay** — seeded routing, recorded decision points, replayable
  graph traces.
- **Termination & loop control** — bounded loops, convergence detection, deadlock/
  livelock (ping-pong) detection for cyclic topologies.
- **Emergent-failure evaluators** — new failure classes: collusion (agreeing on a
  wrong answer), sycophancy cascades, error amplification, debate mode-collapse.
- **Process eval** — grade intermediate node outputs, not only the final outcome
  (a good plan with a bad final answer is diagnosable).

### 6. Extensions to existing ADRs (not rewrites)
- **ADR-009 scoreboard:** composition entity grain; promote **cost** to a gated
  axis; add the attribution view.
- **ADR-008 trust ladder:** compositions reach `verified`; add the edge/protocol-
  fidelity validity gate.
- **ADR-013 authorization:** per-node grants + composition-union policy; inter-agent
  injection as an explicit threat class.
- **ADR-003 task taxonomy:** add composition tasks.
- **Evaluators (#18 family):** protocol_fidelity, error_propagation, termination,
  collusion/amplification + process-eval hooks.

### 7. Phasing (let a real workflow pull the machinery)
Do NOT build a general orchestrator speculatively. Build the **pipeline +
agent-as-tool** layer first (reuses ~all of ADR-012/013), pulled by **NS-3 Research
Synthesizer** going multi-stage (search->read->synthesize->verify) — read-only, so
attribution + protocol-fidelity are exercised at low risk. **NS-2 Code Maintainer**
(plan->edit->test->review) then exercises orchestrator-worker + hierarchical budgets
+ per-node write authz. The peer/cyclic orchestrator waits until a workflow
genuinely needs debate/reflection.

## Consequences
- Multi-agent and pipelines become a measurable, governable grain — not an
  unmeasured capability bolted on the side.
- The agent-as-tool primitive means orchestrator-worker and pipelines reuse the
  runtime/authz/audit/budget substrate; only peer/cyclic adds real new runtime.
- Attribution makes composition scores interpretable and debuggable.
- COST: an orchestration runtime + composition trace/attribution tooling is real
  engineering; mitigated by phasing (pipeline first) and the pull-by-workflow rule.
- COST: scoreboard/trust/authz schemas gain a composition grain (entity, edges,
  per-node attribution).
- RISK: orchestration-framework gold-plating — mitigated by the pull-by-workflow
  discipline and refusing the peer/cyclic orchestrator until earned.

## Considered alternatives
- **Treat multi-agent as just a bigger single agent (one transcript)** — rejected;
  loses attribution, per-node authz, and inter-node fidelity — the whole point.
- **Adopt a multi-agent framework (AutoGen/CrewAI/LangGraph) wholesale** — rejected
  for v0; local-first + the bespoke trust/control/authz substrate; wrapping a
  framework forfeits attribution + audit integration. Revisit for the peer/cyclic
  orchestrator only.
- **Build the general peer/cyclic orchestrator first** — rejected; highest
  complexity, no workflow needs it yet; pipelines + agent-as-tool cover NS-1..3.
- **Composition as a separate registry, not on the trust ladder** — rejected;
  re-verification + provenance want one hash-chained lineage (ADR-008).

## Relationships
Composes ADR-012 (each node is a LAR; Composition Spec = graph of Agent Specs) and
ADR-013 (per-node + union authorization). Extends ADR-003 (task taxonomy), ADR-008
(trust ladder + edge fidelity gate), ADR-009 (composition grain + cost axis), and
the #18 evaluator family. First exercised by the north-star NS-3 (pipeline) and
NS-2 (orchestrator-worker).
