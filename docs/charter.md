---
doc_id: lab-charter
title: 'Lab charter — what the lab is for'
zone: lab
kind: guide
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, guide, charter]
---
# Lab charter

## Mission
The lab is an **agent factory**: build **capable, reliable, safe/controlled LOCAL
agents that drive real software workflows**, deployable across **all of Matt's
personal and professional domains** — anywhere an agent can do the work.

Near-term driver: **replace cloud/Claude-driven agents in our own applications
with local agents.** Illustrative example (not the only target): an app whose
workflow is currently driven by a Claude agent — the goal is a local agent that
drives it at least as well. Generalise the pattern to any app / workflow / domain
(coding, ops, creative pipelines, personal tasks, professional & client work)
where an agent drives the work.

## Why local
Cost, control, privacy, ownership, no vendor lock-in, and the freedom to run
agents wherever and as often as we want.

## Unit of success
A **deployable local agent**: drives a real workflow at least as well as the
cloud agent it replaces, AND clears the lab's bar — capable + reliable +
safe/controlled (the scoreboard tiers, ADR-009). "Deployable" = ready to drive a
real app, not merely scoring on a benchmark.

## How the machinery serves the mission
Everything exists to produce deployable agents we can trust:
- trust + control substrate (ADR-008) — trust an agent's evals; bound its actions.
- objective scoreboard (ADR-009) — "good enough to deploy" is measurable, not vibes.
- eval suites (capability / reliability / safety) — proxies for real-workflow readiness.
- the scout (ADR-010) — find external models/methods/agents that get us there faster.

## Implications (carry into the roadmap)
- Benchmarks are **proxies**; the real target is **driving actual software**. Lab
  suites should increasingly mirror the workflows we deploy into.
- A **deployment tier** sits above measurement: an agent driving a real app.
- **Integration is first-class**: agents must plug into externally-developed
  software, not just run inside the eval harness (a deploy/runtime/adapter path).
