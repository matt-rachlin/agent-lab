---
doc_id: lab-domains-map
title: 'Domain map: what the lab measures, organized by pillar, with maturity
  status per domain'
zone: lab
kind: guide
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
tags:
- lab
- domains
- map
- planning
---

# Domain map

The lab's organizing question: **can you trust a local agent — and how
would you know?** Every domain belongs to one of five pillars. Each
domain climbs a maturity ladder; a domain is not "real" until rung 3.

**Maturity ladder:**
1. **built** — tasks/runner exist, machine-verified at authoring
2. **validated** — proven end-to-end against a live model (usually a
   frontier cloud lane)
3. **benchmarked** — a pre-registered sweep across the model field
4. **finding** — a published F-doc with verdicts

**Rule: no new domain until two existing domains climb a rung.**

## Capability

| domain | asset | maturity | next step |
|---|---|---|---|
| agentic base | pbs-agent-v0.1 (12) | 4 — F-005, F-012 | canary duty (frozen) |
| agentic hard | pbs-agent-hard-v0.1 (32) + CARD | 4 — F-012/F-013/F-014 | EXP-009 N=8 verdicts |
| agentic brutal | pbs-agent-brutal-v0.1 (24) + CARD | 1 → 3 queued | EXP-010 verdicts |
| terminal (community) | Terminal-Bench 2.0 via Harbor + lab-react adapter | 2 (adapter validated 2/2) | gemma4 + lab-react runs queued |
| coding (community) | SWE-Bench via Harbor adapters | 0 (adapter available) | first SWE-Bench-lite run |
| SQL / data-eng | pbs-agent-sql-v0.1 (12) | 1 | first sweep + EXP |
| GUI / browser | /data/lab/gui runner + 6 tasks | 2 (glm 6/6) | local-model run queued; v1 difficulty |
| agentic RAG | pbs-agent-rag-v0.2 (14, citations) | building | verify + first sweep |
| dual-control | tau2-bench vendored | 2 (mock smoke) | local-model domain run; custom domain |
| cross-episode memory | /data/lab/memdom chains (5×3) | building | ablation validation |
| single-turn tool | BFCL v3 AST vendored (1000) | 4 — F-011 | FT-EVAL arm reuse |

## Reliability

| domain | asset | maturity | next step |
|---|---|---|---|
| seed variance | ADR-004 + pass^k + CIs | 4 — F-002 | standing discipline |
| prompt robustness | v1/v2 A/B method | 4 — F-013 | re-verify at N=8 (EXP-009) |
| tool-fault recovery | fault shim + pbs-agent-fault-v0.1 | building | smoke + first sweep |
| reasoning effort | EXP-012 + think-knob configs | 3 queued (smoke-gated) | verdicts |
| stack canary | nightly timer + CANARY sweeps | operational | watch history.csv |

## Safety & control

| domain | asset | maturity | next step |
|---|---|---|---|
| injection resistance | pbs-agent-inject-v0.1 (20) + compliance scanner | 1 | EXP-015 sweep (incl. ft model) |
| constraint compliance | pbs-agent-constraint-v0.1 (16) + scanner | 1 | first sweep |
| ask-vs-assume | /data/lab/askdom (12 + oracle) | building | glm validation |

## Meta-evaluation

| domain | asset | maturity | next step |
|---|---|---|---|
| trajectory audit | trajectory_audit.py (mech + LLM) | 2 (caught real shortcut) | standing post-sweep stage (auto-queued) |
| judge calibration | judge_calibration.py + owned ground truth | building | confusion matrices vs 3 judges |
| confidence calibration | tool_use_system_v3 + scanner | 1 | first v3-prompt sweep |
| injection/constraint scanners | per-domain compliance scanners | 1 | ride their sweeps |

## Training

| domain | asset | maturity | next step |
|---|---|---|---|
| SFT / RFT (eval→train→eval) | /data/lab/ft pipeline + EXP-013 | 3 queued (train + autopilot + 3 evals) | verdicts → writeup |
| agentic RL | Harbor rollout interfaces | 0 | GRPO loop after EXP-013 verdicts |

## Architecture & serving

| domain | asset | maturity | next step |
|---|---|---|---|
| scaffold comparison | react vs plan-exec (building); lab-react vs terminus-2 on TB-2.0 (queued) | building/queued | fixed-budget A/B EXP |
| efficiency archs | EXP-014 granite-vs-gpt-oss | 3 queued | verdicts |
| quant reproduction | Holo2 ScreenSpot Q4/Q8 | 3 queued | verdicts vs model card |
| cloud anchoring | HARD-BENCH-CLOUD method | 4 — F-014 | re-anchor per new suite |

## Deliberately out of scope

Harmful-content red-teaming (wrong tier for a public solo lab),
image/video generation (off-mission; feasibility documented in the
roadmap), voice/realtime agents (revisit with tau2's voice layer),
TheAgentCompany (disk-heavy; revisit after the current wave reaches
rung 3).
