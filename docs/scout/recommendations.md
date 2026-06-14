---
doc_id: scout-recommendations
title: 'Scout: recommendation queue (for triage)'
zone: lab
kind: guide
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, guide, scout, recommendations]
---
# Scout recommendation queue

Each entry: **what · why-relevant-to-us · source (cited) · category · confidence · status**.
Categories: model | architecture | software | paper | method. Status: new | triaged | actioned | rejected.
Appended by the scout; triaged by Matt.

<!-- SCOUT-LOG (append below) -->

## Scan 2026-06-14 (on-demand, sources: arXiv/web/HF; verify=cite+fetch)

### R1 — [method/architecture] Design Patterns for Securing LLM Agents against Prompt Injections
- **What:** ETH Zurich taxonomy of 6 defense patterns (Tool Commitment, Least Privilege, Sandboxing, Dual-Layer input separation, Input Filtering, Output Validation) under defense-in-depth.
- **Why relevant to us:** maps 1:1 onto our action-control substrate + safety axis — Tool Commitment = our `tool_choice=required`/constraint suite; Least Privilege = `lab_agent` DB role; Sandboxing = the deferred #13 container cutover. Use it as the **spec for the injection_violations evaluator (#18) and the #13 cutover.**
- **Source:** https://arxiv.org/abs/2506.08837 (verified: fetched, v3, Beurer-Kellner/Tramèr et al.)
- **Category:** method/architecture · **Confidence:** high · **Status:** new

### R2 — [model] Qwen3.5-9B as a local tool-calling baseline candidate
- **What:** reported best small open tool-caller (~66.1% BFCL v4), 9B fits 12 GB.
- **Why relevant:** we baseline qwen3 4b/8b/14b — 3.5-9b may dominate the capability axis at our VRAM ceiling. Candidate for the next scoreboard baseline.
- **Source:** https://www.xda-developers.com/biggest-local-llm-machine-useless-cant-call-single-tool-how-many-parameters/
- **Category:** model · **Confidence:** medium (claim not independently verified; check HF for exact repo/quant) · **Status:** new

### R3 — [method] Perturbation-based contamination detection (ConStat / CoDeC)
- **What:** detect benchmark contamination via performance gap on rephrased/perturbed items (ConStat) + model-agnostic CoDeC; watermarking benchmarks pre-release.
- **Why relevant:** our `contamination_signal` IS the rephrased-gap idea — ground/strengthen it with ConStat; reuses the verifier variant arm.
- **Source:** https://llm-stats.com/blog/research/what-is-a-contaminated-llm ; https://openreview.net/forum?id=WFGxFzFDmQ
- **Category:** method · **Confidence:** medium · **Status:** new

### R4 — [method/model] Hammer: function masking for on-device function calling
- **What:** robust function-calling for small/on-device models via function masking.
- **Why relevant:** directly our niche (12 GB tool-calling, the qwen3-4b-ft); technique + models worth evaluating.
- **Source:** https://arxiv.org/abs/2410.04587
- **Category:** method/model · **Confidence:** medium · **Status:** new

### R5 — [benchmark/method] Terminal-Bench 2.0 + the scaffold-dominates finding
- **What:** TB 2.0 = 89 human-validated agentic terminal tasks; reported that harness/scaffold changes ALONE moved one model 52.8%->66.5% (Top30->Top5).
- **Why relevant:** we're wiring harbor/terminal-bench (D4) + have react/plan_execute scaffolds; the scaffold-dominates result strongly argues to **prioritize scaffold experiments** (and corroborates F-013 prompt-sensitivity).
- **Source:** https://arxiv.org/html/2603.05344v1 ; https://github.com/sierra-research/tau2-bench
- **Category:** benchmark/method · **Confidence:** medium · **Status:** new

### R6 — [software] OpenHands / smolagents (open agent scaffolds)
- **What:** mature open-source agent frameworks (OpenHands eval'd on SWE-bench/Terminal-Bench; smolagents minimal).
- **Why relevant:** scaffold patterns to borrow into our Inspect-based react/plan_execute.
- **Source:** https://www.firecrawl.dev/blog/best-open-source-agent-frameworks
- **Category:** software · **Confidence:** medium · **Status:** new
