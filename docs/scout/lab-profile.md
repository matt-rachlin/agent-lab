---
doc_id: scout-lab-profile
title: 'Scout: lab profile (relevance filter for the research-scout agent)'
zone: lab
kind: guide
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, guide, scout]
---
# Lab profile — what the scout matches against

Hand-curated, top-level. The scout reads THIS + auto-pulls titles/TL;DRs from
docs/adr/*, docs/findings/*, SETUP.md for currency. (The `kb` knowledgebase is
the test corpus and is OFF-LIMITS as a source.)

## Goal
Build **capable, reliable, safe/controlled AI agents for all manner of tasks**,
run **locally**. A trust+control substrate (ADR-008) + an objective scoreboard
(ADR-009) are built; a research/scout agent is next.

## Hard constraints (shape what's relevant)
- **Single 12 GB GPU** (RTX 3080 Ti on m-box) — models must fit (≈≤14B q4, or MoE
  with small active params). 24B+ dense is out unless quantised hard.
- **Local-first**: llama.cpp / llama-swap / Ollama serving; LiteLLM gateway.
  Cloud only as eval anchors/judges (minimal).
- Python 3.13, torch cu130. No vLLM (VRAM) for now.

## Current stack
- Serving: llama-swap (VRAM-aware multi-model), Ollama, LiteLLM (:4000).
- Eval: own framework (suites bfcl-v3-ast, pbs-agent-*, tau2-bench, harbor),
  evaluators (bfcl_ast_match, llm_judge, constraint_violations, ...), pass^k /
  N≥8 reliability (ADR-004), trust lifecycle + adversarial verifier (ADR-008),
  scoreboard (ADR-009). Tracking: MLflow + Postgres.
- Scaffolds: single_turn, react, plan_execute (agent path via Inspect).
- Models in rotation: qwen3 (4b/8b/14b/30b-a3b-moe), qwen3-4b-ft-toolcall (QLoRA),
  gemma4-12b, phi-4-reasoning-14b, nemotron-nano-2 / 3-nano, gpt-oss-20b, glm-5.1.

## Active interests (HIGH relevance)
- Small/local models strong at **tool-calling / function-calling** (BFCL-style),
  and **agentic** tasks; MoE with small active params for 12 GB.
- **Reasoning models + tool use** (the tool_choice/emission issue, F-017).
- **Eval validity / trust / verification** methods; contamination, judge
  calibration, reliability (pass^k), benchmark-gaming defenses.
- **Agent frameworks / scaffolds**, terminal/agentic benchmarks (tau2, terminal-bench).
- **QLoRA / small-model fine-tuning** for tool-use; **RAG + reranking** (local).
- **Safe/controlled agents**: constraint compliance, prompt-injection resistance,
  sandboxing, trust tiers, action gating.

## Lower relevance / skip
- Frontier-only / huge-model results not runnable locally (note as anchors only).
- Image/video/audio generation (separate hobby projects, not the agent lab).
- Pure infra/devops unrelated to local model serving or eval.

## Known findings to not re-surface as "new"
- Reranker for our RAG: tested + refuted (F-007).
- BFCL tool_choice=auto artefact on reasoning models: known + fixed (F-017).
- Prompt robustness is a model property; single seeds lie (F-013/F-016).
