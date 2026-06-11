---
doc_id: lab-roadmap-2026-06
title: 'Lab roadmap 2026-06: from agent evals to the full eval-train-eval loop
  (deep-research synthesis)'
zone: lab
kind: guide
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
tags:
- lab
- roadmap
- planning
- agentic
- fine-tuning
- modalities
---

# Lab roadmap 2026-06 — what to build next

Synthesized 2026-06-11 from a verified deep-research pass (25/25 claims
survived 3-vote adversarial verification against primary sources) plus
two targeted follow-ups on 12 GB fine-tuning and non-LLM modalities.
Hardware frame: one RTX 3080 Ti (12 GB, Ampere — bf16 yes, FP8 no),
64 GB RAM, local-first stack (ollama/llama.cpp, litellm, Podman+gVisor,
MLflow/OTel, postgres/MinIO).

## Where the field is (verified)

- Agent evaluation has consolidated around **containerized environments
  with machine-verified end-state checks** — the discipline this lab
  already practices. The 2025–26 reference benchmarks are explicitly
  homelab-feasible: **Terminal-Bench 2.0** (89 hard tasks, Apache-2.0,
  Docker+Python harness, frontier < 65%), **tau2-bench** (LiteLLM
  provider-agnostic dual-control user-sim), **TheAgentCompany**
  (Docker-Compose simulated software company, ~30 GB disk, CPU-class;
  best agent 30.3% full completion as of Sep 2025).
- **Harbor** (Terminal-Bench's official harness) runs arbitrary custom
  agents next to Claude Code/OpenHands/Codex CLI, adapts 20+ external
  benchmarks (SWE-Bench, Aider Polyglot), and exposes **RL/SFT rollout
  interfaces** — one framework bridging eval and agentic training.
- Princeton **HAL** (~$40k for full-coverage benchmarking) validates the
  vendor-curated-subsets strategy, and contributes two cheap method
  upgrades: reasoning-effort is non-monotonic (reduced accuracy in
  21/36 settings — sweep it), and **LLM-aided trajectory inspection
  catches cheating that end-state scoring misses**.
- Agentic RL is now a named field (TMLR 2026 survey: POMDP framing, six
  capabilities). **SWE-RL** showed a no-execution rule-based reward
  (patch similarity) scales; ToolRL / Nemotron Tool-N1 showed
  **GRPO with verifiable tool-call rewards beats SFT-only by ~15pp** on
  tool benchmarks. Caveat from the skeptic literature: RLVR mostly
  reweights existing capability — size expectations accordingly.
- 12 GB fine-tuning is fully practical: QLoRA SFT to 7–8B comfortably
  (14B edge); GRPO ceiling ~4B (Qwen3-4B ≈ 10–13 GB with unsloth's
  vLLM standby). Default stack: **unsloth wrapping TRL** (torchtune
  discontinued). License-clean tool-use data exists (xLAM-60k cc-by-4.0,
  ToolACE + Hermes-FC apache-2.0). Training on the lab's own
  verified-successful trajectories is a named recipe (STaR/RFT;
  ReST-meets-ReAct). Export pitfall: **Ollama ignores GGUF-embedded
  chat templates** — Modelfile template must replicate tool-call
  handling.
- Modalities at 12 GB: STT solved (faster-whisper int8 ≈ 2.9 GB;
  Parakeet/Canary), TTS rich and fast-moving (Kokoro/Chatterbox/
  Qwen3-TTS, all Apache/MIT-class), VLM grounding usable locally
  (**Holo2-8B** Apache-2.0, ScreenSpot-Pro 58.9; GGUF Q4 ≈ 5 GB),
  docs/OCR strong (olmOCR-2, dots.ocr, MiniCPM-V 4.5), image gen fits
  via GGUF DiTs (Flux Q6_K 9.86 GB; Z-Image-Turbo native), video gen
  marginal-but-real (Wan 1.3B native; Wan2GP), embeddings verdict:
  **nothing obsoletes embeddinggemma+Qwen3-Reranker-0.6B** without an
  A/B first. MoE-at-home (gpt-oss-20b via --n-cpu-moe) and SSM hybrids
  (Granite 4.0-H, Jamba-3B) are testable today; DIAMOND is the one
  trainable-world-model option (~12 GB, days of wall-clock).

## Priorities (value per GPU-hour and engineering week)

| # | build | why / differentiation | cost |
|---|---|---|---|
| 1 | **Harbor + Terminal-Bench 2.0**: vendor the harness, adapt the lab scaffold via BaseAgent, run the local model field + scaffold-vs-Claude-Code head-to-heads | replaces saturated suites (F-014) with a community-standard hard ceiling; head-to-head scaffold comparison is rare in public | days; CPU-heavy, GPU only for local arms |
| 2 | **Eval→train→eval loop (flagship)**: Qwen3-4B QLoRA SFT on xLAM+ToolACE+own filtered trajectories (RFT), optional GRPO with the harness scorer as reward; GGUF→ollama export; measure delta on own suites + BFCL before/after | very few public solo labs close the full loop on consumer hardware; directly tests F-012's thesis (agentic training is the gap) | ~1–2 weeks; GPU-days |
| 3 | **Trajectory-inspection pipeline**: generalize the EXP-011 H3 scanner into an LLM-aided audit stage (cheat detection, failure-mode classification) run after every sweep | HAL showed end-state metrics miss cheating; differentiated, reusable, cheap; extends an artifact that already exists | days; minimal GPU |
| 4 | **tau2-bench vendoring**: dual-control user-sim domain via existing litellm | new eval genre (user simulation) at near-zero integration cost | days |
| 5 | **Computer-use slice**: Holo2-8B grounding service + independent ScreenSpot-Pro reproduction at 4/8-bit quants | the UI-TARS open-checkpoint discrepancy (38.7 vs claimed 61.6) proves independent reproduction has real value; entry into GUI agents | ~1 week |
| 6 | **Reasoning-effort sweep** on existing suites (HAL replication at lab scale) | cheap pre-registered experiment; publishable either way | GPU-hours |
| 7 | **Architecture evals**: gpt-oss-20b MoE-offload + Granite-4.0-H long-context agent-trace behavior vs same-size transformers | timely, almost nobody evals SSM hybrids agentically | GPU-hours |
| 8 | **Modality side-quests** (only as instrumented evals, not demos): TTS arena with WER-round-trip metric; embedding A/B on the kb corpus before any re-ingest decision | showcases harness discipline applied beyond text; low priority | days each |

Parked: TheAgentCompany (valuable but ~30 GB of services; revisit after
1–4), image/video LoRA training (feasible — kohya Flux LoRA at 12 GB
documented — but off-mission for now), DIAMOND world model (stretch
showcase), diffusion LLMs (curiosity tier).

## Sequencing note

1 → 2 is the spine: Harbor's rollout interfaces mean the Terminal-Bench
vendoring done for evaluation is reusable as the RL/SFT environment for
the fine-tuning flagship. 3 rides along every sweep from now on. The
job-search angle: the verified market read says eval/agent-reliability
skills are the differentiated hiring signal in 2026 — priorities 1–3
are exactly that story.

## Fast-moving caveats

VLM grounding SOTA turns over ~quarterly; TTS majors opened Jan 2026;
Terminal-Bench/Harbor are ~7 months old (APIs may shift). Re-verify
before each build. Leaderboard numbers cited above age fast.
