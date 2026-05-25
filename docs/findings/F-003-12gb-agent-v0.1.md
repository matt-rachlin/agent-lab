---
slug: F-003-12gb-agent-v0.1
title: "F-003: The 12 GB Agent v0.1 — first characterization of local vs Ollama-Cloud on PBS"
status: draft
date: 2026-05-25
experiment: EXP-001
plan_path: docs/exp/EXP-001.md
---

# F-003: The 12 GB Agent v0.1 — first characterization of local + Ollama-Cloud models on PBS v0.1

> Draft. Hypothesis verdicts are computed automatically from `scripts/analyze_exp001.py` after the sweep completes. Do not paste verdicts here until the sweep is fully done + judged.

## TL;DR

(One paragraph. Fill in after the verdicts come back.)

## Setup

- **Experiment:** EXP-001 (pre-registered SHA on file in DB)
- **Sweep config:** [`conf/sweep/EXP-001.yaml`](../../conf/sweep/EXP-001.yaml)
- **Plan:** [`docs/exp/EXP-001.md`](../exp/EXP-001.md)
- **Total cells:** 1,152 = 24 PBS-v0.1 tasks × 6 models × 1 config (greedy-1024) × 8 seeds
- **Models compared:**
  - Local 12 GB: `qwen3-14b-q4`, `llama3.1-8b-q4`, `gemma3-12b-q4`, `phi4`
  - Ollama Cloud Pro: `gpt-oss-20b-cloud`, `gpt-oss-120b-cloud`
- **Hardware:** RTX 3080 Ti (12 GB VRAM), Fedora 43, local Ollama daemon

## Hypothesis verdicts

(Paste output of `uv run python scripts/analyze_exp001.py` here. The script computes verdicts strictly from the pre-registered rules.)

### H1 — Reasoning gap (cloud beats local on math)
TBD

### H2 — Knowledge near-parity
TBD

### H3 — Reasoning-mode advantage on format-following
TBD

### H4 — Reliability cliff
TBD

## Judge calibration

(Paste output of `uv run python scripts/judge_exp001.py` here. Calibration must pass r≥0.6 + kappa≥0.4 before LLM-judge-derived claims are reported.)

## Method footnotes

- All cells run with `temperature=0.0, top_p=1.0, max_tokens=1024`. No prompt variation.
- Models loaded with `keep_alive=5m` in LiteLLM proxy config (committed 2026-05-25). Per-cell GPU lease via Valkey `SETNX`.
- Operational notes from this run captured in `docs/postmortems/EXP-001.md`.

## Open caveats

- `gemma3-12b-q4` runs with 27–28 % layer spillover to CPU on this 12 GB card. Its latency numbers are not directly comparable to the other locals; pass-rate comparisons are.
- `qwen3-14b-q4` runs with reasoning mode default-on. If the 1024 token budget gets eaten by chain-of-thought, `not_empty` pass-rate will dip — flagged in the eval results.
- We did **not** vary the prompt in v0.1. A follow-on (EXP-001b) will sweep minimal-tool vs verbose-tool system prompts on the same matrix to isolate prompt sensitivity.
- We did **not** vary the quant in v0.1 (qwen3:14b ships Q4_K_M only on Ollama registry). The quant axis moves to EXP-002 on a model family with multi-quant releases.

## Reproduction

```bash
cd /data/lab/code
uv run lab sweep run conf/sweep/EXP-001.yaml --enforce-pre-registration
uv run lab eval apply --experiment EXP-001 --evaluator exact_match
uv run lab eval apply --experiment EXP-001 --evaluator regex_match
uv run lab eval apply --experiment EXP-001 --evaluator not_empty
uv run lab eval apply --experiment EXP-001 --evaluator latency_under
uv run lab eval apply --experiment EXP-001 --evaluator tokens_under
uv run python scripts/judge_exp001.py
uv run python scripts/analyze_exp001.py
```
