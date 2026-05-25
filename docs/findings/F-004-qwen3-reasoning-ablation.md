---
slug: F-004-qwen3-reasoning-ablation
title: "F-004: qwen3-14b-q4 reasoning-mode + max_tokens ablation"
status: draft
date: 2026-05-25
experiment: EXP-001b
plan_path: docs/exp/EXP-001b.md
---

# F-004: qwen3-14b-q4 reasoning-mode + max_tokens ablation

> Draft. Verdicts get pasted in after the EXP-001b sweep + evals are done.

## TL;DR

EXP-001b isolates the confound from F-003 H3 — was qwen3's bad format-following caused by chain-of-thought eating the 1024-token budget (a configuration issue), or is reasoning-mode itself the wrong default for short tasks?

(Fill in the punchline once the verdicts are in.)

## Setup

- **Experiment:** EXP-001b (pre-registered, plan: [`docs/exp/EXP-001b.md`](../exp/EXP-001b.md))
- **Sweep config:** [`conf/sweep/EXP-001b.yaml`](../../conf/sweep/EXP-001b.yaml)
- **Model:** `qwen3-14b-q4` (only)
- **Tasks:** all 24 PBS-v0.1 tasks
- **Seeds:** N=8
- **Total cells:** 24 × 1 × 2 × 8 = **384**, plus comparison against the existing 192 baseline cells from EXP-001

## Configurations compared

| Label | reasoning | max_tokens | how |
|---|---|---|---|
| baseline | on (default) | 1024 | (from EXP-001 — not re-run) |
| **B.1** | on (default) | **2048** | (new) |
| **B.2** | **off** | 1024 | (new) `think: false` Ollama API parameter |

## Critical infra finding (caught during smoke)

The naive approach of putting `/no_think` as a system or user message **does not actually disable qwen3 thinking** when running through Ollama. We verified this by hitting the daemon directly: completion-tokens stay high (~340 on a 16-character "list 3 colors" answer, with hundreds of thinking tokens). The only reliable way to disable qwen3 thinking on Ollama is the API-level `think: false` parameter — this drops completion-tokens to ~6 on the same query.

LiteLLM's `ollama_chat` backend forwards top-level `think: false` (and also accepts it via `extra_body`). The sweep runner was extended on 2026-05-25 to forward any `config.extra.*` key (other than `system_prompt`) to the request body, which makes per-config Ollama parameters declarative in the sweep YAML.

This is worth noting in any future qwen3 / DeepSeek / o1-style reasoning-by-default model: don't trust prompt-level mode toggles, use the API parameter.

## Hypothesis verdicts

(Paste the output of `uv run python scripts/analyze_exp001b.py` after evaluators are applied.)

### H1 — budget alone fixes it (B.1 vs baseline)
TBD

### H2 — disabling reasoning alone fixes it (B.2 vs gemma3 0.875)
TBD

### H3 — reasoning earns its keep on math (B.2 should lose vs baseline by ≥10pp)
TBD

## What changes in the lab

Based on the verdicts above:

(Filled in once verdicts are known. Recommendations updated for the default config in lab work going forward.)

## Caveats

- This experiment only ablates qwen3. Other reasoning-by-default models (deepseek-r1, kimi-thinking) may have different sensitivities. EXP-002+ could extend.
- The B.2 (`think: false`) variant uses qwen3 without its reasoning channel at all — it's not a "small think budget" variant, it's a "no reasoning" variant.
- Greedy decoding only. With temperature > 0, reasoning might recover.

## Reproduction

```bash
cd /data/lab/code
uv run lab sweep run conf/sweep/EXP-001b.yaml --enforce-pre-registration
uv run lab eval apply EXP-001b --no-judge
uv run python scripts/analyze_exp001b.py
```

## Files

- Plan: [`docs/exp/EXP-001b.md`](../exp/EXP-001b.md)
- Sweep config: [`conf/sweep/EXP-001b.yaml`](../../conf/sweep/EXP-001b.yaml)
- Analyzer: [`scripts/analyze_exp001b.py`](../../scripts/analyze_exp001b.py)
- Parent finding: [F-003](F-003-12gb-agent-v0.1.md)
