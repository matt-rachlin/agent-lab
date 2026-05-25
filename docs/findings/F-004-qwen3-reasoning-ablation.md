---
slug: F-004-qwen3-reasoning-ablation
title: "F-004: qwen3-14b-q4 reasoning mode is net-negative on PBS-v0.1"
status: confirmed
date: 2026-05-25
experiment: EXP-001b
plan_path: docs/exp/EXP-001b.md
---

# F-004: qwen3-14b-q4 reasoning mode is net-negative on PBS-v0.1

## TL;DR

All three pre-registered hypotheses for EXP-001b were refuted. The interesting result is the H3 inversion:

**For qwen3-14b-q4 on PBS-v0.1, disabling reasoning mode (`think: false`) outperforms reasoning-on at every task category** — not just format-following. It boosts pass@1 by +12.5 pp on fmt, **+28 pp on math**, and +23 pp on knowledge, while dropping empty-response rate from 32 % to 0 %.

Doubling the token budget (B.1: `max_tokens=2048`, reasoning still on) helps less than disabling reasoning entirely (B.2: `max_tokens=1024`, no reasoning). And neither closes the format-following gap to gemma3-12b-q4 (0.875).

The recommendation from F-003 inverts: **don't enable qwen3 reasoning for lab work unless the task specifically benefits from chain-of-thought** (and EXP-001b's task pool didn't surface such a task).

## Per-config results

All numbers from qwen3-14b-q4, greedy decoding (T=0), N=8 seeds per cell, 24 PBS-v0.1 tasks:

| config | empty rate | fmt pass@1 | math pass@1 | know pass@1 |
|---|---|---|---|---|
| baseline (reasoning on, `max_tokens=1024`) | **31.9 %** | 0.500 | 0.344 | 0.766 |
| **B.1** (reasoning on, `max_tokens=2048`) | 25.5 % | 0.625 | 0.375 | 0.984 |
| **B.2** (reasoning **off**, `max_tokens=1024`) | **0.0 %** | **0.625** | **0.625** | **1.000** |

For reference: gemma3-12b-q4 fmt pass@1 from F-003 = 0.875.

## Setup

- **Experiment**: EXP-001b (plan: [`docs/exp/EXP-001b.md`](../exp/EXP-001b.md), pre-reg SHA ae3f1db3)
- **Sweep config**: [`conf/sweep/EXP-001b.yaml`](../../conf/sweep/EXP-001b.yaml)
- **Total cells**: 384 = 24 PBS-v0.1 tasks × 1 model × 2 new configs × N=8. Plus the existing 192 baseline cells from EXP-001 (qwen3-14b-q4).
- **Wall time**: 2 h 50 min generation (B.2 finished in ~5 min; B.1 took the rest — qwen3 with reasoning uses *all* of any budget you give it).
- **Pass rate**: 384 / 384 done, 0 errors. No kill criteria fired.

## Critical infra detail (caught during smoke)

The naive `/no_think` prompt token does NOT disable qwen3 thinking when running through Ollama. We smoke-tested this against the Ollama daemon directly: a "list 3 colors" query with `/no_think` as user-message suffix produced ~340 completion tokens, mostly thinking. The same query with the API-level `think: false` parameter produced **6** tokens.

LiteLLM forwards top-level `think: false` correctly. The sweep runner was extended (commit 2026-05-25) to forward any `config.extra.*` key (other than the locally-consumed `system_prompt`) into the request body, which makes per-config Ollama parameters declarative in the sweep YAML.

This matters beyond qwen3 — deepseek-r1, kimi-thinking, and o1-style models likely have similar disconnects between "prompt tokens that look like switches" and "API params that actually are switches." Default to the API params and verify with completion-token counts.

## Hypothesis verdicts

### H1 — budget alone fixes it · REFUTED

Pre-registered rule: B.1 `empty_rate ≤ 0.10` AND `Δfmt pass@1 ≥ +0.20` vs baseline.

- B.1 empty_rate = **0.255** (rule: ≤0.10 — refuted)
- B.1 fmt pass@1 = 0.625, baseline = 0.500, Δ = **+0.125** (rule: ≥+0.20 — refuted)

Doubling the budget helps, but not nearly as much as predicted. qwen3 just thinks proportionally longer.

### H2 — disabling reasoning alone fixes it · REFUTED

Pre-registered rule: B.2 `empty_rate ≤ 0.05` AND `|fmt pass@1 − 0.875| ≤ 0.05` (i.e. reach gemma3 level).

- B.2 empty_rate = **0.000** (rule: ≤0.05 — confirmed)
- B.2 fmt pass@1 = 0.625, gemma3 (F-003) = 0.875, |Δ| = **0.250** (rule: ≤0.05 — refuted)

Disabling reasoning removes the empty-response problem entirely, but qwen3's underlying format-following is still 25 pp behind gemma3. There's a model-level capability gap on fmt tasks, not a configuration issue.

### H3 — reasoning earns its keep on math · REFUTED (and inverted)

Pre-registered rule: B.2 math pass@1 drops by ≥10 pp vs baseline (i.e. reasoning helped on math).

- baseline math pass@1 = 0.344
- B.2 math pass@1 = **0.625**
- Δ = **−0.281** (i.e. B.2 *improved* by 28 pp; the predicted *drop* would be a positive Δ in the rule)
- Welch p (baseline vs B.2 per-task means) = 0.276

The verdict is "refuted" but the prediction was directional — I expected reasoning to help on math, and it actively hurts on this benchmark. The Welch p-value falls short of significance (n=8 tasks per arm is under-powered), but the effect size is the largest one observed in this experiment.

**Most plausible mechanism**: qwen3's chain-of-thought is consuming the 1024-token budget and getting truncated before reaching the answer. When reasoning is off, the model emits a short final answer directly from prior knowledge / pattern-matching, which on PBS math (grade-school arithmetic, simple probability, light algebra) is often correct.

## What this changes in the lab

The F-003 recommendation about qwen3 needs to invert:

| Use case | Previously (F-003) | Now (F-004) |
|---|---|---|
| Format-strict tasks | "use gemma3 instead" | **unchanged — use gemma3** |
| Math / reasoning tasks | "use qwen3 with reasoning + max_tokens ≥ 2048" | **use qwen3 with `think: false`** |
| Knowledge-recall | "any local model" | **any local model with `think: false` if it's qwen3** |
| Multi-step tool-use scaffold | (untested) | (still untested — `think: false` may need separate testing here) |

The lab's default qwen3 invocation should ship `think: false` until/unless an experiment surfaces a task class where reasoning actually helps. EXP-001b did not find one.

## Caveats and known limitations

1. **N=8 tasks per category** — Welch p-values stay above 0.05 for many of the deltas reported. The effect sizes are large and consistent, but the confidence intervals overlap. EXP-003 or follow-on work should target N≥16 tasks per category.
2. **Greedy decoding only.** With `temperature > 0`, reasoning might sample better paths and recover. Untested.
3. **The B.1 wall-time blew out the original 30-min estimate to 2 h 50 min.** qwen3 with reasoning will use *all* of any budget you give it (~30-50 s/cell at 2048 tokens; some cells hit 100 s). This is operational, not methodological — but it justifies adding a per-cell timeout cap to sweep configs for reasoning-by-default models.
4. **No agent-loop / tool-use scaffold.** This was single-turn chat. Reasoning may earn its keep in multi-step settings where the chain-of-thought can be used as input to subsequent calls (the "reasoning trace as scratchpad" pattern). EXP-004+ territory.
5. **One model.** Other reasoning-by-default models (deepseek-r1, kimi-k2-thinking, the cloud `gpt-oss-*` reasoning checkpoints) may behave differently. The F-004 conclusion is specifically about `qwen3-14b-q4` on `PBS-v0.1`.

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
- Postmortem: [`docs/postmortems/EXP-001b.md`](../postmortems/EXP-001b.md)
