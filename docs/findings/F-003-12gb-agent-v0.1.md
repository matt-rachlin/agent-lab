---
doc_id: f-003-12gb-agent-v0-1
title: 'F-003: The 12 GB Agent v0.1 — three of four pre-registered hypotheses refuted'
zone: lab
kind: finding
status: active
owner: m
created: '2026-05-25'
last_updated: '2026-05-25'
last_verified: '2026-05-25'
depends_on:
- kind: doc
  target: exp-001
tags:
- lab
- finding
- findings
---

# F-003: The 12 GB Agent v0.1 — three of four pre-registered hypotheses refuted

## TL;DR

I pre-registered four hypotheses about how a single-GPU 12 GB lab should fare against frontier cloud models on a small in-house benchmark (PBS v0.1, 24 tasks). With N=8 seeds per cell and 1,152 runs:

- **H1** (cloud-vs-local gap on math ≥ 15 pp) — **REFUTED.** Observed gap was only 10.9 pp, not statistically significant (p = 0.19). Local models punch above their weight on grade-school math.
- **H2** (cloud-vs-local near-parity on knowledge, ≤ 10 pp) — **CONFIRMED.** The gap is *negative*: the best local model (phi4 or llama3.1) actually beat `gpt-oss-120b-cloud` by 1.6 pp on knowledge-recall. Recall-bound questions don't require frontier-scale compute.
- **H3** (qwen3 reasoning-mode beats gemma3/llama3.1 on format-following by ≥ 20 pp) — **REFUTED, spectacularly.** qwen3-14b-q4 was *37.5 pp worse* than gemma3-12b-q4 on format-following. The mechanism is brutal: at `max_tokens=1024`, qwen3's chain-of-thought eats the budget and emits an empty final answer 38 % of the time.
- **H4** (reliability cliff — local pass⁸/pass¹ ≤ 0.50 while cloud ≥ 0.75) — **REFUTED.** No cliff. Every model — local and cloud — had a reliability ratio above 0.85 on these task difficulties. With deterministic decoding, "consistency" is a near-trivial property at N=8 on this task pool.

This is good news for local-first research. The single binding result against locals isn't model quality — it's qwen3's reasoning mode at constrained token budgets. That's a configuration knob, not a capability ceiling.

## Setup

- **Experiment:** EXP-001 (plan: [`docs/exp/EXP-001.md`](../exp/EXP-001.md), pre-reg SHA 53bb021c)
- **Sweep config:** [`conf/sweep/EXP-001.yaml`](../../conf/sweep/EXP-001.yaml)
- **Total cells:** 1,152 = 24 PBS-v0.1 tasks × 6 models × 1 config (greedy, `temperature=0.0`, `max_tokens=1024`) × 8 seeds
- **Pass rate:** 1,128 / 1,152 (97.9 %) on first run. 24 / 1,152 (2.1 %) errored — all qwen3-14b-q4 on fmt-001/002/003 due to VRAM contention when phi4 hadn't fully unloaded. Re-ran cleanly after clearing the GPU lease. **Under the pre-registered 5 % kill criterion.**
- **Wall time:** ~4 hr 20 min generation. Sequential single-GPU lease via Valkey `SETNX`. Models swap with `keep_alive=5m` in the LiteLLM config.
- **Hardware:** RTX 3080 Ti (12 GB VRAM), Fedora 43, local Ollama daemon at 11434; LiteLLM proxy at 4000; Ollama Cloud Pro for cloud models.

## Hypothesis verdicts

### H1 — Reasoning gap on math · REFUTED

Pre-registered rule: `mean_pass@1(gpt-oss-120b-cloud, math) − max(local, math) ≥ 0.15` AND Welch p < 0.05.

| model | mean pass@1 on math-reasoning |
|---|---|
| gpt-oss-120b-cloud | **0.609** |
| gpt-oss-20b-cloud | 0.555 |
| qwen3-14b-q4 | 0.500 |
| phi4 | 0.484 |
| gemma3-12b-q4 | 0.359 |
| llama3.1-8b-q4 | 0.328 |

Observed delta `(cloud-frontier − best-local)` = **+0.109**. Welch's t-test p = **0.195** (frontier vs all-locals per-task means; n=8 tasks × locals=4 → 32 values vs n=8). Effect is in the predicted direction but smaller than my pre-registered threshold and not statistically significant.

**What I had wrong.** I expected reasoning-heavy 8-task math to expose the local-12GB ceiling. The actual gap is ~10 pp, well within what better prompting / scaffold / tool-use could close. Math at this difficulty (grade-school + simple probability + light algebra) is more about *getting the arithmetic right* than discovering novel reasoning paths, and quantized 14B/12B/8B models are competent at it. This is a positive result for local-first research.

### H2 — Knowledge near-parity · CONFIRMED

Pre-registered rule: same delta ≤ 0.10.

| model | mean pass@1 on knowledge-recall |
|---|---|
| phi4 | **1.000** |
| llama3.1-8b-q4 | 1.000 |
| qwen3-14b-q4 | 0.984 |
| gpt-oss-120b-cloud | 0.984 |
| gpt-oss-20b-cloud | 0.984 |
| gemma3-12b-q4 | 0.937 |

Observed delta `(cloud-frontier − best-local)` = **−0.016**. Frontier is *behind* the best local by ~1.6 pp.

This was the easy hypothesis — I predicted near-parity and got it. The interesting nuance: a 9.1 GB `phi4` ties with a 120B cloud model on this knowledge slice. PBS-v0.1 knowledge tasks are factoid-style (capital cities, treaty years, language origin), well within pretraining coverage for any model in this comparison.

### H3 — Reasoning-mode advantage on format-following · REFUTED

Pre-registered rule: `qwen3-14b-q4 mean pass@1 on fmt ≥ 0.20 higher than each of gemma3-12b-q4 and llama3.1-8b-q4`, with p < 0.05.

| model | mean pass@1 on format-following |
|---|---|
| gpt-oss-120b-cloud | **0.937** |
| gemma3-12b-q4 | 0.875 |
| llama3.1-8b-q4 | 0.750 |
| gpt-oss-20b-cloud | 0.711 |
| phi4 | 0.546 |
| qwen3-14b-q4 | 0.500 |

Observed delta: qwen3 vs gemma3 = **−0.375** (Welch p = 0.124); qwen3 vs llama3.1 = **−0.250** (Welch p = 0.335). qwen3 is the *worst* local model on format-following, not the best.

**Mechanism: chain-of-thought is destroying the budget.** Empty-response rate (`not_empty` evaluator) by model on the full sweep:

| model | empty% |
|---|---|
| qwen3-14b-q4 | **38.0%** |
| gpt-oss-20b-cloud | 12.5% |
| gpt-oss-120b-cloud | 7.8% |
| gemma3-12b-q4 | 0.0% |
| llama3.1-8b-q4 | 0.0% |
| phi4 | 0.0% |

At `max_tokens=1024`, qwen3 spends its budget on `<think>...</think>` and never emits the final answer. Worse, on short fmt tasks like "List exactly 3 colors, comma-separated" — exactly the kind of task that doesn't *need* reasoning — qwen3 still thinks for hundreds of tokens before stopping. This is a configuration failure, not a capability one. A follow-up (**EXP-001b**) will sweep `(qwen3-with-reasoning-off, qwen3-with-reasoning-on, max_tokens=1024 vs 2048)` to isolate the budget vs reasoning-mode effects.

The bigger story: **the best on-machine format-follower is gemma3-12b-q4** (0.875), beating gpt-oss-20b-cloud by 16 pp and within 6 pp of gpt-oss-120b-cloud. For lab agents that need strict output formats, gemma3 — not qwen3 — is the default.

### H4 — Reliability cliff · REFUTED

Pre-registered rule: `reliability_ratio(gpt-oss-120b-cloud) ≥ 0.75` AND `min(reliability_ratio(local)) ≤ 0.50`.

Reliability ratio = mean(pass^8) / mean(pass@1), averaged over the 24 (model, task) cells per model. Geometric pass^8 = ∏ score(seed_i).

| model | reliability ratio |
|---|---|
| gemma3-12b-q4 | **1.000** |
| phi4 | 0.976 |
| gpt-oss-20b-cloud | 0.907 |
| gpt-oss-120b-cloud | 0.895 |
| llama3.1-8b-q4 | 0.881 |
| qwen3-14b-q4 | 0.854 |

Cloud frontier *passes* the ≥0.75 part (0.895). But the minimum local ratio is 0.854, nowhere near the predicted 0.50 cliff.

**Why I was wrong.** I expected stochastic decoding effects to dominate. With `temperature=0.0` (greedy), the local models are *deterministic up to backend non-determinism* (CUDA reductions, batching, etc.), and the surviving variance is small at N=8. RELIABILITY-001's pass^8 collapses (F-002) were from N=8 on *intermediate-difficulty* math, where 30-50 % pass@1 leaves room for pass^8 ≪ pass@1. EXP-001's task pool is bimodal (very easy or very hard), so cells live near 0 % or 100 % pass@1, where pass^8 / pass@1 stays near 1. **The reliability-cliff phenomenon is real (F-002 still stands), but it shows up at a different operating point than I targeted here.** Plan EXP-003 around tasks deliberately curated at the 40–60 % pass@1 band to surface it.

## Sample sizes

| model | done | error | empty% |
|---|---|---|---|
| gemma3-12b-q4 | 192 | 0 | 0.0 |
| gpt-oss-120b-cloud | 192 | 0 | 7.8 |
| gpt-oss-20b-cloud | 192 | 0 | 12.5 |
| llama3.1-8b-q4 | 192 | 0 | 0.0 |
| phi4 | 192 | 0 | 0.0 |
| qwen3-14b-q4 | 192 | 0 (24 reran) | 38.0 |
| **total** | **1152** | **0** | — |

## Caveats and known limitations

1. **N=8 is too few for tight CIs on per-task pass-rates.** Welch's t-test on H1 came in at p=0.195 — there's an effect, just under-powered. Future experiments planning to declare significance at α=0.05 should target N=16 or N=24, accepting longer wall times.
2. **qwen3 was running with reasoning mode default-on and no `/no_think` system prompt.** This is a "qwen3 as it ships" comparison, not a "qwen3 vs gemma3 with equivalent inference strategies" comparison. EXP-001b will fix this.
3. **gemma3-12b-q4 runs with 27–28 % layer spillover to CPU on this 12 GB card.** Its latency is ~5 × slower than llama3.1-8b-q4 per token. Pass-rate comparisons are unaffected; throughput comparisons would be misleading.
4. **No tool-use, no agent scaffold.** This was single-turn chat. Real agent workloads add a tool-call layer; results may not generalize to that setting (and probably won't — tool-call validity is a much bigger sensitivity).
5. **LLM-judge slice runs after this finding is filed.** Verdicts above use only deterministic evaluators (`exact_match`, `regex_match`). Judge calibration is a separate, follow-on artifact (not gating these verdicts because all four hypotheses were operationalised on deterministic scores per the pre-registration).
6. **Judge model overlap.** The cheap judge is `gpt-oss-20b-cloud`, which is also under test. For tasks where the same model both generates and judges, the judge score is unreliable. Filter or oracle-replace those when reading the judge results.
7. **Single workstation, single GPU.** No multi-GPU split, no multi-host scheduling.

## What changes in the lab from here

These verdicts produce four immediate decisions:

1. **Default local model for format-strict agent work: `gemma3-12b-q4`.** Not qwen3.
2. **Default local model for math + general reasoning: `qwen3-14b-q4` only when paired with `max_tokens ≥ 2048` or a `/no_think` system prompt.** Otherwise default to `phi4`.
3. **Don't reach for `gpt-oss-120b-cloud` on knowledge-recall tasks.** Local is as good and free.
4. **Plan EXP-003 around 40-60% pass@1 tasks** to actually surface the reliability cliff seen in F-002. PBS-v0.1's bimodal difficulty hid it.

## Follow-ons queued

- **EXP-001b** — qwen3 reasoning-mode ablation: `/no_think` vs default, `max_tokens=1024 / 2048 / 4096`, on the same 24-task pool. Will isolate the configuration question raised by H3.
- **EXP-002** — Quantization sensitivity on a model family with multiple quants on the registry (likely llama3.1:8b @ Q4_K_M / Q5_K_M / Q8_0 / fp16).
- **EXP-003** — Reliability cliff at the 40–60% pass@1 band. Requires curating ~10 new PBS tasks calibrated to that difficulty.

## Reproduction

```bash
cd /data/lab/code

# Confirm pre-registration
uv run lab exp show EXP-001

# Sweep (~4 hr on RTX 3080 Ti, N=8)
uv run lab sweep run conf/sweep/EXP-001.yaml --enforce-pre-registration

# Deterministic evaluators
uv run lab eval apply EXP-001 --no-judge

# Cheap LLM-judge (optional; ~100 min cloud time)
uv run python scripts/judge_exp001.py

# Verdicts
uv run python scripts/analyze_exp001.py
```

## Files

- Plan: [`docs/exp/EXP-001.md`](../exp/EXP-001.md)
- Sweep config: [`conf/sweep/EXP-001.yaml`](../../conf/sweep/EXP-001.yaml)
- Auto-verdict script: [`scripts/analyze_exp001.py`](../../scripts/analyze_exp001.py)
- Judge script: [`scripts/judge_exp001.py`](../../scripts/judge_exp001.py)
- Postmortem: [`docs/postmortems/EXP-001.md`](../postmortems/EXP-001.md)
trust_level: unverified
