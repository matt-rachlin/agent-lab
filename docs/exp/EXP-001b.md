# EXP-001b: qwen3 reasoning-mode + max_tokens ablation

Date created: 2026-05-25
Status: planned
Parent finding: [F-003](../findings/F-003-12gb-agent-v0.1.md) — H3 was refuted by qwen3 producing empty responses at `max_tokens=1024`. This experiment isolates the cause.

## Why this experiment exists

F-003 H3 expected qwen3-14b-q4 to beat gemma3/llama3.1 on format-following by ≥20 pp. It lost by 37.5 pp instead. The mechanism appears to be chain-of-thought eating the 1024-token budget — qwen3 emitted an empty final answer in **38 %** of EXP-001 cells.

That mechanism is a configuration confound. The experiment did not separate:

- **reasoning-mode-by-default** (qwen3's "thinking" mode emits an unbounded `<think>...</think>` channel before any user-visible content)
- **token-budget pressure** (`max_tokens=1024` may simply not be enough head-room when reasoning is on)

EXP-001b isolates the two axes on the same task set so we can attribute the H3 failure correctly.

## Hypothesis

Comparisons are against the existing EXP-001 baseline: qwen3-14b-q4, reasoning=on (default), `max_tokens=1024`, no system prompt. That baseline's stats on PBS-v0.1 are pre-known and are not allowed to change.

- **H1 — Budget alone fixes it.** Holding reasoning on but raising `max_tokens` to 2048 reduces `not_empty`-rate to **≤ 10 %** AND raises format-following mean pass@1 by **≥ 20 pp** vs baseline.
- **H2 — Disabling reasoning is enough.** Setting system prompt to `/no_think` while keeping `max_tokens=1024` reduces `not_empty`-rate to **≤ 5 %** AND raises format-following mean pass@1 to **within 5 pp of gemma3-12b-q4's EXP-001 fmt mean pass@1 (0.875)**.
- **H3 — Reasoning earns its keep on math.** On math-reasoning, disabling reasoning (`/no_think`, 1024) **drops mean pass@1 by ≥ 10 pp** vs baseline. (If true, this validates that qwen3's reasoning mode is not just overhead; it's helping somewhere.)

Mutually independent. All three verdicts reported regardless of how they fall.

## Why this design and not a 6-cell factorial

A full `(reasoning_on, reasoning_off) × (1024, 2048, 4096)` factorial would be 6 configs × 24 tasks × N=8 = 1152 cells on qwen3 alone (slow model with reasoning) = ~3 hr+ wall time. That's overkill for the question.

Instead we run **2 NEW configurations** and **compare both to the existing EXP-001 baseline**:

| Config | reasoning | max_tokens | system_prompt |
|---|---|---|---|
| EXP-001 baseline (existing, not re-run) | on | 1024 | (none) |
| **B.1** (this experiment) | on | 2048 | (none) |
| **B.2** (this experiment) | off (`/no_think`) | 1024 | `/no_think` |

This isolates each axis once. Total cells = 24 tasks × 1 model × 2 NEW configs × N=8 = **384 cells**. Expected wall time on a warm qwen3: ~30 min.

If both H1 and H2 confirm, the cause is "either axis fixes it" → set max_tokens=2048 by default for qwen3 in lab work. If only H2 confirms, reasoning is the wrong default for short tasks. If only H1 confirms, the budget alone is the gate.

## Method

### Model

Only `qwen3-14b-q4`. (Other models are not the question.)

### Configs

```yaml
configs:
  - name: greedy-2048-thinking
    temperature: 0.0
    top_p: 1.0
    max_tokens: 2048
    scaffold: single_turn
    # reasoning on by default — no system_prompt override

  - name: greedy-1024-no-think
    temperature: 0.0
    top_p: 1.0
    max_tokens: 1024
    scaffold: single_turn
    extra:
      system_prompt: "/no_think"
```

The `system_prompt` field is consumed by the sweep runner (added 2026-05-25). The `/no_think` token is qwen3's documented mode-flip directive — placing it in the system prompt switches the model to single-channel output.

`config_hash` will differ between the two new configs (and from the EXP-001 baseline) because either `max_tokens` or `extra.system_prompt` differs. `run_id` will therefore differ too, so resume + idempotency stay clean.

### Tasks

Same 24 PBS-v0.1 tasks as EXP-001.

### Seeds

`[1, 2, 3, 4, 5, 6, 7, 8]` — same as EXP-001.

### Evaluators (pre-registered)

Same six deterministic evaluators applied: `exact_match`, `regex_match`, `not_empty`, `latency_under`, `tokens_under`, `json_valid`. LLM-judge **not** required for these verdicts — H1/H2/H3 are operationalised on `not_empty` rate and category-level pass@1 only.

### Statistics

Per `protocols/reliability-sweep.md`:
- per-(category) mean pass@1 with bootstrap 95 % CI (n_resamples=2000)
- Welch's t-test on per-task means for any reported delta
- empty-rate per config

## Success / failure criteria

Applied after the sweep + evals are done, no peeking:

- **H1 confirmed** ⇔ `empty_rate(B.1) ≤ 0.10` AND `(mean_pass1_fmt(B.1) − 0.500_baseline) ≥ 0.20`
- **H2 confirmed** ⇔ `empty_rate(B.2) ≤ 0.05` AND `|mean_pass1_fmt(B.2) − 0.875| ≤ 0.05`
- **H3 confirmed** ⇔ `(0.500_baseline − mean_pass1_math(B.2)) ≥ 0.10`

Where `0.500_baseline` and `0.875` are pre-known EXP-001 values for qwen3 fmt pass@1 and gemma3 fmt pass@1, respectively (filed in F-003).

## Confounders

- Same hardware (RTX 3080 Ti), same daemon (Ollama with `keep_alive=5m`), same proxy (LiteLLM). No silent infra drift.
- Only qwen3 — no cross-model contamination.
- Same tasks, same seeds, same temperature/top_p — only the two declared axes vary.
- We are **not** re-running the EXP-001 baseline. The H1/H2 deltas use the recorded EXP-001 numbers; if there is any concern that the baseline drifted, we'd need to redo it. The lab's git SHA, model SHA, and `keep_alive=5m` LiteLLM config are unchanged from when EXP-001 ran ~6 hr ago, so this is safe for v0.1.

## Kill criteria

- If `/no_think` doesn't actually disable qwen3 thinking (manually verify on 3 smoke cells before the full sweep), abort and revise the experiment.
- If empty-rate stays >30 % under either new config, that itself is a finding — record it in F-004 with the verdicts as-is, do NOT iterate to find a "fix".
- >5 % error rate → stop, fix, re-run.

## Pre-mortem

What plausibly goes wrong:

1. `/no_think` in system prompt doesn't actually disable qwen3 thinking — it may need to be appended to the user message instead. Mitigation: smoke 3 cells with both placements before the full sweep, pick whichever actually drops empty-rate.
2. `max_tokens=2048` still isn't enough — qwen3 thinks for >2k tokens on hard math. Mitigation: 4096 is a fallback config we'd add as EXP-001c if needed; not v0.1.
3. The two-axis design hides interactions — what if `reasoning_off + max_tokens=2048` is the actual winner? Mitigation: not for v0.1. Recorded as an EXP-001c followup.

## Estimated cost

| Resource | Estimate |
|---|---|
| Wall time | 384 cells × ~5s avg (qwen3 hot) = **~30 min** |
| Cloud GPU $ | $0 — qwen3 is local |
| Local electricity | ~$0.05 |
| Risk | low — single model, deterministic decoding, idempotent resume |

## Reproduction

```bash
cd /data/lab/code
uv run lab exp register docs/exp/EXP-001b.md --hypothesis "..."
uv run lab sweep run conf/sweep/EXP-001b.yaml --enforce-pre-registration
uv run lab eval apply EXP-001b --no-judge
uv run python scripts/analyze_exp001b.py
```

## Pre-registration commitment

Plan committed before sweep launch. Not edited after sweep starts. Kill-criterion outcomes (if any) recorded in F-004 with the verdicts.
