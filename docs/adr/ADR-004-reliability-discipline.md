# ADR-004: N ≥ 8 seeds + pass^k as the reliability discipline

Status: accepted
Date: 2026-05-25
Deciders: Matt Rachlin

## Context

The 2026 evaluation literature is unambiguous: single-seed `pass@1` reporting is malpractice. Variance is 2–6 percentage points at temperature 0 ([Bjarnason et al. 2026](https://hf.co/papers/2602.07150)). Sierra's `τ²-bench` introduced `pass^k` (probability all k attempts succeed) as the operational reliability metric.

Our own first reliability sweep ([F-002](../findings/F-002-reliability-pass-caret-k.md)) confirmed the principle with sharp local data: gemma3-12b and llama3.1-8b deterministically produced WRONG answers to `(47*8) − (12*19)` across all 8 seeds, while qwen3-14b (with reasoning) was correct 8/8. Single-seed reporting would have characterised these as random variance instead of systematic failure.

We need to commit to a default that prevents future-us from cutting this corner under time pressure.

## Decision

**The lab default is `N ≥ 8` seeds per (model, config, task) cell, with `pass^k` (k ∈ {1, 4, 8}) and a bootstrap 95% CI as the reported metrics.** Documented in [`protocols/reliability-sweep.md`](../protocols/reliability-sweep.md).

Specifically:

- Every EXP plan must state the seed count (≥ 8) and the exact seed list.
- Every finding must report N, the 95% CI, and the exact evaluator + config the number derives from.
- The default seed schedule is `[1, 2, 3, 4, 5, 6, 7, 8]`. No randomisation per cell.
- For higher discrimination needs (model differences within ~3pp), N=16 is allowed. For known-flaky cells, N=32 with an explicit reason in the plan.

The `lab.analyze.stats` module computes pass^k and bootstrap CIs automatically; `lab analyze report` includes them in the markdown output for any evaluator with `eval_results`.

## Consequences

- **Easier**: every sweep is an explicit reliability sweep. We never report a number we can't defend on variance grounds.
- **Harder**: every sweep costs 8× the compute of a single-seed sweep. For a 5-model × 20-task matrix that's 800 cells, not 100.
- **Risks**: for pass^8 to be meaningfully > 0, the model must succeed at least once on every cell. Otherwise the metric pegs at 0 and the reliability ratio looks worse than it is. Mitigation: pair pass^k with pass@1 always; a low pass^8 with high pass@1 means "good on average, flaky in worst case" — both numbers tell the story.

## Considered alternatives

- **N = 1 with confidence intervals via paired comparisons**. Rejected — variance is intrinsic to the run, not just to model differences. CIs from a single run are meaningless.
- **N = 3 as a "first pass"**. Rejected — pass^k for k=8 collapses; reliability ratio is undefined. Skipping straight to N=8 means we don't have to re-run the same matrix when we want the proper metric.
- **N = 8 but only sample a subset of tasks**. Acceptable, but tracked in the plan as a known scope limitation. Don't pretend a 5-task slice characterises general capability.
