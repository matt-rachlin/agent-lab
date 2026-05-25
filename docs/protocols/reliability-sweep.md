# Protocol: Reliability sweep (N ≥ 8)

**Cite this protocol from any EXP plan that reports a `pass@k`, `pass^k`, or mean-score number.**

The 2026 literature (Bjarnason et al., Mehta CLEAR, Rabanser/Kapoor/Narayanan) and our own [F-002](../findings/F-002-reliability-pass-caret-k.md) demonstrate that single-seed reporting on LLM evaluation produces misleading numbers — variance is 2–6 percentage points even at temperature 0, and "failures" may be deterministically wrong, not stochastic.

This protocol is the minimum bar for reporting reliability claims in the lab.

---

## 1. Sample size

- **N ≥ 8 seeds** per (model, config, task) cell. Hard rule for any reported number.
- For Pareto charts: N=8 is the default. N=16 if the metric needs to discriminate models within ~3pp.
- For an unreliable cell (judge variance or stochastic agent), N=32. Document the reason in the plan.

## 2. Seeds

- Use the same seed schedule for every cell in a sweep: `[1, 2, 3, 4, 5, 6, 7, 8]`. Don't randomize per cell.
- Seeds map to LiteLLM `seed` parameter (where supported) AND to per-cell Python `random` state used by scaffold logic.
- At temperature 0, GPU non-determinism still produces small variance ([arXiv 2506.09501](https://arxiv.org/html/2506.09501v2)). We report it; we don't pretend otherwise.

## 3. Per-cell metrics

For each (model, config, task) cell with N seeds, the lab computes:

- **pass@1** — mean pass rate across seeds (the average of binary outcomes)
- **pass^k** for k ∈ {1, 4, 8} — empirical probability that all of k random draws (without replacement) pass: `C(M, k) / C(N, k)` where M is the number of seeds that passed. **pass^8 is 0 unless all 8 seeds pass.**
- **bootstrap 95% CI** on the pass rate (n_resamples = 2000)
- **deterministic-failure check**: if N ≥ 4 and pass@1 ∈ {0, 1}, flag "deterministic" (the model always gets it right or always wrong)

All of this lives in `lab.analyze.stats` and is included automatically in `lab analyze report` for any evaluator with persisted `eval_results`.

## 4. Per-model summary

`reliability_ratio = mean(pass^8) / mean(pass@1)`, averaged across (model, task) cells:

- 1.0 — fully reliable: every cell is either always-pass or always-fail
- ~0.5 — flaky: half the wins are flukes
- 0 — no cell achieves pass^8 (likely undersampled at N=8)

A model can have high pass@1 with low reliability ratio (= it gets it right "most of the time" but flakes) or low pass@1 with high reliability ratio (= it confidently fails consistently — like the gemma3/llama3.1 finding in F-002).

## 5. Pre-registration

The plan must specify:
- Number of seeds (≥ 8)
- The exact seed list
- Which metric drives the success/failure criterion
- The statistical test if comparing two models (Welch's t-test default; report p-value, effect size, n)

## 6. Reporting in findings

Every reported number must carry:
- N
- 95% CI (bootstrap or analytic)
- The exact evaluator name + version
- The exact config (temperature, top_p, max_tokens, scaffold)

If a single number is reported without these, it didn't clear the bar. Single-number reporting is malpractice in 2026; F-002 is the lab's own evidence of why.

## 7. Common mistake to avoid

**Iterating on the rubric or evaluator AFTER seeing the per-seed scores.** Goodhart's law is empirically vicious — fixing the eval until your favorite model wins is p-hacking. Define the rubric in the plan; commit; then evaluate.

## 8. Worked example

[`EXP RELIABILITY-001`](../exp/RELIABILITY-001.md) and the resulting [F-002](../findings/F-002-reliability-pass-caret-k.md) are the canonical worked example. Read both before designing your first sweep.
