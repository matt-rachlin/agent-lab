# Protocol: LLM-as-judge calibration

**Pre-registered protocol for any LLM-judge use in the lab. Cite this protocol from EXP plans.**

LLM judges have well-documented biases (position, verbosity, self-preference, authority). The 2024–26 literature (JudgeBench, JudgeBench-Pro, Sage, CJE) is unambiguous: an uncalibrated judge model produces unreliable rankings. This protocol is the minimum bar for using a judge to score lab runs.

---

## 1. Choose the judge model

By default:

| Slice | Model | Notes |
|---|---|---|
| Bulk (90%+) | `qwen3:14b-q4` (local) | Free, ~30s/judgment |
| Standard | `gpt-oss-120b-cloud` | ~5s/judgment, Pro-tier budget |
| Oracle (5–10%) | `deepseek-v31-671b-cloud` or `kimi-k2-thinking-cloud` | Sparingly — Pro tier daily budget headroom limited |

Override only when an experiment plan explicitly justifies a different choice. Never use the generator model as its own judge (self-preference bias is severe).

## 2. Pre-register the rubric

The judge prompt and scoring scale must be pre-registered in the experiment plan or a referenced rubric file. **Once a sweep is running, the rubric does not change.**

Minimum rubric:

```
Score 0.0 to 1.0.
1.0 = fully addresses the task per the rubric.
0.0 = does not address the task at all.
Reply with JSON only: {"score": <float>, "reasoning": "<one sentence>"}.
```

## 3. Apply position-swap when relevant

For **pairwise** prompts (A vs B), evaluate both orderings and average. The lab judge factory supports this:

```python
judge = make_judge(model="gpt-oss-120b-cloud", position_swap=True)
```

For single-output scoring (the common case), position swap is N/A.

## 4. Calibrate against an oracle slice

For any sweep where the judge result drives a claim:

1. **Sample 5–10% of (run, evaluator) pairs uniformly at random.** Document the seed used for sampling.
2. **Score that slice with the oracle judge** (`deepseek-v31-671b-cloud`).
3. **Compute agreement**:
   - Pearson correlation between cheap-judge score and oracle score
   - Cohen's kappa on the binary pass/fail derived from the threshold
4. **Decision rule**:
   - Pearson r ≥ 0.80 AND kappa ≥ 0.60 → cheap judge is calibrated enough; use it for the rest
   - Pearson r ≥ 0.60 AND kappa ≥ 0.40 → use CJE-style correction (linear regression: oracle_score = a*cheap_score + b on the slice; apply (a, b) to the rest)
   - Below those bands → don't use the cheap judge; switch to oracle for the full sweep, or change rubric

5. **Report** all of: r, kappa, n_oracle, calibration coefficients in the finding. Hide nothing.

## 5. Reproducibility constraints

- Judge temperature = 0.0
- Judge `max_tokens` = 256 (rubric verdict is short; keeps cost predictable)
- LiteLLM proxy clamps Ollama Cloud outputs at 16384 — verify the rubric doesn't depend on longer judge outputs
- Pin the judge model's litellm_id in the sweep config; if the underlying tag changes (e.g. `deepseek-v3.2` released), that's a new experiment, not a rerun

## 6. Failure modes to call out in the finding

- **Bottoming out**: if >50% of judge scores are exactly 0.0 or 1.0, the judge isn't discriminating — either the rubric is too binary or the task is too easy/hard for the judge
- **High variance across seeds**: if the same (run, rubric) pair scored twice gives very different numbers, the judge isn't deterministic enough → set temperature=0 if not already, or switch judge
- **Refusal patterns**: judges may refuse to score certain content. Count refusals separately from low scores.

## 7. Cost budget

Pro-tier Ollama Cloud heuristic (per `RESEARCH_OPS_PLAN.md §"Inference & sweep execution"`):

| Judge | Cost shape | Comfortable daily volume |
|---|---|---|
| `qwen3:14b-q4` (local) | electricity only | unlimited |
| `gpt-oss-20b-cloud` | low GPU-sec | ~100/day |
| `gpt-oss-120b-cloud` | medium GPU-sec | ~50/day |
| `deepseek-v31-671b-cloud` | high GPU-sec | ~10–30/day |
| `kimi-k2-thinking-cloud` | very high GPU-sec | ~5–10/day |

Plan accordingly. Always pre-flight calculate before any sweep with >50 judge calls.

## 8. Implementation hooks

- `lab.eval.judge.make_judge(model=..., position_swap=...)` — factory
- `lab.eval.judge.parse_judge_response(text)` — tolerant parser (JSON → "score: N" → leading number)
- CJE calibration helper is **deferred to Phase 4** (`lab eval calibrate`); until then, do the agreement check manually in a notebook.
