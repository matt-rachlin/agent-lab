# F-002: pass^k reveals systematic wrong answers that pass@1 misreports as noise

Date: 2026-05-25
Confidence: high
Source: EXP RELIABILITY-001 (120 runs: 3 models × 4 tasks × 8 seeds × 1 config)

## Claim

At greedy decoding (temperature=0, top_p=1.0, max_tokens=1024), local 12 GB-class instruction-tuned models without explicit reasoning mode produce **deterministically wrong answers** on multi-step arithmetic. The same wrong answer appears across all 8 seeds. Single-seed benchmarks would misrepresent these as random noise or model variance; the N=8 sweep exposes the systematic nature of the failure.

## Evidence

`math-001`: "Compute (47 * 8) - (12 * 19) and reply with just the number." Gold: 148.

| Model | pass@1 | pass^8 | Actual answer (across 8 seeds) |
|---|---:|---:|---|
| `qwen3-14b-q4` (reasoning on, default) | 1.0 | 1.0 | `148` (correct, 8/8) |
| `gemma3-12b-q4` | 0.0 | 0.0 | `254` (wrong, 8/8 — same answer every time) |
| `llama3.1-8b-q4` | 0.0 | 0.0 | `376` (the partial product 47×8, 8/8) |

Three observations:

1. The failures are **identical** across seeds at T=0 — these are deterministic model behaviors, not stochastic errors.
2. llama3.1's answer `376` reveals the failure mode: it computed the first multiplication and then ignored the rest of the expression. gemma3 produced `254`, which doesn't match any partial computation — it's just confidently wrong.
3. **A naive single-seed benchmark would have shown the same wrong number once and not flagged anything.** The N=8 sweep makes clear that "got the answer right once" and "always gets the answer right" are different things, and that "got the answer wrong once" and "always gets the answer wrong" are different things.

### Per-model summary across the 5 PBS tasks evaluated (N=8 seeds each)

| Model | exact_match pass rate | reliability_ratio (pass^8 / pass@1_mean) | latency_under (10s) | tokens_under (500) |
|---|---:|---:|---:|---:|
| `qwen3-14b-q4` | 96.9% | 0.774 | 37.5% | 60.0% |
| `gemma3-12b-q4` | 75.0% | 1.000 | 80.0% | 100% |
| `llama3.1-8b-q4` | 71.9% | 0.696 | 100% | 100% |

The Pareto picture is immediate:

- **qwen3** wins on correctness but pays for it in latency (p95 = 47.7s) and token volume (mean 495 tokens/response, 40% of runs exceed the 500-token budget).
- **llama3.1** is the fastest (mean 456ms, p95 1.5s) and tightest on tokens but loses on reasoning-style tasks.
- **gemma3** is the most *reliable* (reliability ratio = 1.0 — what it knows, it always knows; what it doesn't, it always doesn't) but middling on accuracy and latency.

## Caveats / limits

- Only 4 tasks evaluated (one was filtered as a regex-rubric task, not exact_match). Larger PBS subset needed before drawing general conclusions.
- All three models are at Q4_K_M quant; quant sensitivity is a separate question (planned for EXP-001).
- Only one config (`greedy`, T=0). High-temperature sampling would change the picture entirely — and that's the point: every claim needs the config that produced it pinned.
- LLM-as-judge evaluator was registered but skipped this run (`--no-judge`) to conserve Ollama Cloud budget. Will run separately to calibrate.
- The 60% `tokens_under` for qwen3 isn't a failure of qwen3 — it's a budget-design choice. The default 500-token threshold is too tight for a reasoning model.

## Implications

- **Pre-registered metric for variance discipline is justified.** Without N≥8 + pass^k, we'd be reporting unreliable results and not know it.
- **Reasoning mode matters more than parameter count** for tasks that require multi-step arithmetic. A 14B-Q4 model with reasoning beat 8B and 12B models without reasoning, but it's paying ~50× more tokens for the privilege.
- **The qwen3 reasoning vs no-reasoning ablation** is the obvious follow-up: re-run the same sweep with `/no_think` (or qwen3's reasoning-off flag) to isolate the reasoning contribution from the model contribution.

## Open questions

- What does qwen3 score on math-001 without reasoning? Would it still get `148`?
- For gemma3 / llama3.1, does adding a CoT prompt close the gap, or is there a model-capability ceiling?
- How does this generalize to a larger task pool (the full PBS v0.1 = 24 tasks)?

## Status

- [x] Logged
- [ ] Replicated
- [ ] Published
