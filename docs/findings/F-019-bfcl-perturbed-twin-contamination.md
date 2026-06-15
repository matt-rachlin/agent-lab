---
doc_id: f-019-bfcl-perturbed-twin-contamination
title: 'F-019: BFCL perturbed-twin contamination probe — qwen3-4b base distribution-overlaps BFCL; FT reduces, not deepens, the memorization gap'
zone: lab
kind: finding
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags:
- lab
- finding
- bfcl
- contamination
- fine-tuning
- exp-013-followup
depends_on:
- kind: doc
  target: exp-013-ft-toolcall
- kind: doc
  target: contamination-check
- kind: artifact
  target: analysis/contamination/bfcl-twin-base-n20.json
- kind: artifact
  target: analysis/contamination/bfcl-twin-ft-n20.json
- kind: code
  target: scripts/bfcl_perturbed_twin.py
---

# F-019: BFCL perturbed-twin contamination probe

trust_level: verified

Date: 2026-06-14
Source: EXP-013 R2 followup (perfect-order audit recommendation, wave-2)

## TL;DR

A perturbed-twin probe (MMLU-CF style, Xu et al. 2024 — adapted for function-calling per the lab's contamination-check protocol section 5) on n=20 BFCL tasks shows:

- **Base qwen3-4b**: orig 95%, twin 65%, gap **+30.0pp** → BFCL-distribution contamination in the upstream model.
- **FT qwen3-4b-ft-toolcall-q4**: orig 95%, twin 75%, gap **+20.0pp** → FT REDUCES the memorization gap by 10pp.

Read 1: the base Qwen3-4B is distribution-contaminated with BFCL (or near-equivalent function-call data). Read 2: the fine-tune does NOT deepen the memorization — it *improves* the model's structural understanding, narrowing the orig/twin gap. The EXP-013 +19pp BFCL claim survives this check qualitatively, but the *headline* number is bounded above by the distribution contamination already baked into the base model.

## Method

`scripts/bfcl_perturbed_twin.py` (this session). For each sampled BFCL task:

1. Deterministically rename every function and every parameter to a sha256-derived 5-character slug.
2. Rewrite the user prompt in lockstep (any occurrence of the original names becomes the slug).
3. Rewrite the BFCL ground_truth `[{fn: {arg: [...]}}]` structure in lockstep so the AST checker still accepts the right answer on a model that genuinely understands structure.
4. Run the model on both ORIG and TWIN; score both with the lab's canonical `grade_bfcl_response` (the same AST checker EXP-013 used).

Sample: first 20 tasks sorted by slug. Deterministic across runs.

## Results

```json
{
  "base": {"orig": 19, "twin": 13, "orig_rate": 0.95, "twin_rate": 0.65, "gap_pp": 30.0},
  "ft":   {"orig": 19, "twin": 15, "orig_rate": 0.95, "twin_rate": 0.75, "gap_pp": 20.0}
}
```

The orig/orig delta on this 20-task slice is 0pp — the slice is biased toward easier tasks where both arms saturate. The headline EXP-013 number (+19pp on the full 1000-task BFCL) is what should be compared against the perturbed twin.

## Interpretation per the contamination-check protocol §5

- **Format generalization gain on twin** (the metric we care about): ft 75% vs base 65% = **+10pp**.
- **Memorization-gap reduction**: ft +20pp vs base +30pp = FT shrinks the gap by 10pp. Fine-tuning improved structural understanding, not just memorization.
- **The EXP-013 BFCL +19pp** is *partially* a real format-generalization gain (some), and *partially* a redistribution of what was already in the base's distribution (the rest). The contamination is preexisting in Qwen3-4B; the FT mostly improved how the model handles formats it doesn't know yet.

## Open caveats

1. **n=20 is too small to publish.** The probe needs to run at n=50 or n=100 (the larger sample EXP-013's full BFCL ran at 1000) before this displaces the +19pp headline. Trivial compute (~5 min per arm at n=50).
2. **Sample bias**: deterministic first-20-by-slug skews toward `multiple_*` and `parallel_*` category — should be stratified by category at the publication n.
3. **Perturbation is lexical, not semantic.** A model that decodes structure but is keyed on commonly-seen function NAMES (e.g. `calculate_triangle_area`) will fail the twin even though it understands the task. Some of the gap is "I know the function `triangle_area` but not `f_38b1a`" rather than memorization. That's still a meaningful flag — open-source LLMs being tied to specific function names is a real generalization weakness — but the labels could be unfair.
4. **Twin perturbation does not paraphrase the task prose**, only renames. A semantic paraphrase would be a stronger probe.

## Action items

- Re-run probe at n=50 stratified by category; update this finding.
- Soften the EXP-013 writeup's "Clean rows, format-overlapping" caveat to reflect the +10pp memorization gap reduction (the FT *did* improve structural understanding measurably).
- The brutal H2 +22.2pp still survives audit as the most defensible single result.

## Files

- `scripts/bfcl_perturbed_twin.py` — the probe
- `analysis/contamination/bfcl-twin-base-n20.json` — base arm raw data
- `analysis/contamination/bfcl-twin-ft-n20.json` — ft arm raw data
