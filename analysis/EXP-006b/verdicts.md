# EXP-006b verdicts

Pre-registered decision rules, applied to the post-sweep DB read.

## H1 — Baseline measurement (NOT a gate)

Pre-reg: report mean(end_state | qwen3-14b-q4) over all 96 dense cells
with a 95% bootstrap CI. No pass/fail threshold.

- n = 96
- mean end_state = 0.6667
- 95% bootstrap CI = [0.5729, 0.7604]

This number is the new lab reference for the post-fix PBS-Agent v0.1
surface. It supersedes F-005's 0.750 anchor and EXP-006's 0.583
measurement (both of which were on different surfaces).

## H2 — Headline (RELATIVE DELTA — promotion gate)

Rule: `lower_95_CI(end_state | qwen3-30b-a3b-moe, n=96)` >=
`mean(end_state | qwen3-14b-q4, n=96) + 0.10`.

- n = 96
- mean end_state = 0.8333
- 95% bootstrap CI = [0.7500, 0.9062]
- lower CI bound = 0.7500
- threshold = dense_pe + 0.10 = 0.7667

Verdict: **REFUTED**.

## H3 — Gap closure (RATIO — promotion gate)

Rule: `gap_closure_pe := (moe_pe - dense_pe) / (cloud_pe - dense_pe) >= 0.50`,
all on the same 96-cell denominator (12 tasks x 8 seeds).

- dense  end_state = 0.6667
- moe    end_state = 0.8333
- cloud  end_state = 0.9688
- denom (cloud - dense) = 0.3021
- numer (moe - dense)   = 0.1667
- gap_closure_pe = 0.5517

Verdict: **CONFIRMED**.

## H4 — Tool-correctness ceiling (RELAXED — promotion gate)

Rule: `lower_95_CI(tool_correctness | qwen3-30b-a3b-moe, n=96) >= 0.90`.

- n = 96
- mean tool_correctness = 0.9167
- 95% bootstrap CI = [0.8646, 0.9688]
- lower CI bound = 0.8646
- threshold = 0.90

Verdict: **REFUTED**.

Cloud-arm reference (n = 96): end_state = 0.9688.

## Promotion rule

Pre-registered rule: promote MoE iff H2 AND H3 both pass. If H4 also
passes, promotion is quality-clean; if H4 fails but H2 + H3 pass,
promote with H4 recorded as a quality caveat (follow-up: MoE template
audit).

- H2 pass: False
- H3 pass: True
- H4 pass: False
- Promote: **NO**
