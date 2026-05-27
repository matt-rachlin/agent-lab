# EXP-006 verdicts

Pre-registered decision rules, applied to the post-sweep DB read.

## H1 — Replication of F-005 qwen3-14b-q4 baseline

Rule: `|mean(end_state | qwen3-14b-q4, all 96 cells) − 0.750| ≤ 0.05`.

- n = 96
- mean end_state = 0.5833
- 95% bootstrap CI = [0.4792, 0.6875]
- |observed − anchor| = 0.1667
- pre-reg band = 0.05

Verdict: **REFUTED — sweep INVALID**.

## H2 — Headline lift (qwen3-30b-a3b-moe end_state ≥ 0.850)

Rule: `mean(end_state | qwen3-30b-a3b-moe, all 96 cells) ≥ 0.850`.

- n = 96
- mean end_state = 0.5833
- 95% bootstrap CI = [0.4896, 0.6771]
- threshold = 0.850

Verdict: **REFUTED**.

## H3 — Gap closure (≥ 0.50)

Rule: `gap_closure := (moe − dense) / (cloud − dense) ≥ 0.50`,
all three terms on the same 96-cell denominator (12 tasks × 8 seeds).

- dense  end_state = 0.5833
- moe    end_state = 0.5833
- cloud  end_state = 0.9688
- denom (cloud − dense) = 0.3854
- numer (moe − dense)   = 0.0000
- gap_closure = 0.0000

Verdict: **REFUTED**.

## H4 — Tool-correctness ceiling (qwen3-30b-a3b-moe ≥ 0.95)

Rule: `mean(tool_correctness | qwen3-30b-a3b-moe, all 96 cells) ≥ 0.95`.

- n = 96
- mean tool_correctness = 0.5000
- 95% bootstrap CI = [0.4062, 0.5938]
- threshold = 0.95

Verdict: **REFUTED**.

Cloud-arm reference (n = 96): end_state = 0.9688.
