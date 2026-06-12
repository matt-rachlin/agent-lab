# EXP-009 / HARD-BENCH-003 — summary

| model | pass@1 | 95% CI | pass^4 | pass^8 | seed spread |
| --- | --- | --- | --- | --- | --- |
| devstral-24b | 0.520 | [0.355, 0.684] | 0.471 | 0.469 | 0.125 |
| gemma4-12b | 0.914 | [0.820, 0.988] | 0.878 | 0.875 | 0.062 |
| qwen3-coder-30b | 0.746 | [0.590, 0.875] | 0.734 | 0.719 | 0.031 |

## Verdicts

- **H1**: INCONCLUSIVE (ordering holds, CIs overlap)
- **H2**: REFUTED for qwen3-coder-30b
- **H3**: REFUTED (spread outside [0.02, 0.06])
- **H4**: CONFIRMED
- **H5**: CONFIRMED
