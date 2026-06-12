# EXP-010 / BRUTAL-BENCH-001 — summary

| model | overall | debug | longhaul | recovery | spec |
| --- | --- | --- | --- | --- | --- |
| devstral-24b | 0.125 | 0.167 | 0.000 | 0.167 | 0.167 |
| gemma4-12b | 0.708 | 0.833 | 0.333 | 1.000 | 0.667 |
| qwen3-coder-30b | 0.708 | 1.000 | 0.167 | 0.833 | 0.833 |

## Verdicts

- **H1**: CONFIRMED (0.708 <= 0.85)
- **H2**: INCONCLUSIVE (gemma4-vs-qwen3 tie, qwen3-vs-devstral strict)
- **H3**: INCONCLUSIVE (20/24; audit pending)
- **H4**: REFUTED (qwen3 debug >= own mean)

## Defect-audit list (4 all-models-fail tasks)

- longhaul-fx-invoice-rounding (longhaul) — audit trajectory before trusting
- longhaul-shipment-rate-pagination (longhaul) — audit trajectory before trusting
- longhaul-vendor-ledger-pagination (longhaul) — audit trajectory before trusting
- spec-sensor-window (spec) — audit trajectory before trusting
