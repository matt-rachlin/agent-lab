# EXP-003b — SUMMARY

Cells: 240 (239 done, 1 error)


## Per-(model, condition) end_state means

| model | with-kb | without-kb | delta |
|---|---|---|---|
| qwen3-14b-q4 | 0.800 (n=20) | 0.000 (n=20) | +0.800 |
| llama3.1-8b-q4 | 0.500 (n=20) | 0.000 (n=20) | +0.500 |
| gpt-oss-20b-cloud | 0.750 (n=20) | 0.600 (n=20) | +0.150 |
| glm-5.1-cloud | 0.750 (n=20) | 0.650 (n=20) | +0.100 |
| gpt-oss-120b-cloud | 1.000 (n=20) | 0.750 (n=20) | +0.250 |

## Verdicts

- **H1** (locals gain more): **CONFIRMED** (delta_local - delta_cloud = +0.483)
- **H2** (models call kb_query): **REFUTED** (3 failing cells)
- **H3** (faithfulness improves with KB): **UNDEFINED** (delta = +nan)
- **H4** (catastrophic without-KB on KB task): **CONFIRMED** (9 failing cells)
