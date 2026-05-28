# EXP-005 — BFCL v3 external benchmark — summary

Cells: total=4000 done=4000 error=0 (0.0%)


## Per-model overall accuracy (n=1000 per model)

| model | n | mean | 95% CI |
|---|---|---|---|
| glm-5.1-cloud | 1000 | 0.9250 | [0.908, 0.941] |
| gpt-oss-120b-cloud | 1000 | 0.5220 | [0.491, 0.555] |
| gpt-oss-20b-cloud | 1000 | 0.5200 | [0.489, 0.552] |
| qwen3-14b-q4 | 1000 | 0.9100 | [0.892, 0.927] |

## Per-(model, category) accuracy

| model | simple | multiple | parallel | parallel_multiple |
|---|---|---|---|---|
| glm-5.1-cloud | 0.943 | 0.940 | 0.920 | 0.880 |
| gpt-oss-120b-cloud | 0.858 | 0.895 | 0.000 | 0.000 |
| gpt-oss-20b-cloud | 0.860 | 0.880 | 0.000 | 0.000 |
| qwen3-14b-q4 | 0.927 | 0.935 | 0.885 | 0.875 |

## Hypothesis verdicts

- **H1** (cloud_best - dense >= 0.10): REFUTED. Best cloud = glm-5.1-cloud (0.9250); dense = qwen3-14b-q4 (0.9100); delta = 0.0150.

- **H2** (qwen3-14b-q4 in [0.35, 0.65]): REFUTED. Measured 0.9100.

- **H3** (model ordering): REFUTED. Measured: gpt-oss-120b-cloud(0.522) >= glm-5.1-cloud(0.925) >= gpt-oss-20b-cloud(0.520) >= qwen3-14b-q4(0.910).

- **H4** (per-category profile): REFUTED.
  - Failing models: ["gpt-oss-120b-cloud: ['simple=0.858', 'multiple=0.895', 'parallel=0.000', 'parallel_multiple=0.000']", "gpt-oss-20b-cloud: ['simple=0.860', 'multiple=0.880', 'parallel=0.000', 'parallel_multiple=0.000']", "qwen3-14b-q4: ['simple=0.927', 'multiple=0.935', 'parallel=0.885', 'parallel_multiple=0.875']"]


## Headline

glm-5.1-cloud (0.925) > qwen3-14b-q4 (0.910) on BFCL v3 AST — H1 REFUTED (delta +0.015).
