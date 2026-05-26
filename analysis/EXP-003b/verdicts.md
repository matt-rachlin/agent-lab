# EXP-003b verdicts — 240 cells (239 done, 1 error)

## H1 — Locals gain more from kb_query than cloud (delta_local - delta_cloud >= 0.10)

- delta_local  (local with - local without):   **0.650**  (with=0.650, without=0.000, n=80)
- delta_cloud  (cloud with - cloud without):   **0.167**  (with=0.833, without=0.667, n=120)
- difference: **0.483**  (threshold 0.100)
- **H1: CONFIRMED**

## H2 — Models actually call kb_query when available (mean >= 1.0 per (model, task) with-kb cell)

- (model, task) cells checked: 30
- FAILING cells (mean kb_query calls < 1.0):
  - llama3.1-8b-q4 / rag-bash-faithful-answer-shopt: mean=0.00
  - gpt-oss-20b-cloud / rag-bash-cite-section-for-arrays: mean=0.75
  - gpt-oss-20b-cloud / rag-bash-redirection-operator: mean=0.75
- **H2: REFUTED**

## H3 — Faithfulness improves with kb_query on rag-bash-faithful-answer-shopt (delta >= 0.10)

- with-kb mean faithfulness:   **0.267** (n=15)
- without-kb mean faithfulness: **nan** (n=0)
- delta: **nan**  (threshold 0.100)
- **H3: UNDEFINED**

## H4 — At least one (model, task) cell in without-kb with mean(end_state) <= 0.25 on a KB task

- Failing (model, task) cells in without-kb:
  - qwen3-14b-q4 / rag-bash-compare-test-bracket: mean end_state = 0.000 (n=4)
  - qwen3-14b-q4 / rag-bash-param-expansion-forms: mean end_state = 0.000 (n=4)
  - qwen3-14b-q4 / rag-bash-redirection-operator: mean end_state = 0.000 (n=4)
  - llama3.1-8b-q4 / rag-bash-compare-test-bracket: mean end_state = 0.000 (n=4)
  - llama3.1-8b-q4 / rag-bash-param-expansion-forms: mean end_state = 0.000 (n=4)
  - llama3.1-8b-q4 / rag-bash-redirection-operator: mean end_state = 0.000 (n=4)
  - gpt-oss-20b-cloud / rag-bash-compare-test-bracket: mean end_state = 0.250 (n=4)
  - glm-5.1-cloud / rag-bash-compare-test-bracket: mean end_state = 0.000 (n=4)
  - gpt-oss-120b-cloud / rag-bash-param-expansion-forms: mean end_state = 0.000 (n=4)
- **H4: CONFIRMED**

