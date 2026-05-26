# EXP-003a — verdicts

Pre-registered rules in docs/exp/EXP-003a.md §Success/failure criteria.


## H1 — Hybrid beats both endpoints on recall@5

**Verdict: CONFIRMED**

- best alpha by mean(recall@5): 0.75
- mean(recall@5) at best alpha: 0.820
- recall@5 by alpha:
  - alpha=0.00: 0.600
  - alpha=0.25: 0.740
  - alpha=0.50: 0.800
  - alpha=0.75: 0.820
  - alpha=1.00: 0.780

## H2 — Top-k matters meaningfully (recall@10 − recall@5 ≥ 0.10)

**Verdict: REFUTED**

- alpha used: 0.75  (H1 alpha if confirmed, else 0.5)
- recall@5: 0.820
- recall@10: 0.860
- delta: 0.040  (threshold 0.100)

## H3 — BM25 (alpha=0.0) plausibly competitive vs dense (alpha=1.0)

**Verdict: REFUTED**

- BM25 recall@5: 0.600
- Dense recall@5: 0.780
- BM25 best MRR@10 (over k): 0.453
- Dense best MRR@10 (over k): 0.611
