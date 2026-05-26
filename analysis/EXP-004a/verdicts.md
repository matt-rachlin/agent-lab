# EXP-004a — verdicts

Pre-registered in `docs/exp/EXP-004a.md`.


## H1 — best reranked cell ≥ 0.92 (i.e. +10pp over C0=0.820)

**Verdict: REFUTED**

- C0 (alpha=0.75, no rerank) recall@5: 0.800
- C1 (RRF, no rerank) recall@5: 0.780
- C2 (alpha=0.75 + rerank) recall@5: 0.840
- C3 (RRF + rerank) recall@5: 0.840
- max(C2, C3) = 0.840; threshold = 0.920
- delta over C0: +0.040

## H2 — rerank always improves (paired Wilcoxon, one-sided, both p<0.05)

**Verdict: REFUTED**

- C2 vs C0: +3 / -1 / ties=46; Wilcoxon one-sided p = 0.3125
- C3 vs C1: +4 / -1 / ties=45; Wilcoxon one-sided p = 0.1875

## H3 — RRF beats alpha-blend as stage-1 (informational)

- delta_alpha (C1 - C0): -0.020
- delta_rerank_arm (C3 - C2): +0.000
- NOTE: at least one comparison favors alpha-blend; see SUMMARY.md
