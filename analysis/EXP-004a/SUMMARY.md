# EXP-004a — reranker validation — SUMMARY

N queries: 50  KB: bash  Rerank model: Qwen/Qwen3-Reranker-0.6B


## Per-cell metrics

| cell | fusion | alpha | stage-1 top-k | rerank | final-k | recall@5 | MRR@10 | nDCG@10 | gold-in-pool | errors | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C0_alpha_baseline | alpha | 0.75 | 5 | no | 5 | 0.8000 | 0.6767 | 0.7075 | 0.8000 | 0 | 0.0 |
| C1_rrf_baseline | rrf |  | 5 | no | 5 | 0.7800 | 0.6123 | 0.6545 | 0.7800 | 0 | 0.0 |
| C2_alpha_rerank | alpha | 0.75 | 50 | yes | 5 | 0.8400 | 0.6703 | 0.7123 | 0.9400 | 0 | 35.5 |
| C3_rrf_rerank | rrf |  | 50 | yes | 5 | 0.8400 | 0.6647 | 0.7079 | 0.9400 | 0 | 38.7 |

## Hypothesis verdicts

- H1 (aggressive, ≥0.92 best reranked): **REFUTED**  max(C2,C3)=0.840; C0 baseline=0.800; delta=+0.040
- H2 (rerank always improves, paired Wilcoxon both p<0.05): **REFUTED**  C2 vs C0 p=0.3125; C3 vs C1 p=0.1875
- H3 (informational): delta_alpha (C1-C0)=-0.020; delta_rerank_arm (C3-C2)=+0.000

## Rerank-service stats

- calls: 100
- errors: 0 (0.0%)
- latency p50: 735.4 ms
- latency p95: 1313.2 ms
- latency mean: 741.9 ms
- latency max: 2478.0 ms
