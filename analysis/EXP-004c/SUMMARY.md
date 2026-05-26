# EXP-004c — reranker validation at higher N — SUMMARY

N queries: 199  KB: bash


## Per-cell metrics

| cell | mode | rerank model | trunc | recall@5 | MRR@10 | nDCG@10 | gold-in-pool | errors | wall (s) | rerank p50 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| B0_alpha_baseline | — | Qwen/Qwen3-Reranker-0.6B | — | 0.7990 | 0.6920 | 0.7190 | 0.7990 | 0 | 0.0 | 0.0 |
| Q1_qwen3_1500c | rpc | Qwen/Qwen3-Reranker-0.6B | 1500 | 0.8492 | 0.6914 | 0.7313 | 0.8995 | 0 | 131.1 | 686.4 |
| Q2_qwen3_2500c | rpc | Qwen/Qwen3-Reranker-0.6B | 2500 | 0.8543 | 0.6982 | 0.7376 | 0.8995 | 0 | 167.2 | 896.0 |
| Q3_qwen3_notrunc | rpc | Qwen/Qwen3-Reranker-0.6B | none | 0.5427 | 0.4579 | 0.4793 | 0.8995 | 77 | 143.1 | 619.8 |
| Q4_qwen3_1500c_inproc | inproc | Qwen/Qwen3-Reranker-0.6B | 1500 | 0.8492 | 0.6914 | 0.7313 | 0.8995 | 0 | 134.8 | 685.6 |
| B1_bge_1500c_inproc | inproc | BAAI/bge-reranker-v2-m3 | 1500 | 0.8241 | 0.6717 | 0.7101 | 0.8995 | 0 | 184.7 | 1040.6 |

## Hypothesis verdicts

- **H1** (best reranked ≥ 0.92, +10pp over B0=0.799): **REFUTED**  max(rerank cells) = 0.854; delta over B0 = +0.055
- **H2** (truncation monotone Q3>Q2>Q1): **REFUTED**  Q3=0.543, Q2=0.854, Q1=0.849 — not strictly increasing
- **H3** (|Q4-Q1| ≤ 0.02 — RPC overhead): **CONFIRMED**  Q4=0.849, Q1=0.849, delta=+0.000
- **H4** (Qwen3 vs BGE, inproc): winner = **tie (keep Qwen3)**  Q4=0.849, B1=0.824, delta=+0.025

## Wilcoxon vs B0 (one-sided, treat > control)

- Q1_qwen3_1500c: +12 / -2 / ties=185; p = 0.0038
- Q2_qwen3_2500c: +13 / -2 / ties=184; p = 0.0023
- Q3_qwen3_notrunc: +8 / -59 / ties=132; p = 1.0000
- Q4_qwen3_1500c_inproc: +12 / -2 / ties=185; p = 0.0038
- B1_bge_1500c_inproc: +9 / -4 / ties=186; p = 0.1334

## Rerank-service stats (RPC cells only)

- calls: 995
- errors: 77 (7.7%)
- latency p50: 759.4 ms
- latency p95: 1307.5 ms
- latency mean: 764.6 ms
- latency max: 6662.0 ms
