---
doc_id: exp-004c-verdicts
title: EXP-004c — verdicts
zone: lab
kind: guide
status: active
owner: m
created: '2026-05-26'
last_updated: '2026-05-26'
last_verified: '2026-05-26'
tags:
- lab
- guide
- analysis
- exp-004c
---
# EXP-004c — verdicts

Pre-registered in `docs/exp/EXP-004c.md`.


## H1 — best reranked cell ≥ 0.92 (i.e. +10pp over B0)

**Verdict: REFUTED**

- B0 (alpha=0.75, no rerank) recall@5: 0.799
- Q1 (Qwen3 + 1500c, rpc) recall@5: 0.849
- Q2 (Qwen3 + 2500c, rpc) recall@5: 0.854
- Q3 (Qwen3 + no-trunc, rpc) recall@5: 0.543
- Q4 (Qwen3 + 1500c, inproc) recall@5: 0.849
- B1 (BGE + 1500c, inproc) recall@5: 0.824
- max(rerank cells) = 0.854; threshold = 0.920
- delta over B0: +0.055

## H2 — truncation monotone: Q3 > Q2 > Q1

**Verdict: REFUTED**

- Q3=0.543, Q2=0.854, Q1=0.849 — not strictly increasing

## H3 — RPC overhead: |Q4 - Q1| ≤ 0.02

**Verdict: CONFIRMED**

- Q4 (in-process) recall@5: 0.849
- Q1 (rpc) recall@5: 0.849
- delta (Q4 - Q1): +0.000  (threshold ±0.020)

## H4 — rerank-model comparison: Q4 (Qwen3 inproc) vs B1 (BGE inproc)

**Winner: tie (keep Qwen3)**

- Q4 (Qwen3-Reranker-0.6B): 0.849
- B1 (bge-reranker-v2-m3): 0.824
- delta (Q4 - B1): +0.025  (threshold ±0.050)

## Wilcoxon vs B0 (one-sided, treat > control)

- Q1_qwen3_1500c: +12 / -2 / ties=185; p = 0.0038
- Q2_qwen3_2500c: +13 / -2 / ties=184; p = 0.0023
- Q3_qwen3_notrunc: +8 / -59 / ties=132; p = 1.0000
- Q4_qwen3_1500c_inproc: +12 / -2 / ties=185; p = 0.0038
- B1_bge_1500c_inproc: +9 / -4 / ties=186; p = 0.1334
