# EXP-006 — SUMMARY

Cells: 288/288 (dense=96, moe=96, cloud=96, errored=0).

## Pre-registered hypotheses

- **H1 — Replication.** qwen3-14b-q4 end_state = **0.583** [0.479, 0.688] ; target 0.750 ± 0.05. → **REFUTED**
- **H2 — Headline lift.** qwen3-30b-a3b-moe end_state = **0.583** [0.490, 0.677] ; threshold ≥ 0.850. → **INVALID — H1 replication failed; H2 result not load-bearing**
- **H3 — Gap closure.** gap_closure = **0.000** (dense=0.583, moe=0.583, cloud=0.969); threshold ≥ 0.50. → **INVALID — H1 replication failed; H3 result not load-bearing**
- **H4 — Tool-correctness ceiling.** qwen3-30b-a3b-moe tool_correctness = **0.500** [0.406, 0.594] ; threshold ≥ 0.95. → **INVALID — H1 replication failed; H4 result not load-bearing**

## Headline

Sweep INVALID — H1 (replication of F-005's qwen3-14b-q4 baseline) is outside the ±0.05 pp band. H2/H3/H4 verdicts are not load-bearing.
