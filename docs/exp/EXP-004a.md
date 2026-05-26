# EXP-004a: Reranker validation sweep on bash KB (synthetic-only)

Date created: 2026-05-26
Status: planned
Pre-registered: (commit SHA filled by `lab exp register` at registration time)

## Question

Does adding the Qwen3-Reranker-0.6B cross-encoder as a stage-2 reranker
improve synthetic-query retrieval on the bash KB over the best stage-1
hybrid configuration from EXP-003a?

This is a pure retrieval benchmark — no agent loop, no LLM generation,
no tool call. Each cell runs stage-1 hybrid retrieval, optionally
posts the candidate pool to the host-side rerank service
(`LAB_RAG_RERANKER_URL=http://127.0.0.1:8401`), and measures whether
each query's originating chunk surfaces in the final top-5.

EXP-004b (agent sweep, separate run) is **explicitly out of scope here**;
user scoped 004a to synthetic-only, ~10 min wall.

## Setup

- **KB**: sealed `bash` KB, 4,620 indexed chunks, qwen3-embedding:8b-q8_0
  (4,096-dim), LanceDB + per-row JSON sparse vectors.
- **Reranker**: `Qwen3-Reranker-0.6B` (Apache 2.0, ~1.2 GB VRAM at load),
  served by `lab.rag.rerank_server` on `127.0.0.1:8401`. All rerank calls
  go over HTTP to that singleton — **no in-process model load** by the
  runner. If the service is unreachable at start, STOP.
- **Queries**: reuse the same 50 synthetic queries from EXP-003a at
  `analysis/EXP-003a/queries.jsonl`. **Do NOT regenerate.**
- **Embedder**: same `qwen3-embedding:8b-q8_0`, queries embedded once
  and reused across cells.

## Matrix (4 cells)

| cell | stage-1 fusion | stage-1 top-k | stage-2 rerank | final top-k |
|---|---|---|---|---|
| C0 baseline | alpha-blend (α=0.75) | 5 | none | 5 |
| C1 RRF baseline | RRF (k=60) | 5 | none | 5 |
| C2 alpha + rerank | alpha-blend (α=0.75) | 50 | Qwen3-Reranker-0.6B | 5 |
| C3 RRF + rerank | RRF (k=60) | 50 | Qwen3-Reranker-0.6B | 5 |

C0 is the EXP-003a winning baseline (recall@5 = 0.820 published).
C1 establishes the RRF stage-1 baseline (the new default shipped in
Phase 7). C2 and C3 are the rerank treatments — they pull a top-50
candidate pool from stage-1 and let the cross-encoder pick the final 5.

## Method

The runner (`scripts/retrieval_sweep.py`) detects the per-cell config
shape in `conf/sweep/EXP-004a.yaml` and dispatches to the per-cell
sweep code path. For each cell:

1. Stage-1 retrieval per the cell's `fusion` (`alpha`-blend or `rrf`)
   with `top_k_stage1` candidates (5 for baselines, 50 for rerank cells).
   Stage-1 results are cached per (fusion, alpha, top_k_stage1) and
   reused across cells that share the same stage-1 config.
2. If `rerank` is set, POST stage-1 candidates to the host-side
   rerank service at `http://127.0.0.1:8401/rerank` and take the
   server-returned top `final_k` (=5).
3. Per query: rank of the origin chunk in the final list → recall@5,
   MRR@10, nDCG@10.
4. Per cell: mean of each metric, plus per-cell error count + wall
   time + `gold_in_stage1_pool` fraction (diagnostic — if the pool
   doesn't include the gold chunk, reranking cannot help).

## Pre-registered hypotheses

- **H1 (aggressive, per plan)**: the best reranked cell —
  `max(recall@5(C2), recall@5(C3))` — is **≥ 0.92**, i.e. adds **≥ 10pp
  absolute** over the C0 baseline (0.820).
- **H2**: reranking always improves recall@5 — both C2 > C0 and
  C3 > C1 on mean recall@5. Paired Wilcoxon signed-rank tests over the
  50 queries: both p < 0.05.
- **H3 (informational, not a kill criterion)**: RRF beats alpha-blend
  as stage-1 — C1 ≥ C0 and C3 ≥ C2 on mean recall@5. Descriptive,
  no significance threshold.

## Metrics

For each cell, per query: was the origin chunk in the final top-5?
And at what rank (1-5, or 0 = miss)?

- **recall@5** (primary) — fraction of queries whose origin chunk is in
  the final top-5.
- **MRR@10** — mean reciprocal rank, capped at 10. For C0/C1 (top-k=5)
  the cap is structurally 5; for C2/C3 (rerank from top-50) the rerank
  returns the requested final top-k=5, so MRR is also bounded at 5 in
  practice. We retain the @10 notation for symmetry with EXP-003a but
  the effective cap is the final top-k.
- **nDCG@10** — same cap caveat; `1/log2(1 + rank)` if hit in top-k,
  else 0.

Per-cell mean + per-query rows written to `analysis/EXP-004a/raw.csv`.

## Hypothesis

See "Pre-registered hypotheses" above — the three rules (H1
aggressive +10pp threshold, H2 paired Wilcoxon, H3 informational
RRF-vs-alpha-blend stage-1 comparison) are restated here only so the
plan validator finds a `## Hypothesis` heading.

## Success / failure criteria

Strictly applied AFTER the sweep + analyzer complete.

- **H1 confirmed** ⇔ `max(mean(recall@5)(C2), mean(recall@5)(C3))
  ≥ 0.92`. **REFUTED** otherwise — no soft language. If H1 is REFUTED
  we revert the rerank default (per the Phase 7 acceptance criterion).
- **H2 confirmed** ⇔ paired-Wilcoxon `recall@5(C2) > recall@5(C0)`
  **and** paired-Wilcoxon `recall@5(C3) > recall@5(C1)`, each at
  `p < 0.05` (one-sided, the alternative is "rerank improves"). The
  paired sample is per-query indicator values. If either test
  produces an undefined p-value (e.g. zero non-zero differences),
  H2 is **REFUTED** for that pair.
- **H3 informational**: report `delta_alpha = C1 - C0` and
  `delta_rerank_arm = C3 - C2`; if either is negative, note "RRF
  underperformed alpha-blend at this stage". No CONFIRMED/REFUTED
  label.

## Kill criteria

- **rerank service unreachable at sweep start** (cannot GET /healthz):
  STOP. Do NOT fall back to in-process model load.
- **rerank-service error rate > 5%** of rerank calls (target: 50 × 2
  = 100 rerank calls total): STOP and triage. Document in F-007.
- **embed-service error rate > 5%** of queries: STOP.
- **Fewer than 25 valid queries loaded from cache** (shouldn't happen
  since EXP-003a wrote 50): STOP, raise.

## Confounders to control

- **Single KB**: only the bash KB is in scope. Generalization to other
  KBs is out of scope.
- **Single rerank model**: `Qwen3-Reranker-0.6B`. The plan lists
  `bge-reranker-v2-m3` as a fallback — not run here.
- **Synthetic queries only**: reuses EXP-003a's queries.jsonl. No
  human-written queries.
- **Single gold chunk per query**: same construction as EXP-003a.
- **Same query embeddings**: regenerated this run from the cached
  question text via the same embedder; should be identical to
  EXP-003a's by construction.

## Pre-mortem

It's a day from now and EXP-004a was a methodological failure.
Plausible:

1. **Rerank service crashes mid-sweep / OOMs** when both embedder and
   reranker resident. Mitigation: reranker is on-demand (lazy load,
   idle TTL 300s); the embedder is Ollama-resident and uses
   keep_alive=5m. Sweep is single-threaded so we never pay both
   peak VRAMs simultaneously.
2. **Reranker returns degenerate scores** (e.g. all-0) when fed bash
   symbol-heavy queries. Mitigation: per-query CSV rows preserve
   rerank scores; if the top-1 score doesn't differ from the bottom-1
   we'll flag in verdicts.
3. **Top-k=50 candidate pool fails to include the gold chunk on most
   queries** (stage-1 ceiling effect). Mitigation: per-query CSV row
   carries `gold_in_stage1_pool` flag; if many queries have
   gold-not-in-pool, reranking cannot help by construction and the
   verdict is descriptive about that.

## Budget estimate

- 50 queries × 4 cells = 200 retrieval evaluations
- 50 queries × 2 rerank cells = 100 rerank calls; each ~150-400 ms
  with the 0.6B cross-encoder on a 50-doc pool
- Stage-1 cost amortised: alpha-blend and RRF reuse the same dense +
  sparse candidate pool per query, so the runner embeds queries once
  and runs stage-1 once per (fusion, top_k_stage1) combo per query.
- Estimated wall: **~10 min total** (50 embeddings ~30s, stage-1 ~60s,
  rerank ~150-300s + startup).

## Outputs

- `analysis/EXP-004a/SUMMARY.md` — per-cell metrics table + headline
  verdicts.
- `analysis/EXP-004a/verdicts.md` — per-hypothesis verdicts with
  numbers + Wilcoxon p-values for H2.
- `analysis/EXP-004a/raw.csv` — per-cell per-query results.
- `analysis/EXP-004a/per_cell.csv` — per-cell aggregated metrics +
  wall-time / error counts.
- `analysis/EXP-004a/rerank_stats.json` — rerank-service stats
  (call count, error count, p50 / p95 latency).

## Reporting

- 3 hypothesis verdicts (one line each).
- Per-cell recall@5 / MRR@10 / nDCG@10 (4 rows).
- F-007 path + claim slug.
- Commit SHA.
- Rerank-service stats during the run.

## Components NOT run end-to-end

- **EXP-004b** — agent sweep — user explicitly scoped out.
- **bge-reranker-v2-m3 fallback** — not run.
- **Other KBs** — bash only.
- **Other final top-k values** — only top-5.
