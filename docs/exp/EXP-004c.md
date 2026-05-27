---
doc_id: exp-004c
title: 'EXP-004c: Reranker validation at higher N + setup ablations (synthetic-only)'
zone: lab
kind: exp
status: active
owner: m
created: '2026-05-26'
last_updated: '2026-05-26'
last_verified: '2026-05-26'
tags:
- lab
- exp
---
# EXP-004c: Reranker validation at higher N + setup ablations (synthetic-only)

Date created: 2026-05-26
Status: planned
Pre-registered: (commit SHA filled by `lab exp register` at registration time)

## Question

EXP-004a (N=50) REFUTED all three pre-registered hypotheses about
cross-encoder rerank lift on the bash KB. The best reranked cell
delivered +4pp recall@5 (0.800 → 0.840), short of the +10pp threshold,
and the paired-Wilcoxon test was underpowered (p=0.31 / 0.19). The
public-literature claim for 2026-era cross-encoders on technical KBs is
+24% nDCG / +40% MRR; we measured +0.6% / -0.5% nDCG/MRR. That's a
massive gap.

Before reverting the Phase 7 rerank default we want to rule out that
the setup is undersells the reranker. EXP-004c re-runs the validation
with three changes:

1. **N=200 queries** instead of 50 — paired-Wilcoxon has real power.
2. **Truncation ablation** — 3 levels (1500, 2500, no-trunc) — the
   EXP-004a 1500-char cap was set empirically after OOMs, not against
   the rerank signal. The cross-encoder may be reading a clipped
   "abstract" of long bash sections.
3. **In-process control** — load `LabReranker` in the runner directly
   for one cell, to isolate RPC / queue overhead vs. signal.
4. **Reranker-model ablation** — Qwen3-Reranker-0.6B (default) vs
   bge-reranker-v2-m3 (Phase 7 plan's fallback).

This is a pure retrieval benchmark — no agent loop, no LLM generation,
no tool call. Each cell runs stage-1 hybrid retrieval (α=0.75, the
EXP-003a winner), optionally rerank, measures whether each query's
origin chunk surfaces in the final top-5.

EXP-004b (agent sweep) remains explicitly out of scope.

## Setup

- **KB**: sealed `bash` KB, 4,620 indexed chunks, qwen3-embedding:8b-q8_0
  (4,096-dim), LanceDB + per-row JSON sparse vectors. Same KB as
  EXP-003a / EXP-003b / EXP-004a.
- **Reranker (primary)**: `Qwen/Qwen3-Reranker-0.6B`, served by
  `lab.rag.rerank_server` on `127.0.0.1:8401`. Lazy-load + idle TTL=300s.
  Cells Q1-Q3 use this over HTTP. Q4 loads the same model in-process.
- **Reranker (fallback ablation)**: `BAAI/bge-reranker-v2-m3` — Apache
  2.0, mature, BEIR 56.51. Used by B1 over HTTP. First call cold-downloads
  ~600 MB; expected and budgeted for.
- **Queries**: 200 synthetic queries on the bash KB.
  - 50 from `analysis/EXP-003a/queries.jsonl` (reused verbatim — the
    EXP-003a / EXP-004a set, identical chunk origin + question text).
  - 150 newly generated this run via the same procedure
    (`qwen3:14b-q4_K_M`, `_gen_question` in `scripts/retrieval_sweep.py`)
    against the same sealed KB, stratified by `section_path[:2]` like
    EXP-003a.
  - Cached at `analysis/EXP-004c/queries.jsonl`. **Do NOT touch the
    EXP-003a cache.** Re-runs reuse this cache; query-gen is the slow
    step (~75 min).
- **Embedder**: same `qwen3-embedding:8b-q8_0`. Queries embedded once
  and reused across all 6 cells.
- **Stage-1**: α=0.75 alpha-blend (EXP-003a's H1-winning config). RRF
  is not re-tested here — EXP-004a found it -2pp vs alpha-blend on this
  KB.

## Matrix (6 cells)

| cell | stage-1 | top-k1 | rerank model | truncation | mode |
|---|---|---|---|---|---|
| B0 baseline | alpha=0.75 | 5 | none | — | — |
| Q1 Qwen3 + 1500c | alpha=0.75 | 50 | Qwen3-Reranker-0.6B | 1500 | rpc |
| Q2 Qwen3 + 2500c | alpha=0.75 | 50 | Qwen3-Reranker-0.6B | 2500 | rpc |
| Q3 Qwen3 + no-trunc | alpha=0.75 | 50 | Qwen3-Reranker-0.6B | none | rpc |
| Q4 Qwen3 + 1500c in-proc | alpha=0.75 | 50 | Qwen3-Reranker-0.6B | 1500 | inproc |
| B1 BGE + 1500c | alpha=0.75 | 50 | bge-reranker-v2-m3 | 1500 | inproc |

6 cells × 200 queries = 1,200 evaluations. 1,000 rerank calls (5
rerank cells × 200).

- B0 is the baseline replicating EXP-004a C0 at N=200.
- Q1 replicates EXP-004a C2 at N=200 (the headline cell).
- Q2 / Q3 isolate the truncation factor.
- Q4 isolates RPC overhead — for this cell the runner constructs
  `LabReranker(model_name="Qwen/Qwen3-Reranker-0.6B")` directly with
  `LAB_RAG_RERANKER_URL` cleared in the local environment, so the
  client falls through to the in-process predict path.
- B1 swaps the reranker model only. **Mode: inproc** because the
  host-side rerank service is configured at startup for a single model
  (Qwen3-Reranker-0.6B) and returns 409 on a different `model` field
  in the request payload. Loading BGE in-process via `LabReranker`
  matches the Q4 execution path exactly — the model is the only varied
  factor between Q4 (Qwen3 inproc) and B1 (BGE inproc).

## Method

Runner: extended `scripts/retrieval_sweep.py`. EXP-004c-shape detected
by per-cell `rerank_model` / `truncation` / `mode` fields in the cells
list (back-compat with EXP-004a-shape).

Per cell:

1. Stage-1 alpha-blend (α=0.75, top-50) — cached once and reused across
   all 5 rerank cells.
2. Apply per-cell passage truncation when assembling the rerank
   payload: `text[:1500]`, `text[:2500]`, or full passage.
3. Rerank by mode:
   - `rpc`: POST stage-1 candidates to `http://127.0.0.1:8401/rerank`
     with the cell's `rerank_model`. The server returns 409 on model
     mismatch; the runner does a per-cell `/healthz` warmup so the
     server's lazy load completes before we start timing.
   - `inproc`: construct `LabReranker(model_name=...)`, call
     `.rerank(query, candidates, top_n=5)` directly. The runner
     temporarily unsets `LAB_RAG_RERANKER_URL` so the client doesn't
     dispatch over HTTP. Single model instance reused across all queries
     for this cell.
4. Per query: rank of origin chunk in final list → recall@5, MRR@10,
   nDCG@10.
5. Per cell: mean of each metric + per-cell error count + wall time +
   `gold_in_stage1_pool` fraction. Per-request latency split
   (`queue_wait`, `model_infer`, `network`) recorded for RPC cells from
   the rerank_server timings header (or, if absent, derived from local
   client measurements).

Pre-cell housekeeping:

- Before any rerank cell, POST `keep_alive=0` to the embedder model so
  Ollama frees VRAM. EXP-004a established this is necessary on a 12 GB
  GPU.
- Before Q4 (in-process), same `keep_alive=0` pulse plus an explicit
  reset of `LAB_RAG_RERANKER_URL` so the in-process predict path is
  exercised.
- Between Q3 (no-truncation) and the next cell, restart the rerank
  service if its VRAM did not drop after the idle TTL — manual
  intervention point if the next cell OOMs on warmup. (This is not
  automated by the runner.)

## Pre-registered hypotheses

- **H1 (replication of EXP-004a's H1 at higher N)**: the best
  rerank cell over all 5 variants —
  `max(recall@5(Q1), Q2, Q3, Q4, B1)` — is **≥ 0.92**, i.e. clears
  the EXP-004a pre-registered +10pp threshold over B0=0.820 (the
  published EXP-003a C0). If CONFIRMED at N=200, the EXP-004a result
  was statistical noise.
- **H2 (truncation effect)**: longer truncation → higher recall@5.
  Specifically: `recall@5(Q3) > recall@5(Q2) > recall@5(Q1)`, strict
  monotonic. If REFUTED, truncation is not the bottleneck on bash KB.
- **H3 (RPC overhead)**: Q4 (in-process) and Q1 (RPC, same model +
  truncation) within **2pp recall@5** of each other:
  `|recall@5(Q4) - recall@5(Q1)| ≤ 0.02`. If REFUTED with `Q4 >> Q1`,
  the rerank server is eating signal; if `Q1 >> Q4`, the in-process
  path has a bug. Either way, this is the cell that tells us the
  service architecture isn't the problem.
- **H4 (rerank-model comparison)**: between **Q4** (Qwen3-Reranker-0.6B
  inproc) and **B1** (bge-reranker-v2-m3 inproc) — same execution
  path, model the only varied factor — whichever wins by ≥5pp recall@5
  becomes the recommended default. Tie within ±5pp → keep Qwen3 (it's
  already the configured default and the smaller model). If B1 wins by
  ≥5pp, the recommendation is to swap.

## Metrics

For each cell, per query: was the origin chunk in the final top-5?
And at what rank (1-5, or 0 = miss)?

- **recall@5** (primary) — fraction of queries whose origin chunk is in
  the final top-5.
- **MRR@10** — mean reciprocal rank, capped at 10 (effective cap = 5
  since final top-k=5).
- **nDCG@10** — same effective cap.

Per-cell aggregates + per-query rows written to
`analysis/EXP-004c/raw.csv` for downstream paired Wilcoxon on any pair
of cells.

## Hypothesis

See "Pre-registered hypotheses" above — restated here only so the plan
validator finds a `## Hypothesis` heading.

H1 replication, H2 truncation monotone, H3 RPC overhead bounded, H4
model winner.

## Success / failure criteria

Strictly applied AFTER the sweep + analyzer complete.

- **H1 CONFIRMED** ⇔ `max(recall@5(Q1..Q4, B1)) ≥ 0.920`. **REFUTED**
  otherwise — no soft language. If REFUTED at N=200, the EXP-004a
  verdict replicates; F-007 stands and we revert the rerank default.
- **H2 CONFIRMED** ⇔ strict monotone `recall@5(Q3) > recall@5(Q2) >
  recall@5(Q1)`. Equal or non-monotone → REFUTED.
- **H3 CONFIRMED** ⇔ `|recall@5(Q4) - recall@5(Q1)| ≤ 0.02`. REFUTED
  otherwise; the gap direction is reported.
- **H4 winner** = Qwen3 if `recall@5(Q4) - recall@5(B1) ≥ 0.05`,
  BGE if `recall@5(B1) - recall@5(Q4) ≥ 0.05`, else "tie (keep
  Qwen3)". Inproc-vs-inproc isolates the model factor.

Paired-Wilcoxon p-values (one-sided, treat > control) reported for
every rerank cell vs B0 as descriptive context.

## Kill criteria

- **rerank service unreachable at sweep start** (cannot GET /healthz):
  STOP. Do NOT fall back to in-process (Q4 deliberately uses in-process,
  but if the service is down the RPC cells cannot run).
- **rerank-service error rate > 5%** of rerank calls (target: 200 × 4
  RPC rerank cells = 800 RPC calls): STOP and triage in verdicts.md.
  This is the same kill rule as EXP-004a; the EXP-004a final run had
  0/100 errors after Ollama-unload + 1500c truncation were applied.
- **embed-service error rate > 5%** of queries: STOP.
- **Fewer than 100 valid queries loaded from cache** at sweep start:
  STOP, raise. The 50 EXP-003a queries are guaranteed; this triggers
  only if query-gen produced fewer than 50 new valid questions.
- **VRAM OOM during Q3 (no-truncation)**: mark Q3 as `cell_error` and
  continue. Do not abort the sweep — Q3's failure is itself a
  legitimate finding ("no-truncation is infeasible on 12 GB").

## Confounders to control

- **Single KB**: bash only. Generalization to other KBs out of scope.
- **Single stage-1**: α=0.75 only. RRF excluded per EXP-004a's H3
  diagnostic (RRF -2pp on bash KB).
- **Single final top-k**: 5. Phase 7 ships `final_k=10`; not exercised
  here for power reasons.
- **Same embedder, same chunker**: as EXP-003a. Embedding-model and
  chunk-size ablations remain queued for EXP-003c.
- **Same query-gen model + temperature**: `qwen3:14b-q4_K_M` at
  temperature 0.3, same prompt as EXP-003a. The 50 reused EXP-003a
  queries were generated this way; the 150 new ones use the same
  procedure for parity.
- **Same sealed bash KB**: 4,620 chunks, no re-index between cells.

## Pre-mortem

It's a day from now and EXP-004c was a methodological failure.
Plausible:

1. **Query-gen produces low-quality questions at N=150** that don't
   stress retrieval — recall@5 ceiling-effect at 0.95+. Mitigation:
   per-query CSV captures origin chunk + section; we can post-hoc
   filter for "hard" queries (rank > 5 in B0) and recompute H1 on the
   hard slice as a sensitivity check.
2. **Q3 (no-truncation) OOMs all queries** — kill criterion captures
   this; result is "Q3 cell_error, infeasible". Sweep continues, H2 is
   reported with Q3 missing.
3. **bge-reranker-v2-m3 cold-download trips up the rerank service**
   (~600 MB, may take 60-120s on first call). Mitigation: pre-warm B1
   cell with a single dummy `/rerank` POST that the runner discards
   from latency stats. If the cold-download fails repeatedly, mark B1
   as `cell_error` and report.
4. **Rerank service crashes between Q3 (no-trunc) and the next cell**
   because Q3's full-passage attention matrix OOMs and corrupts state.
   Mitigation: the runner detects rerank-service errors and the user
   can manually `systemctl --user restart rerank.service` between
   cells if needed; the cache makes restart cheap.
5. **In-process Q4 conflicts with the rerank service for VRAM** —
   both want ~2.5 GiB. Mitigation: the runner unsets
   `LAB_RAG_RERANKER_URL` in-process, but the rerank service is still
   running. Q4 runs after Q1/Q2/Q3 so service has idled out (TTL=300s)
   by the time Q4 starts. If still resident, manual restart.
6. **At N=200, H1 still doesn't clear +10pp** but H2 / H3 reveal
   that truncation + RPC together account for half the missing
   signal. The replicate-vs-explain split is informative.

## Budget estimate

- Query gen: ~75 min (150 new × ~30s/each with qwen3:14b-q4_K_M)
- Embedding: ~2 min (200 queries × ~600ms)
- Stage-1 cache: ~3 min (200 queries × α=0.75 hybrid)
- Per-cell rerank: 200 queries × ~700ms p50 = ~3 min/cell × 5 cells =
  ~15 min, plus cold-loads. Budget 25 min.
- Total: ~110 min wall. Budget 2 hours.

## Outputs

- `analysis/EXP-004c/SUMMARY.md` — per-cell metrics table + headline
  verdicts.
- `analysis/EXP-004c/verdicts.md` — per-hypothesis verdicts with
  numbers + Wilcoxon p-values.
- `analysis/EXP-004c/raw.csv` — per-cell per-query results.
- `analysis/EXP-004c/per_cell.csv` — per-cell aggregated metrics +
  wall-time / error counts + latency split.
- `analysis/EXP-004c/rerank_stats.json` — rerank-service stats by
  cell (call count, error count, p50 / p95 latency).
- `analysis/EXP-004c/queries.jsonl` — the 200-query cache (50 reused
  + 150 new).

## Reporting

- 4 hypothesis verdicts (one line each).
- Per-cell recall@5 / MRR@10 / nDCG@10 (6 rows).
- F-007 status (amended in place / superseded by F-008 / stands as-is).
- Whether the revert was triggered + commit SHA if so.
- Per-cell wall time.
- Rerank-service error rate during this run.
- Components NOT run end-to-end.

## Components NOT run end-to-end

- **EXP-004b** — agent sweep — explicitly scoped out.
- **Other KBs** — bash only.
- **Other final top-k values** — only top-5.
- **Other stage-1 fusion strategies** — α=0.75 only (RRF excluded per
  EXP-004a's H3).
- **Other rerank models beyond Qwen3-0.6B + bge-v2-m3** — no
  bge-reranker-large, no Cohere/Voyage cloud rerank, no listwise LLM
  rerank.
- **Other embedders / chunk sizes** — queued for EXP-003c.
