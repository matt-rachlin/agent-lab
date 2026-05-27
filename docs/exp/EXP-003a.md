---
doc_id: exp-003a
title: 'EXP-003a: Bash KB retrieval-quality sweep'
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
# EXP-003a: Bash KB retrieval-quality sweep

Date created: 2026-05-26
Status: planned
Pre-registered: (commit SHA filled by `lab exp register` at registration time)

## Question

Within the lab's existing RAG infrastructure (the sealed bash KB, embedding
model `qwen3-embedding:8b-q8_0`, hybrid dense+BM25 retrieval), how do the
two cheap retrieval hyperparameters — hybrid blend `alpha` and top-`k` —
affect synthetic-query recall, MRR, and nDCG?

This is a *pure retrieval benchmark*. There is no agent loop, no LLM
generation step, no tool call. Each "cell" is `hybrid_query(kb_dir, q,
k=k, alpha=alpha)` against one of N synthetic queries; we measure whether
each query's originating chunk surfaces in the top-`k`.

EXP-003b runs the *agent* side (does giving a model `kb_query` change
its task success rate); EXP-003a tells us how good the retrieval is
*before* anything downstream uses it.

## Hypothesis

Three pre-registered hypotheses on the bash KB (sealed, 4,620 chunks,
4,096-dim embeddings, hybrid_query as wired in `lab.rag.index`).

- **H1 — Hybrid beats both endpoints on recall@5.** The best mean
  `recall@5` across alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0} is attained at
  some `alpha* ∈ [0.25, 0.75]`. (Equivalently: neither `alpha=0.0` —
  pure BM25 — nor `alpha=1.0` — pure dense — is the best alpha for
  recall@5.)

- **H2 — Top-k matters meaningfully.** Holding `alpha` at its best
  H1-confirmed value (or, if H1 is refuted, at `alpha=0.5`), going from
  `k=5` to `k=10` gives an absolute recall@k gain of **≥ 10 pp**.

- **H3 — BM25 is plausibly competitive on bash docs.** `alpha=0.0`
  (pure BM25) outperforms `alpha=1.0` (pure dense) on *at least one* of
  {recall@5, MRR@10} at the best (alpha=0.0, k) combination vs the best
  (alpha=1.0, k) combination. (Bash documentation is very symbol-heavy
  — `$@`, `${var:-x}`, `2>&1` — and exact-token matching plausibly
  beats vector similarity on those queries.)

These hypotheses are independent; each is judged on its own evidence.

## Why this matters

1. **It calibrates the bash-KB infrastructure** before EXP-003b
   (RAG-augmented agent) is allowed to draw conclusions about model
   behaviour. If recall@5 is ≤ 0.20, "the model didn't answer correctly
   from KB" is confounded by "the KB didn't return the right chunks."
2. **It informs the default `alpha` and `k`** the `kb_query` tool ships
   with. The current default is `alpha=0.5`, `k=5`; a pre-registered
   sweep gives us evidence-based defaults rather than the "structural
   compromise" defaults we landed on in 6h-a.
3. **It surfaces whether BM25 deserves to be the default** on
   symbol-heavy KBs (man pages, source code, JSON schemas) where lots of
   the answer signal lives in literal token overlap.

## Method

### KB (1, fixed)

| field | value |
|---|---|
| name | `bash` |
| status | `sealed` |
| chunks | 4,620 |
| embedding_model | `qwen3-embedding:8b-q8_0` (4,096-dim) |
| chunker | `structural-markdown v1` |
| index | LanceDB + per-chunk JSON-encoded sparse vectors |
| index_path | `~/db/kb/bash/index/` |

The KB is not rebuilt for this experiment. The "chunk-size variation"
and "embedding-model variation" axes are explicitly **out of scope** —
both would require a multi-hour rebuild and are queued for **EXP-003c**.
See "Components not run end-to-end" in F-006.

### Synthetic queries (N = 50, cached, deterministic)

The bash KB's `eval/queries.jsonl` is regenerated **once** at the start
of EXP-003a via `lab.rag.eval_retrieval.run_eval` with `n=50` and the
default `qwen3:14b-q4_K_M` query-generation model. The same 50-query
set is reused for every (alpha, k) cell so configurations are directly
comparable. Seed: deterministic (the sampler uses
`random.Random(0)`); the LLM-generated questions vary on the
generation model's temperature (0.3) but are written to disk and
re-read so the cells are reproducible.

If `run_eval` produces fewer than 50 valid queries (some get skipped
when the LLM returns an unparseable line), we accept the lower count
rather than retrying — N is reported in F-006.

### Matrix

- `alpha` ∈ {0.0, 0.25, 0.5, 0.75, 1.0} — 5 values
- `k` ∈ {1, 3, 5, 10} — 4 values
- queries: 50

= **5 × 4 = 20 cells**, each evaluating 50 queries → **1,000 query
evaluations total.**

Pure CPU + Ollama-embedding lookups; expected wall ~30-60 min depending
on how cleanly the embedding cache hits across cells. (`hybrid_query`
embeds the query once per cell — there's no query-level cache, so each
(alpha, k) cell re-embeds all 50 queries. Future optimization: cache
embeddings across cells. Out of scope for v0.1.)

### Metrics (per query, then aggregated)

For each `(alpha, k)` cell:

- `recall@k` — fraction of queries whose originating chunk is in the
  top-`k` retrieved.
- `mrr@10` — Mean Reciprocal Rank: each query's score is `1 / rank` if
  the originating chunk appears in the top-10, else 0; averaged across
  queries. (We use a fixed cap of 10 even when `k < 10` so MRR is
  comparable across `k` values — for `k < 10` we re-run the same
  `hybrid_query` with `k=10` internally for this metric only.)
- `ndcg@k` — single-relevant-document nDCG (since each query has
  exactly one gold chunk by construction): `1/log2(1 + rank)` if the
  chunk appears in the top-`k`, else 0.

The cell-level summary is `mean ± 95 % bootstrap CI` over the N
queries.

### Pre-flight pilot (REQUIRED, runs BEFORE the full sweep)

5 queries × 2 cells (alpha=0.5/k=5 and alpha=0.0/k=5). Sanity checks:

- analyzer wires up end-to-end (CSV + verdicts.md produced for a
  toy run)
- `hybrid_query` returns non-empty hit lists
- expected_chunk membership lookup works (query's gold chunk_id is in
  the LanceDB row payload)
- a single bad query (e.g. an empty string) does NOT crash the cell
  loop

Only after the pilot is green does the full 20-cell sweep run.

## Success / failure criteria

Each hypothesis is judged by the pre-registered rule, applied AFTER the
sweep + analyzer complete.

- **H1 confirmed** ⇔ `argmax_alpha mean(recall@5)` over `α ∈ {0.0, 0.25,
  0.5, 0.75, 1.0}` is in `{0.25, 0.5, 0.75}`. If two alphas tie within
  ≤ 0.005 of each other and one of them is an endpoint (0.0 or 1.0),
  we report the verdict as **MIXED** rather than confirmed.
- **H2 confirmed** ⇔ `mean(recall@10) − mean(recall@5) ≥ 0.10` at
  `alpha = alpha*` (the H1-confirmed alpha) or, if H1 is refuted, at
  `alpha = 0.5`.
- **H3 confirmed** ⇔ at the best `k` for each pure endpoint,
  `recall@5(alpha=0.0) > recall@5(alpha=1.0)` **OR**
  `MRR@10(alpha=0.0) > MRR@10(alpha=1.0)`.

Any failure modes (synthetic-query generation produces fewer than 25
valid queries; recall@5 stays below 0.10 across all configs;
embedding-service errors > 5 % of cells) are escalated in F-006 and the
analyzer marks the verdict UNDEFINED rather than CONFIRMED/REFUTED.

## Confounders to control

- **Single KB**: only the bash KB is in scope. Generalization to other
  KBs is out of scope.
- **Single embedding model**: only `qwen3-embedding:8b-q8_0`. Embedding
  ablation (e.g. `qwen3-embedding:4b`, `bge-large`, OpenAI ada-002) is
  queued for EXP-003c.
- **Synthetic queries only**: no human-written queries. The generated
  questions reflect the bash docs' phrasing more than typical user
  language. We accept this tradeoff for v0.1 (cost of human curation is
  high; queries are reproducible because they're cached to disk).
- **Single gold chunk per query**: each synthetic query has exactly
  one "originating" chunk. Real queries often have multiple relevant
  chunks; recall@5 is therefore conservative (any hit is good enough)
  while nDCG and MRR are exact.

## Kill criteria

- **Query-generation produces < 25 valid queries** at N=50 ask: STOP,
  raise the synthetic-question generation model's quality threshold,
  retry. Document in F-006.
- **Embedding service error rate > 5 % of cells**: STOP. The Ollama
  embedding endpoint is unreliable; switch to a smoke run before the
  full sweep.
- **`hybrid_query` returns empty hits on > 10 % of queries** at
  `alpha=0.5`: STOP. The KB index is broken (lance corruption / table
  schema drift); do not continue.

## Pre-mortem

It's 3 days from now and EXP-003a was a methodological failure. What
plausibly went wrong?

1. **LiteLLM/Ollama returns timeouts on the embedding model**
   mid-sweep. Mitigation: each query is embedded synchronously and
   exceptions are caught per query. Cells log per-query error count and
   continue.
2. **Synthetic-query generation produces leaky questions** — the LLM
   embeds the exact answer into the question text and BM25 trivially
   wins. Mitigation: examine the first 5 generated queries by eye
   during the pilot; document in F-006 caveats if leakage is visible.
3. **`hybrid_query`'s normalization changes behavior between
   `alpha=0.5` and `alpha=0.51`** in unexpected ways. Mitigation: the
   sweep uses 5 alpha values spaced at 0.25, which is well above any
   per-alpha numerical instability.

## Budget estimate

- 50 queries × 20 cells = 1,000 `hybrid_query` calls
- Each call: 1 embedding lookup (~150 ms on the qwen3-embedding model)
  + 1 lance dense search + 1 BM25 scan over 4,620 rows (~50 ms)
- Estimated wall: 1,000 × 200 ms ≈ **3-4 min compute** + embedding
  warm-up + dataset arrow scans = **30-60 min total wall**

## Outputs

- `analysis/EXP-003a/SUMMARY.md` — top-line verdicts + headline numbers
- `analysis/EXP-003a/verdicts.md` — per-hypothesis verdicts + supporting
  tables
- `analysis/EXP-003a/raw.csv` — per-cell per-query results
- `analysis/EXP-003a/best_configs.csv` — best (alpha, k) per metric

Findings rolled into the combined F-006 (with EXP-003b).
