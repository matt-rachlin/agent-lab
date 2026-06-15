---
doc_id: f-007-reranker-validation-refuted
title: 'F-007: Cross-encoder reranker REFUTED at +10pp threshold on bash KB — +5pp
  gain replicated at N=199 with paired Wilcoxon p=0.0038; +10pp infeasible on this
  KB'
zone: lab
kind: finding
status: active
owner: m
created: '2026-05-26'
last_updated: '2026-05-26'
last_verified: '2026-05-26'
depends_on:
- kind: doc
  target: exp-004a
- kind: doc
  target: exp-004c
- kind: code
  target: lab:scripts/retrieval_sweep.py
- kind: doc
  target: exp-004a-summary
- kind: doc
  target: exp-004a-verdicts
- kind: artifact
  target: lab:analysis/EXP-004a/raw.csv
- kind: artifact
  target: lab:analysis/EXP-004a/per_cell.csv
- kind: artifact
  target: lab:analysis/EXP-004a/rerank_stats.json
- kind: doc
  target: exp-004c-summary
- kind: doc
  target: exp-004c-verdicts
- kind: artifact
  target: lab:analysis/EXP-004c/raw.csv
- kind: artifact
  target: lab:analysis/EXP-004c/per_cell.csv
- kind: artifact
  target: lab:analysis/EXP-004c/rerank_stats.json
tags:
- lab
- finding
- findings
- confidence-high
- importance-7
---

# F-007: Cross-encoder reranker REFUTED at +10pp threshold on bash KB — +4pp gain not significant at N=50

## TL;DR

EXP-004a — synthetic-only reranker validation sweep, 4 cells (rerank
on/off × RRF/alpha-blend) × 50 EXP-003a queries — closes out the
Phase 7 acceptance gate. All three pre-registered hypotheses called
against the pre-reg rule:

- **H1 (aggressive, +10pp over C0=0.820): REFUTED.** Best reranked
  cell `max(C2, C3) = 0.840`; threshold 0.920. Observed lift over
  C0 (measured this round at 0.800): **+0.040 absolute**, far short
  of the +0.10 the plan pre-committed to.
- **H2 (rerank always improves, paired Wilcoxon both p<0.05):
  REFUTED.** C2 vs C0: +3 / -1 / ties=46, p=0.3125 (NS).
  C3 vs C1: +4 / -1 / ties=45, p=0.1875 (NS). At N=50, the rerank
  signal is too sparse to clear significance.
- **H3 (RRF beats alpha-blend as stage-1, informational):
  MIXED.** `delta_alpha = C1 - C0 = -0.020` (alpha-blend wins as
  baseline stage-1). `delta_rerank_arm = C3 - C2 = +0.000` (tied
  under rerank). RRF does not improve recall@5 here; the alpha=0.75
  default from EXP-003a remains the strongest stage-1.

**Decision (per Phase 7 acceptance gate)**: H1 is the kill criterion
the plan pre-committed to ("we revert the rerank default if so"). H1
is REFUTED. The rerank-by-default behaviour shipped in Phase 7 does
not earn its keep on this KB at this N.

## Setup

- **Plan**: [`docs/exp/EXP-004a.md`](../exp/EXP-004a.md) (pre-reg
  committed in same commit as runner extension per single-commit
  policy this round)
- **KB**: sealed `bash` KB, 4,620 chunks, qwen3-embedding:8b-q8_0
  (4,096-dim), LanceDB + per-row JSON sparse vectors. Same KB as
  EXP-003a / EXP-003b.
- **Reranker**: `Qwen/Qwen3-Reranker-0.6B`, served by
  `lab.rag.rerank_server` on `127.0.0.1:8401`. All 100 rerank calls
  in the final run went over HTTP to the host-side singleton.
  Lazy-load + idle TTL=300s.
- **Queries**: 50 synthetic queries reused verbatim from
  `analysis/EXP-003a/queries.jsonl` — no regeneration. The same
  queries that produced EXP-003a's published numbers.
- **Sweep config**:
  [`conf/sweep/EXP-004a.yaml`](../../conf/sweep/EXP-004a.yaml) — 4
  cells: C0 (α=0.75, no rerank, k=5), C1 (RRF, no rerank, k=5),
  C2 (α=0.75 + rerank, stage-1 top-50, final 5),
  C3 (RRF + rerank, stage-1 top-50, final 5).
- **Runner**: per-cell extension of
  [`scripts/retrieval_sweep.py`](../../scripts/retrieval_sweep.py).
  Pre-embeds queries once, unloads Ollama embedder (`keep_alive=0`)
  before kicking off rerank cells so the cross-encoder has the GPU,
  caches stage-1 results per (fusion, top_k_stage1) and reuses across
  cells that share a stage-1 config.

### Per-cell metrics (full sweep)

| cell | fusion | α | stage-1 top-k | rerank | final-k | recall@5 | MRR@10 | nDCG@10 | gold-in-pool | errors | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 baseline | alpha | 0.75 | 5  | no  | 5 | **0.800** | 0.677 | 0.708 | 0.80 | 0 | 0.0 |
| C1 RRF baseline | rrf | — | 5  | no  | 5 | 0.780 | 0.612 | 0.655 | 0.78 | 0 | 0.0 |
| C2 alpha + rerank | alpha | 0.75 | 50 | yes | 5 | **0.840** | 0.670 | 0.712 | 0.94 | 0 | 35.5 |
| C3 RRF + rerank | rrf | — | 50 | yes | 5 | **0.840** | 0.665 | 0.708 | 0.94 | 0 | 38.7 |

**Headline**: reranking lifts recall@5 by **+4pp** (0.800 → 0.840)
on both stage-1 arms, but the lift is **(a) below the pre-registered
+10pp threshold** and **(b) not paired-Wilcoxon significant** at
this N. The candidate-pool diagnostic (`gold-in-pool`) shows the
ceiling for rerank: with top-50 stage-1 the gold chunk is in the
candidate pool for 47/50 queries (0.940); the rerank cell hits 0.840
recall@5, i.e. it correctly surfaces the gold chunk for 42/47 of the
queries where it had a chance. The remaining 5/47 are cases where
the cross-encoder ranks something else above the gold.

### Rerank-service stats (final run)

- **Calls**: 100 (50 queries × 2 rerank cells)
- **Errors**: 0
- **Timeouts**: 0
- **p50 latency**: 735 ms
- **p95 latency**: 1313 ms
- **mean latency**: 742 ms
- **max latency**: 2478 ms (first-query cold-start: lazy weight load)

The p50 of ~735 ms for a 50-doc cross-encoder rerank on the 0.6B Qwen3
model fits comfortably inside the 30-second service timeout.

### Cumulative server metrics (across all 3 sweep attempts)

The first two runs hit `torch.OutOfMemoryError` on most rerank calls:

- **Attempt 1** (rerank service competing with Ollama embedder at 8.78 GiB
  resident): 92/100 calls failed with OOM. KB candidates are bash markdown
  sections that were embedded without truncation; the cross-encoder was
  trying to allocate up to 50 GiB for the attention matrix.
- **Attempt 2** (Ollama unloaded, no passage truncation): 39/100 calls
  failed — better, but the cross-encoder still OOM'd on the longest bash
  chunks (some sc####.md sections exceed 5k chars).
- **Attempt 3** (Ollama unloaded + passage truncation to 1,500 chars):
  **0/100 calls failed**. Final reported numbers above.

Server-side counters (cumulative across all 3 runs):
`requests_total=300`, `errors_total=131`, `timeouts=0`,
`duration_ms_sum=136,282`.

The truncation cap (1500 chars ≈ 400-500 tokens) was chosen
empirically to give the cross-encoder enough passage context to
score relevance while bounding the attention matrix. The EXP-003a
query-generation pass uses `chunk_text[:1500]` for the same reason.
We did not test whether shorter or longer caps materially change
the rerank verdict; queued for a follow-up.

## Per-hypothesis verdict

### H1 — best reranked cell ≥ 0.92 · REFUTED

Pre-reg: `max(mean(recall@5)(C2), mean(recall@5)(C3)) ≥ 0.92`.

- C0 (alpha=0.75, no rerank): **0.800** (this round; published
  EXP-003a value was 0.820 — single-query difference, see Caveats)
- C2 (alpha=0.75 + rerank): **0.840**
- C3 (RRF + rerank): **0.840**
- `max(C2, C3) = 0.840` vs threshold **0.920**

Delta over published EXP-003a C0=0.820: **+0.020**. Delta over
this-round C0=0.800: +0.040. Either way, well short of +0.100.
**REFUTED.**

### H2 — rerank always improves (paired Wilcoxon both p<0.05) · REFUTED

Pre-reg: paired one-sided Wilcoxon signed-rank, alternative
"rerank > baseline", both p<0.05.

| pair | +/-/ties | Wilcoxon p (one-sided) |
|---|---|---|
| C2 vs C0 | +3 / −1 / 46 | **0.3125** |
| C3 vs C1 | +4 / −1 / 45 | **0.1875** |

Neither comparison clears p<0.05. The directional signal is mildly
positive (3-4 wins per pair vs 1 loss), but at N=50 with this much
agreement-on-ties between baseline and rerank, the test has too
little power. **REFUTED.**

### H3 — RRF beats alpha-blend as stage-1 (informational) · MIXED

Pre-reg: informational only — no CONFIRMED/REFUTED label.

| comparison | observed delta |
|---|---|
| `delta_alpha = C1 - C0` (RRF vs alpha-blend, no rerank) | **−0.020** |
| `delta_rerank_arm = C3 - C2` (RRF vs alpha-blend, with rerank) | **+0.000** |

RRF does not improve recall@5 here. The Phase 7 default switch
from alpha-blend (α=0.75) to RRF (k=60) loses 2pp of stage-1 recall
on the bash KB. Under the rerank arm the two stage-1 strategies tie
exactly at recall@5=0.840 — the cross-encoder reranker washes out
the small stage-1 difference, which is the expected behaviour.

## Why this matters

1. **Phase 7's acceptance gate fires**. The plan committed to
   "EXP-003a re-run shows reranker beats alpha-only by the
   pre-registered margin, or we revert". The reranker delivers
   +0.040 (vs the pre-reg's +0.100 threshold) and the lift is not
   paired-significant at N=50. **The Phase 7 default of
   `LAB_RAG_RERANKER=qwen3-reranker-0.6b` + `rerank=True` on
   `hybrid_query` should be reverted to `rerank=False`** until
   evidence at higher N, on a richer KB, or with a stronger
   reranker model justifies the latency cost.
2. **Latency cost is real.** Rerank cells take 35-38s wall for 50
   queries (≈700-800ms per query, including the cold-start first
   call), vs ~0s for the baseline cells (stage-1 already cached).
   That's ~750ms of added latency per `kb_query` call at the
   current settings. Whether the +4pp recall@5 gain is worth that
   latency cost depends on the downstream agent's tolerance —
   EXP-003b found that for catastrophic local-without-kb cells
   (mean 0.000 → 0.500-0.800) the binding constraint is whether
   the KB is reachable at all, not the last 4pp of recall@5.
3. **The cross-encoder is not broken — the test is sharp.** The
   gold-in-pool fraction jumps from 0.78-0.80 (top-5 stage-1) to
   **0.94** (top-50 stage-1), so the rerank cells have a
   substantially richer candidate pool. The reranker correctly
   exploits this on the queries where the gold *was* missing from
   top-5 stage-1: +4pp recall@5. But that's the entire ceiling on
   this KB at this N — the alpha=0.75 stage-1 already captures the
   "easy" 80% of queries, and the cross-encoder doesn't recover
   most of the remaining 20%.
4. **RRF is not free.** The Phase 7 plan claimed "RRF beats
   alpha-blend zero-shot — rank-based, no score-normalization
   problem." On this KB, with the EXP-003a-tuned α=0.75, RRF loses
   2pp of stage-1 recall and ties under rerank. The α=0.75 default
   should stay as the documented stage-1 winner.

## Operational findings from the run

- **Reranker + Ollama embedder cannot co-reside in 12 GB VRAM.**
  The first sweep attempt failed because the embedding model was
  still resident (8.78 GiB) when the cross-encoder tried to load
  (needs ~2.5 GiB peak). The runner now POSTs `keep_alive=0` to the
  Ollama embedder after pre-embedding all queries and before
  kicking off any rerank cells. This is the standard
  VRAM-coexistence pattern from the Phase 7 plan.
- **Bash KB chunks are too long to feed to the cross-encoder
  whole.** Without per-passage truncation, the longest bash markdown
  sections (>5k chars) blow the cross-encoder's attention matrix on
  the 12 GB GPU. The runner truncates passage `text` to 1,500
  characters before sending to the rerank service. Open question:
  is this hurting the rerank's recall@5 signal? Plausible but not
  measured this round.
- **The host-side rerank service worked as designed in the final
  run**: 0 errors / 0 timeouts on 100 calls, p50 735ms,
  predictable lazy-load + idle-unload behaviour, VRAM peak under
  the 12 GiB budget.

## Caveats

- **N=50.** The H2 paired Wilcoxon would likely clear p<0.05 at
  N=200-500, but the test cost (synthetic-query generation +
  reranker latency) was budgeted at ~10 min wall. Increasing N is
  the cheapest next experiment.
- **C0 baseline = 0.800 here, vs published EXP-003a = 0.820.** The
  EXP-003a `hybrid_query_cached` and my per-cell `_stage1_only`
  share the math (alpha-blend over union of dense + BM25
  candidates), but the EXP-003a runner pre-computes hits at
  `max_k=10` and slices for k=5, while my per-cell runner pulls a
  pool sized `max(top_k_stage1 * 2, 40)`. For C0 (top_k_stage1=5)
  this is 40 vs EXP-003a's 40 — same. The 1-query difference
  (40/50 vs 41/50) is likely the order in which the sparse-pool
  ranking ties were broken; not significant. Reporting both numbers
  for honesty.
- **Truncation may understate the reranker.** Chunks truncated to
  1,500 chars may lose the most-relevant passage span. Empirical
  bound was set at run time after two OOM-driven retries; not
  optimised against the rerank signal. Queued for a follow-up.
- **Single rerank model.** `Qwen3-Reranker-0.6B` only.
  `bge-reranker-v2-m3` (the plan's fallback, BEIR 56.51, more
  mature) was not run.
- **Synthetic queries only.** Per user scoping ("synthetic only, no
  agent sweep"). EXP-004b — the planned agent-level
  with-rerank-vs-without sweep — was explicitly **not** run this
  round.
- **Single KB.** Bash only. Generalisation to other corpora is out
  of scope.
- **Single final top-k.** Only top-5 measured. Phase 7 default ships
  with `final_k=10`; not exercised here.

## Components NOT run end-to-end

- **EXP-004b** (agent-level rerank sweep) — explicitly scoped out
  by the user.
- **bge-reranker-v2-m3 fallback** — not exercised.
- **Other KBs** — bash only.
- **Other final top-k values** — only top-5.
- **Truncation-length ablation** — 1,500 chars chosen empirically;
  no measurement of whether shorter/longer caps move the rerank
  signal.
- **Higher N** — N=50 is what EXP-003a's cache contains; not
  expanded.

## Recommended next steps

1. **Revert the Phase 7 rerank default.** Per the plan's pre-reg
   acceptance gate, `hybrid_query(rerank=True)` should default to
   `False` until a higher-powered re-run clears H1. The host-side
   service can stay running; callers must opt into `rerank=True`
   explicitly.
2. **Run a higher-N follow-up (EXP-004c) if the +4pp signal is
   worth chasing.** N=200 with the same queries.jsonl extension
   would likely clear H2. H1's +10pp threshold is unlikely to clear
   on the bash KB regardless — the 0.80-0.84 plateau looks
   structural, not noise.
3. **Truncation-length ablation.** Cheap: rerun C2/C3 at 1,000 /
   1,500 / 3,000 / 6,000 chars to see whether the rerank gain is
   bottlenecked on passage truncation.
4. **Try bge-reranker-v2-m3 as the fallback.** Apache 2.0, mature,
   BEIR 56.51. Single-cell ablation against C2 should be ~5 min.
5. **EXP-004b (agent sweep) is no longer urgent.** Without H1 in
   pocket, there's no a priori reason to expect the cross-encoder
   to change EXP-003b's verdicts (locals depend on `kb_query`,
   cloud is near-ceiling). Defer.

## Reproduction

```bash
cd /data/lab/code
uv run lab exp register docs/exp/EXP-004a.md
uv run python scripts/retrieval_sweep.py conf/sweep/EXP-004a.yaml
# outputs: analysis/EXP-004a/{raw.csv, SUMMARY.md, verdicts.md,
#          per_cell.csv, rerank_stats.json}
```

## Files

- Pre-reg: [`docs/exp/EXP-004a.md`](../exp/EXP-004a.md)
- Config: [`conf/sweep/EXP-004a.yaml`](../../conf/sweep/EXP-004a.yaml)
- Runner: [`scripts/retrieval_sweep.py`](../../scripts/retrieval_sweep.py)
  (per-cell dispatch path)
- Results: `analysis/EXP-004a/`
- Rerank service:
  [`src/lab/rag/rerank_server.py`](../../src/lab/rag/rerank_server.py),
  unit `~/.config/systemd/user/rerank.service`

---

## Amendment 2026-05-26: EXP-004c replicates at N=199 with setup ablations

EXP-004c re-ran the validation at N=199 (50 EXP-003a queries + 149 newly
generated this run) with three setup ablations to rule out that the
EXP-004a result was driven by the test harness. Pre-reg
[`docs/exp/EXP-004c.md`](../exp/EXP-004c.md), config
[`conf/sweep/EXP-004c.yaml`](../../conf/sweep/EXP-004c.yaml), results
`analysis/EXP-004c/`.

### Per-cell metrics (full sweep, N=199)

| cell | mode | rerank model | trunc | recall@5 | MRR@10 | nDCG@10 | gold-in-pool | errors | wall (s) | rerank p50 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 (alpha=0.75) | — | — | — | **0.799** | 0.692 | 0.719 | 0.799 | 0 | 0.0 | — |
| Q1 (Qwen3 + 1500c, rpc) | rpc | Qwen3-Reranker-0.6B | 1500 | **0.849** | 0.691 | 0.731 | 0.900 | 0 | 131.1 | 686 |
| Q2 (Qwen3 + 2500c, rpc) | rpc | Qwen3-Reranker-0.6B | 2500 | **0.854** | 0.698 | 0.738 | 0.900 | 0 | 167.2 | 896 |
| Q3 (Qwen3 + no-trunc, rpc) | rpc | Qwen3-Reranker-0.6B | none | 0.543 | 0.458 | 0.479 | 0.900 | **77 OOMs** | 143.1 | 620 |
| Q4 (Qwen3 + 1500c, inproc) | inproc | Qwen3-Reranker-0.6B | 1500 | **0.849** | 0.691 | 0.731 | 0.900 | 0 | 134.8 | 686 |
| B1 (BGE + 1500c, inproc) | inproc | bge-reranker-v2-m3 | 1500 | **0.824** | 0.672 | 0.710 | 0.900 | 0 | 184.7 | 1041 |

### Per-hypothesis verdict (EXP-004c)

- **H1 (replication, ≥0.92, i.e. +10pp over B0)**: **REFUTED again.**
  `max(rerank cells) = 0.854`, delta over B0 = +0.055. Same direction
  as EXP-004a (+0.040 there, +0.055 here at higher N), still well short
  of the +10pp the plan pre-committed to. **The +10pp threshold is
  structurally infeasible on this KB at this stage-1 ceiling
  (gold-in-pool fraction = 0.90 at top-50; remaining 10% rerank cannot
  recover).** The rerank lift is real and now paired-Wilcoxon
  significant (Q1 vs B0 p=0.0038, Q2 vs B0 p=0.0023, Q4 vs B0 p=0.0038,
  one-sided), but it caps at +5-6pp, not +10pp.
- **H2 (truncation monotone Q3>Q2>Q1)**: **REFUTED.** Q3 (no-truncation)
  OOM'd on 77/199 queries — the bash KB has long markdown sections
  whose attention matrices exceed 12 GB VRAM. Q3's recall@5 collapses
  to 0.543 (39% misses by construction; remaining queries' recall is
  comparable to Q1/Q2). The monotone claim is unrunnable. Comparing
  the truncation-feasible pair: Q2 (2500c) = 0.854 vs Q1 (1500c) =
  0.849 — a +0.5pp gain from 67% more context. **Truncation is not
  the bottleneck**; 1500 chars captures the rerank signal.
- **H3 (RPC overhead, |Q4 - Q1| ≤ 0.02)**: **CONFIRMED, exactly.**
  Q4 (in-process LabReranker.rerank) = 0.849, Q1 (HTTP RPC to
  rerank_server) = 0.849. Per-cell wall: 134.8s vs 131.1s. **The
  host-side rerank service architecture has zero impact on the
  measured rerank signal or wall time.**
- **H4 (Q4 vs B1, Qwen3 vs BGE in-process)**: Qwen3 wins by 2.5pp
  (Q4 = 0.849 vs B1 = 0.824). Below the 5pp swap threshold;
  recommendation: **keep Qwen3-Reranker-0.6B** as the default. BGE is
  configured as the fallback per existing `FALLBACK_RERANKER_MODEL`.

### Statistical significance at N=199 vs N=50

At N=50 the paired Wilcoxon was underpowered:

| pair | EXP-004a (N=50) | EXP-004c (N=199) |
|---|---|---|
| Qwen3 1500c rpc vs baseline | p = 0.3125 | **p = 0.0038** |
| Qwen3 2500c rpc vs baseline | (not tested) | **p = 0.0023** |
| Qwen3 1500c inproc vs baseline | (not tested) | **p = 0.0038** |
| BGE 1500c inproc vs baseline | (not tested) | p = 0.1334 |

EXP-004a's H2 ("rerank always improves, paired Wilcoxon p<0.05")
**now clears for Qwen3 at N=199**. The +5pp lift is reliable. What
doesn't clear is the +10pp threshold — that's a real ceiling on this
KB, not a power problem.

### Operational findings from EXP-004c

- **`/unload` between RPC and in-process cells works**. The runner POSTs
  `/unload` to the rerank service before constructing the first
  in-process `LabReranker`. VRAM peak stayed under 11 GB throughout.
- **Q3's OOMs were per-query, not catastrophic.** The rerank service
  surfaced 77 HTTP 500s but kept serving; the runner counted each as
  a cell-level error. No mid-sweep crash. The kill criterion
  (>5% rerank errors overall) tripped — 77/995 = 7.7% — but that's
  entirely Q3; the four feasible rerank cells (Q1, Q2, Q4, B1)
  produced 0/796 = 0% errors.
- **BGE p50 latency = 1040 ms vs Qwen3's 686 ms** (1.5x slower) and
  Qwen3 outperforms by 2.5pp recall@5. There's no reason to swap.
- **In-process load adds ~3-15s cold-start** (Qwen3) / **~30s cold-start**
  (BGE first-call on cached weights), then per-query latency matches the
  RPC service almost exactly.

### Decision

**H1 stands as REFUTED at N=199.** The Phase 7 +10pp threshold is
structurally infeasible on the bash KB (stage-1 ceiling). The
reverted-default decision from the original F-007 stands:

- `LAB_RAG_RERANKER` env-var default should fall to `"none"` when
  unspecified (vs current fallthrough to `DEFAULT_RERANKER_MODEL`),
  since the +5pp lift doesn't earn the +700ms per-query latency at
  every call site by default.
- `kb_query` MCP tool default `rerank: bool = False` (was True). Callers
  who care opt in.
- `hybrid_query` default `rerank: bool = False` (was True). Same
  rationale.

The +5pp lift IS real and statistically significant. Callers with a
recall@5-critical workflow should opt in to `rerank=True`; the
infrastructure (host-side service, in-process fallback, BGE fallback,
truncation defaults) is validated and stable.

### Components NOT run end-to-end in EXP-004c (replicating EXP-004a scope)

- **EXP-004b** (agent-level rerank sweep) — still explicitly scoped out.
- **Other KBs** — bash only.
- **Other final top-k values** — only top-5.
- **Other stage-1 fusion strategies** — α=0.75 only; RRF excluded per
  EXP-004a's H3 finding (RRF -2pp on bash KB).
- **Other rerank models** — only Qwen3-Reranker-0.6B and
  bge-reranker-v2-m3. No bge-large, no Cohere/Voyage cloud rerank.
- **Other embedders / chunk sizes** — still queued for EXP-003c.

### Reproduction (EXP-004c)

```bash
cd /data/lab/code
uv run lab exp register docs/exp/EXP-004c.md
uv run python scripts/retrieval_sweep.py conf/sweep/EXP-004c.yaml
# outputs: analysis/EXP-004c/{raw.csv, SUMMARY.md, verdicts.md,
#          per_cell.csv, rerank_stats.json, queries.jsonl}
```

Estimated wall: ~110 min (75 min q-gen for 150 new queries + ~15 min
sweep). Cache the queries.jsonl to make re-runs cheap.

### Files (EXP-004c)

- Pre-reg: [`docs/exp/EXP-004c.md`](../exp/EXP-004c.md)
- Config: [`conf/sweep/EXP-004c.yaml`](../../conf/sweep/EXP-004c.yaml)
- Runner: [`scripts/retrieval_sweep.py`](../../scripts/retrieval_sweep.py)
  (EXP-004c dispatch detects per-cell `mode`/`truncation`/`rerank_model`)
- Results: `analysis/EXP-004c/`
- Query cache: `analysis/EXP-004c/queries.jsonl` (199 queries: 50
  reused from EXP-003a + 149 newly generated)
trust_level: verified

## Promotion history
- 2026-06-14: unverified -> verified (by Matt Rachlin)
