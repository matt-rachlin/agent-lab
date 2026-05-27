"""Retrieval-quality sweep runner.

Supports two config shapes:

* **EXP-003a shape (alpha x k matrix)**: top-level keys ``matrix.alpha``
  and ``matrix.k``. Runs the legacy 5x4 = 20 cell sweep.
* **EXP-004a shape (per-cell configs)**: top-level key ``cells:`` with
  per-cell `fusion / alpha / top_k_stage1 / rerank / final_k`. Used for
  rerank validation -- each cell can independently enable the host-side
  reranker (URL from the config's ``rerank.url``).

The script detects the shape by the presence of ``cells:`` and dispatches.
Both shapes share the cached synthetic-query path (the EXP-004a runner
reuses EXP-003a's ``queries.jsonl`` verbatim per pre-reg).
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import yaml
from ollama import Client
from tenacity import retry, stop_after_attempt, wait_exponential

# lab.rag imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lab.rag import DEFAULT_EMBED_MODEL
from lab.rag._util import console
from lab.rag.embedder import embed_texts, tokenize_for_bm25
from lab.rag.index import TABLE_NAME

REPO_ROOT = Path(__file__).resolve().parents[1]
KB_ROOT = Path.home() / "db" / "kb"

PILOT_QUERIES = 5
ALL_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
ALL_KS = [1, 3, 5, 10]


# ----------------------------------------------------------------------
# query generation (re-implemented locally so we can cache + log per query)
# ----------------------------------------------------------------------


@dataclass
class SyntheticQuery:
    question: str
    origin_chunk_id: str
    origin_doc_path: str
    origin_section: list[str]
    qvec: list[float] | None = None  # cached embedding


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6))
def _gen_question(client: Client, chunk_text: str, section: list[str], model: str) -> str:
    sec = " / ".join(section) if section else "(none)"
    resp = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write a realistic search query a user would type into a "
                    "knowledge-base search to find the given passage. Output ONLY "
                    "the question, no commentary."
                ),
            },
            {
                "role": "user",
                "content": (f"Section: {sec}\nPassage:\n---\n{chunk_text[:1500]}\n---\nQuestion:"),
            },
        ],
        options={"num_ctx": 4096, "temperature": 0.3},
    )
    text = (resp.get("message") or {}).get("content") or ""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line.lstrip("> ").strip().strip('"')


def _sample_chunks(rows: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    import random

    rnd = random.Random(seed)
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for r in rows:
        key = tuple((r.get("section_path") or [])[:2])
        buckets.setdefault(key, []).append(r)
    keys = list(buckets.keys())
    rnd.shuffle(keys)
    picked: list[dict[str, Any]] = []
    while len(picked) < n and any(buckets[k] for k in keys):
        for k in keys:
            if not buckets[k]:
                continue
            picked.append(buckets[k].pop())
            if len(picked) >= n:
                break
    return picked


def generate_or_load_queries(
    cache_path: Path,
    kb_dir: Path,
    n_target: int,
    question_model: str,
) -> list[SyntheticQuery]:
    if cache_path.exists():
        rows = [json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
        cached: list[SyntheticQuery] = [
            SyntheticQuery(
                question=r["question"],
                origin_chunk_id=r["origin_chunk_id"],
                origin_doc_path=r["origin_doc_path"],
                origin_section=list(r.get("origin_section") or []),
            )
            for r in rows
        ]
        console.print(f"[green]loaded[/] {len(cached)} cached queries from {cache_path}")
        return cached

    db = lancedb.connect(str(kb_dir / "index"))
    if TABLE_NAME not in db.list_tables().tables:
        raise RuntimeError(f"no index at {kb_dir}/index — build the KB first")
    rows_arrow = db.open_table(TABLE_NAME).to_arrow().to_pylist()
    if not rows_arrow:
        raise RuntimeError(f"empty index at {kb_dir}/index")
    console.print(f"[dim]bash KB rows: {len(rows_arrow)}")

    chosen = _sample_chunks(rows_arrow, n_target)
    client = Client(host="http://localhost:11434")
    queries: list[SyntheticQuery] = []
    skipped = 0
    for i, row in enumerate(chosen, 1):
        try:
            q = _gen_question(
                client, row["text"], list(row.get("section_path") or []), question_model
            )
        except Exception as e:
            console.print(f"[red]q-gen failed[/] {row['chunk_id']}: {e}")
            skipped += 1
            continue
        if not q or len(q) < 8:
            console.print(f"[yellow]q-gen empty[/] {row['chunk_id']}")
            skipped += 1
            continue
        queries.append(
            SyntheticQuery(
                question=q,
                origin_chunk_id=row["chunk_id"],
                origin_doc_path=row["doc_path"],
                origin_section=list(row.get("section_path") or []),
            )
        )
        console.print(f"[dim]q-gen {i}/{len(chosen)}: {q[:80]}[/]")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        for sq in queries:
            f.write(
                json.dumps(
                    {
                        "question": sq.question,
                        "origin_chunk_id": sq.origin_chunk_id,
                        "origin_doc_path": sq.origin_doc_path,
                        "origin_section": sq.origin_section,
                    }
                )
                + "\n"
            )
    console.print(f"[green]wrote[/] {len(queries)} queries to {cache_path} (skipped {skipped})")
    return queries


# ----------------------------------------------------------------------
# retrieval — re-implemented hybrid_query that takes a pre-embedded qvec
# ----------------------------------------------------------------------


@dataclass
class RankedHit:
    chunk_id: str
    score: float
    dense_score: float
    sparse_score: float


def hybrid_query_cached(
    tbl: Any,
    all_rows: list[dict[str, Any]],
    qvec: list[float],
    query_text: str,
    *,
    k: int,
    alpha: float,
) -> list[RankedHit]:
    """Pre-embedded hybrid retrieval — same math as lab.rag.index.hybrid_query
    but accepts a pre-computed qvec so embeddings are cached across cells.

    The `all_rows` is a pre-loaded snapshot of the table — sparse_json is
    on every row.
    """
    pool = max(k * 8, 40)

    # Dense candidates
    dense = tbl.search(qvec).limit(pool).to_list()

    if alpha < 1.0:
        # Sparse over all rows
        q_tokens = tokenize_for_bm25(query_text)
        sparse_scores: list[tuple[int, float]] = []
        if q_tokens:
            for idx, row in enumerate(all_rows):
                sj = row.get("sparse_json") or "{}"
                try:
                    sp = json.loads(sj)
                except Exception:
                    sp = {}
                s = sum(sp.get(tok, 0.0) for tok in q_tokens)
                if s > 0:
                    sparse_scores.append((idx, s))
        sparse_scores.sort(key=lambda x: x[1], reverse=True)
        sparse_top = sparse_scores[:pool]

        seen: dict[str, dict[str, Any]] = {}
        for r in dense:
            seen[r["chunk_id"]] = r
        sparse_score_by_id: dict[str, float] = {}
        for idx, s in sparse_top:
            r = all_rows[idx]
            seen.setdefault(r["chunk_id"], r)
            sparse_score_by_id[r["chunk_id"]] = s

        dense_distances = {r["chunk_id"]: float(r.get("_distance", 1.0)) for r in dense}
        d_sims = {cid: 1.0 / (1.0 + d) for cid, d in dense_distances.items()}
        max_dsim = max(d_sims.values()) if d_sims else 1.0
        d_sims = {cid: v / max_dsim for cid, v in d_sims.items()}

        max_sparse = max(sparse_score_by_id.values()) if sparse_score_by_id else 1.0
        s_norms = {cid: v / max_sparse for cid, v in sparse_score_by_id.items()}

        scored: list[RankedHit] = []
        for cid, _row in seen.items():
            d_score = d_sims.get(cid, 0.0)
            s_score = s_norms.get(cid, 0.0)
            combined = alpha * d_score + (1.0 - alpha) * s_score
            scored.append(
                RankedHit(chunk_id=cid, score=combined, dense_score=d_score, sparse_score=s_score)
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]
    else:
        # Pure dense
        max_d = max((1.0 / (1.0 + float(r.get("_distance", 1.0))) for r in dense), default=1.0)
        out: list[RankedHit] = []
        for r in dense[:k]:
            d_sim = (1.0 / (1.0 + float(r.get("_distance", 1.0)))) / max_d
            out.append(
                RankedHit(chunk_id=r["chunk_id"], score=d_sim, dense_score=d_sim, sparse_score=0.0)
            )
        return out


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------


def metric_recall_at(hits: list[RankedHit], gold: str, k: int) -> float:
    return 1.0 if gold in [h.chunk_id for h in hits[:k]] else 0.0


def metric_mrr(hits: list[RankedHit], gold: str, cap: int = 10) -> float:
    for i, h in enumerate(hits[:cap], 1):
        if h.chunk_id == gold:
            return 1.0 / i
    return 0.0


def metric_ndcg(hits: list[RankedHit], gold: str, k: int) -> float:
    for i, h in enumerate(hits[:k], 1):
        if h.chunk_id == gold:
            return 1.0 / math.log2(1 + i)
    return 0.0


# ----------------------------------------------------------------------
# sweep
# ----------------------------------------------------------------------


@dataclass
class CellMetrics:
    alpha: float
    k: int
    n_queries: int
    recall: float
    recall_ci_lo: float
    recall_ci_hi: float
    mrr10: float
    ndcg: float
    n_errors: int = 0


@dataclass
class PerQueryRow:
    alpha: float
    k: int
    query_idx: int
    question: str
    origin_chunk_id: str
    hit_rank: int  # 0 = miss; 1..k otherwise
    recall_at_1: int
    recall_at_3: int
    recall_at_5: int
    recall_at_10: int
    mrr10: float
    ndcg_at_k: float


def _bootstrap_ci(
    values: list[float], n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    import random

    rnd = random.Random(seed)
    if not values:
        return (0.0, 0.0)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rnd.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples)]
    return lo, hi


def run_cells(
    queries: list[SyntheticQuery],
    tbl: Any,
    all_rows: list[dict[str, Any]],
    alphas: list[float],
    ks: list[int],
) -> tuple[list[CellMetrics], list[PerQueryRow]]:
    """Run every (alpha, k) cell over the query set.

    Optimization: MRR@10 and recall@10 require k=10 hits, recall@1/3/5
    are subsets. So we run hybrid_query_cached once per (alpha, query)
    at k=max(10, k_max) and slice for each metric. This avoids redundant
    dense+sparse work per k cell.
    """
    max_k = max(*ks, 10)
    metrics: list[CellMetrics] = []
    per_query_rows: list[PerQueryRow] = []
    for alpha in alphas:
        # Run all queries once at this alpha (at the largest k)
        all_query_hits: list[list[RankedHit]] = []
        errors = 0
        for q in queries:
            try:
                assert q.qvec is not None, "embedding missing — call embed_queries first"
                hits = hybrid_query_cached(tbl, all_rows, q.qvec, q.question, k=max_k, alpha=alpha)
            except Exception as e:
                console.print(f"[red]retrieval failed alpha={alpha} q={q.question[:40]}: {e}[/]")
                hits = []
                errors += 1
            all_query_hits.append(hits)

        for k in ks:
            recalls: list[float] = []
            mrrs: list[float] = []
            ndcgs: list[float] = []
            for qi, (q, hits) in enumerate(zip(queries, all_query_hits, strict=False)):
                # Find rank of origin in the (already sorted by score, but
                # for the alpha=1.0 branch the hits list may be only k items,
                # so re-rank by re-doing the lookup is unnecessary — we
                # always store >= max_k).
                top_k_ids = [h.chunk_id for h in hits[:k]]
                r = 1.0 if q.origin_chunk_id in top_k_ids else 0.0
                recalls.append(r)
                m = metric_mrr(hits, q.origin_chunk_id, cap=10)
                mrrs.append(m)
                d = metric_ndcg(hits, q.origin_chunk_id, k)
                ndcgs.append(d)

                rank = 0
                for i, h in enumerate(hits[:max_k], 1):
                    if h.chunk_id == q.origin_chunk_id:
                        rank = i
                        break
                per_query_rows.append(
                    PerQueryRow(
                        alpha=alpha,
                        k=k,
                        query_idx=qi,
                        question=q.question,
                        origin_chunk_id=q.origin_chunk_id,
                        hit_rank=rank,
                        recall_at_1=int(rank == 1),
                        recall_at_3=int(1 <= rank <= 3) if rank > 0 else 0,
                        recall_at_5=int(1 <= rank <= 5) if rank > 0 else 0,
                        recall_at_10=int(1 <= rank <= 10) if rank > 0 else 0,
                        mrr10=m,
                        ndcg_at_k=d,
                    )
                )
            mean_recall = statistics.mean(recalls) if recalls else 0.0
            ci_lo, ci_hi = _bootstrap_ci(recalls)
            mean_mrr = statistics.mean(mrrs) if mrrs else 0.0
            mean_ndcg = statistics.mean(ndcgs) if ndcgs else 0.0
            metrics.append(
                CellMetrics(
                    alpha=alpha,
                    k=k,
                    n_queries=len(queries),
                    recall=mean_recall,
                    recall_ci_lo=ci_lo,
                    recall_ci_hi=ci_hi,
                    mrr10=mean_mrr,
                    ndcg=mean_ndcg,
                    n_errors=errors,
                )
            )
            console.print(
                f"[bold]alpha={alpha:.2f} k={k:2d}[/]  recall={mean_recall:.3f}  "
                f"mrr10={mean_mrr:.3f}  ndcg={mean_ndcg:.3f}  errors={errors}"
            )
    return metrics, per_query_rows


# ----------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------


def write_raw_csv(rows: list[PerQueryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "alpha",
                "k",
                "query_idx",
                "question",
                "origin_chunk_id",
                "hit_rank",
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "recall_at_10",
                "mrr10",
                "ndcg_at_k",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.alpha,
                    r.k,
                    r.query_idx,
                    r.question.replace("\n", " ")[:300],
                    r.origin_chunk_id,
                    r.hit_rank,
                    r.recall_at_1,
                    r.recall_at_3,
                    r.recall_at_5,
                    r.recall_at_10,
                    f"{r.mrr10:.6f}",
                    f"{r.ndcg_at_k:.6f}",
                ]
            )


def write_best_configs(metrics: list[CellMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by = {
        "recall@5": max((m for m in metrics if m.k == 5), key=lambda m: m.recall, default=None),
        "recall@10": max((m for m in metrics if m.k == 10), key=lambda m: m.recall, default=None),
        "mrr10": max(metrics, key=lambda m: m.mrr10),
        "ndcg": max(metrics, key=lambda m: m.ndcg),
    }
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "alpha", "k", "value", "n_queries"])
        for name, cm in by.items():
            if cm is None:
                continue
            val = cm.recall if "recall" in name else cm.mrr10 if "mrr" in name else cm.ndcg
            w.writerow([name, cm.alpha, cm.k, f"{val:.4f}", cm.n_queries])


def write_summary_md(metrics: list[CellMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# EXP-003a — bash KB retrieval-quality sweep — SUMMARY\n")
    lines.append(f"N queries: {metrics[0].n_queries if metrics else 0}\n")
    lines.append(f"Total cells: {len(metrics)}\n")
    lines.append("\n## recall@k by alpha (rows) × k (cols)\n")
    alphas = sorted({m.alpha for m in metrics})
    ks = sorted({m.k for m in metrics})
    lines.append("| alpha | " + " | ".join(f"k={k}" for k in ks) + " |")
    lines.append("|---|" + "|".join(["---"] * len(ks)) + "|")
    for a in alphas:
        row = [f"{a:.2f}"]
        for k in ks:
            cm = next((m for m in metrics if m.alpha == a and m.k == k), None)
            row.append(f"{cm.recall:.3f}" if cm else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n## MRR@10 by alpha × k\n")
    lines.append("| alpha | " + " | ".join(f"k={k}" for k in ks) + " |")
    lines.append("|---|" + "|".join(["---"] * len(ks)) + "|")
    for a in alphas:
        row = [f"{a:.2f}"]
        for k in ks:
            cm = next((m for m in metrics if m.alpha == a and m.k == k), None)
            row.append(f"{cm.mrr10:.3f}" if cm else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n## nDCG@k by alpha × k\n")
    lines.append("| alpha | " + " | ".join(f"k={k}" for k in ks) + " |")
    lines.append("|---|" + "|".join(["---"] * len(ks)) + "|")
    for a in alphas:
        row = [f"{a:.2f}"]
        for k in ks:
            cm = next((m for m in metrics if m.alpha == a and m.k == k), None)
            row.append(f"{cm.ndcg:.3f}" if cm else "-")
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n")


def compute_verdicts(metrics: list[CellMetrics]) -> dict[str, Any]:
    """Apply the three pre-registered decision rules."""
    by_alpha_k: dict[tuple[float, int], CellMetrics] = {(m.alpha, m.k): m for m in metrics}

    # H1: argmax_alpha mean(recall@5) ∈ {0.25, 0.5, 0.75}
    h1_alphas = sorted(by_alpha_k.keys())
    recall5_by_alpha = {a: by_alpha_k[(a, 5)].recall for (a, k) in h1_alphas if k == 5}
    best_alpha = max(recall5_by_alpha, key=lambda a: recall5_by_alpha[a])
    best_val = recall5_by_alpha[best_alpha]
    # tie-with-endpoint check
    tied_with_endpoint = False
    for a, v in recall5_by_alpha.items():
        if a in (0.0, 1.0) and abs(v - best_val) <= 0.005 and a != best_alpha:
            tied_with_endpoint = True
    h1_alpha_star = best_alpha if best_alpha in (0.25, 0.5, 0.75) else None
    h1_verdict = (
        "MIXED" if tied_with_endpoint else ("CONFIRMED" if h1_alpha_star is not None else "REFUTED")
    )

    # H2 alpha: H1-confirmed alpha else 0.5
    h2_alpha = h1_alpha_star if h1_alpha_star is not None else 0.5
    r5 = by_alpha_k[(h2_alpha, 5)].recall
    r10 = by_alpha_k[(h2_alpha, 10)].recall
    h2_delta = r10 - r5
    h2_verdict = "CONFIRMED" if h2_delta >= 0.10 else "REFUTED"

    # H3: at best k per endpoint, BM25 (alpha=0.0) beats dense (alpha=1.0)
    # on at least one of {recall@5, MRR@10}.
    bm25_recall5 = by_alpha_k[(0.0, 5)].recall
    dense_recall5 = by_alpha_k[(1.0, 5)].recall
    best_bm25_mrr = max(by_alpha_k[(0.0, k)].mrr10 for k in ALL_KS)
    best_dense_mrr = max(by_alpha_k[(1.0, k)].mrr10 for k in ALL_KS)
    h3_verdict = (
        "CONFIRMED"
        if (bm25_recall5 > dense_recall5) or (best_bm25_mrr > best_dense_mrr)
        else "REFUTED"
    )

    return {
        "H1": {
            "verdict": h1_verdict,
            "best_alpha": best_alpha,
            "best_recall5": best_val,
            "recall5_by_alpha": recall5_by_alpha,
        },
        "H2": {
            "verdict": h2_verdict,
            "alpha_used": h2_alpha,
            "recall5": r5,
            "recall10": r10,
            "delta": h2_delta,
        },
        "H3": {
            "verdict": h3_verdict,
            "bm25_recall5": bm25_recall5,
            "dense_recall5": dense_recall5,
            "best_bm25_mrr10": best_bm25_mrr,
            "best_dense_mrr10": best_dense_mrr,
        },
    }


def write_verdicts_md(verdicts: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# EXP-003a — verdicts\n")
    lines.append("Pre-registered rules in docs/exp/EXP-003a.md §Success/failure criteria.\n")

    h1 = verdicts["H1"]
    lines.append("\n## H1 — Hybrid beats both endpoints on recall@5\n")
    lines.append(f"**Verdict: {h1['verdict']}**\n")
    lines.append(f"- best alpha by mean(recall@5): {h1['best_alpha']}")
    lines.append(f"- mean(recall@5) at best alpha: {h1['best_recall5']:.3f}")
    lines.append("- recall@5 by alpha:")
    for a, v in sorted(h1["recall5_by_alpha"].items()):
        lines.append(f"  - alpha={a:.2f}: {v:.3f}")

    h2 = verdicts["H2"]
    lines.append("\n## H2 — Top-k matters meaningfully (recall@10 − recall@5 ≥ 0.10)\n")
    lines.append(f"**Verdict: {h2['verdict']}**\n")
    lines.append(f"- alpha used: {h2['alpha_used']}  (H1 alpha if confirmed, else 0.5)")
    lines.append(f"- recall@5: {h2['recall5']:.3f}")
    lines.append(f"- recall@10: {h2['recall10']:.3f}")
    lines.append(f"- delta: {h2['delta']:.3f}  (threshold 0.100)")

    h3 = verdicts["H3"]
    lines.append("\n## H3 — BM25 (alpha=0.0) plausibly competitive vs dense (alpha=1.0)\n")
    lines.append(f"**Verdict: {h3['verdict']}**\n")
    lines.append(f"- BM25 recall@5: {h3['bm25_recall5']:.3f}")
    lines.append(f"- Dense recall@5: {h3['dense_recall5']:.3f}")
    lines.append(f"- BM25 best MRR@10 (over k): {h3['best_bm25_mrr10']:.3f}")
    lines.append(f"- Dense best MRR@10 (over k): {h3['best_dense_mrr10']:.3f}")

    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# EXP-004a — per-cell rerank validation sweep
# ----------------------------------------------------------------------


def _stage1_only(
    tbl: Any,
    all_rows: list[dict[str, Any]],
    qvec: list[float],
    query_text: str,
    *,
    fusion: str,
    alpha: float | None,
    top_k_stage1: int,
) -> list[tuple[str, float, float, float, dict[str, Any]]]:
    """Stage-1 hybrid retrieval — fusion in {"rrf","alpha"}.

    Returns list of (chunk_id, combined_score, dense_score, sparse_score, row),
    truncated to ``top_k_stage1``, best-first.
    """
    import json as _json

    pool = max(top_k_stage1 * 2, 40)
    dense = tbl.search(qvec).limit(pool).to_list()
    if not dense:
        return []

    q_tokens = tokenize_for_bm25(query_text)
    sparse_scores: list[tuple[int, float]] = []
    if q_tokens:
        for idx, row in enumerate(all_rows):
            sj = row.get("sparse_json") or "{}"
            try:
                sp = _json.loads(sj)
            except Exception:
                sp = {}
            s = sum(sp.get(tok, 0.0) for tok in q_tokens)
            if s > 0:
                sparse_scores.append((idx, s))
    sparse_scores.sort(key=lambda x: x[1], reverse=True)
    sparse_top = sparse_scores[:pool]

    seen: dict[str, dict[str, Any]] = {}
    for r in dense:
        seen[r["chunk_id"]] = r
    sparse_score_by_id: dict[str, float] = {}
    for idx, s in sparse_top:
        r = all_rows[idx]
        seen.setdefault(r["chunk_id"], r)
        sparse_score_by_id[r["chunk_id"]] = s

    dense_distances = {r["chunk_id"]: float(r.get("_distance", 1.0)) for r in dense}
    d_sims_raw = {cid: 1.0 / (1.0 + d) for cid, d in dense_distances.items()}
    max_dsim = max(d_sims_raw.values()) if d_sims_raw else 1.0
    d_sims = {cid: v / max_dsim for cid, v in d_sims_raw.items()}
    max_sparse = max(sparse_score_by_id.values()) if sparse_score_by_id else 1.0
    s_norms = {cid: v / max_sparse for cid, v in sparse_score_by_id.items()}

    if fusion == "rrf":
        dense_ranking = [r["chunk_id"] for r in dense]
        sparse_ranking = [all_rows[idx]["chunk_id"] for idx, _ in sparse_top]
        # RRF constant k=60 (Cormack et al. 2009) — matches lab.rag default.
        k_const = 60
        fused: dict[str, float] = {}
        for rank, cid in enumerate(dense_ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k_const + rank)
        for rank, cid in enumerate(sparse_ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k_const + rank)
        scored: list[tuple[str, float, float, float, dict[str, Any]]] = []
        for cid, row in seen.items():
            scored.append(
                (cid, fused.get(cid, 0.0), d_sims.get(cid, 0.0), s_norms.get(cid, 0.0), row)
            )
    else:
        a = 0.5 if alpha is None else float(alpha)
        scored = []
        for cid, row in seen.items():
            d_score = d_sims.get(cid, 0.0)
            s_score = s_norms.get(cid, 0.0)
            combined = a * d_score + (1.0 - a) * s_score
            scored.append((cid, combined, d_score, s_score, row))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k_stage1]


def _rerank_via_http(
    *,
    url: str,
    model: str | None,
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int,
    timeout: float = 30.0,
) -> tuple[list[dict[str, Any]], float, str | None]:
    """POST to the rerank service. Returns (hits, latency_ms, err_or_None)."""
    import httpx as _httpx

    if not candidates or top_n <= 0:
        return [], 0.0, None
    base = url.rstrip("/")
    payload: dict[str, Any] = {
        "query": query,
        "candidates": candidates,
        "top_n": top_n,
    }
    if model:
        payload["model"] = model
    t0 = time.perf_counter()
    try:
        with _httpx.Client(timeout=timeout) as cli:
            resp = cli.post(f"{base}/rerank", json=payload)
    except Exception as exc:
        return [], (time.perf_counter() - t0) * 1000.0, f"transport: {exc}"
    lat_ms = (time.perf_counter() - t0) * 1000.0
    if resp.status_code >= 400:
        return [], lat_ms, f"http {resp.status_code}: {resp.text[:120]}"
    try:
        body = resp.json()
    except Exception as exc:
        return [], lat_ms, f"json: {exc}"
    hits = body.get("hits")
    if not isinstance(hits, list):
        return [], lat_ms, f"missing hits: {body!r}"
    return hits, lat_ms, None


def _wilcoxon_one_sided_greater(deltas: list[float]) -> float:
    """Paired Wilcoxon signed-rank, one-sided alternative "treat > control".

    deltas = treat[i] - control[i]. Returns p-value for H1: median(delta) > 0.
    Implemented via scipy if available; falls back to exact small-sample
    sign-test if not (recall@5 deltas are -1/0/+1 over 50 paired observations).
    """
    try:
        from scipy.stats import wilcoxon  # type: ignore

        nonzero = [d for d in deltas if d != 0.0]
        if not nonzero:
            return 1.0  # no signal at all -> cannot reject
        res = wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
        return float(res.pvalue)
    except Exception:
        # Fallback: sign test on non-zero pairs (binomial under H0=0.5).
        nonzero = [d for d in deltas if d != 0.0]
        if not nonzero:
            return 1.0
        n = len(nonzero)
        k = sum(1 for d in nonzero if d > 0)
        # P(X >= k | n, p=0.5) one-sided.
        from math import comb

        p = sum(comb(n, i) for i in range(k, n + 1)) / (2.0**n)
        return p


def _extend_query_cache(
    *,
    cache_path: Path,
    seed_cache_path: Path,
    kb_dir: Path,
    n_target: int,
    question_model: str,
) -> list[SyntheticQuery]:
    """EXP-004c — seed a query cache from another cache, then generate up to n_target.

    If `cache_path` already has `>= n_target` queries, returns them verbatim.
    Otherwise: copies the seed cache, then generates additional questions
    against the same KB using :func:`_gen_question`, deduping by
    ``origin_chunk_id`` so seeded chunks aren't re-used. Writes the merged
    cache to `cache_path`.
    """
    # Existing extended cache?
    if cache_path.exists():
        rows = [json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
        if len(rows) >= n_target:
            console.print(
                f"[green]loaded[/] {len(rows)} cached queries from {cache_path} (n_target={n_target})"
            )
            return [
                SyntheticQuery(
                    question=r["question"],
                    origin_chunk_id=r["origin_chunk_id"],
                    origin_doc_path=r["origin_doc_path"],
                    origin_section=list(r.get("origin_section") or []),
                )
                for r in rows
            ]
        console.print(
            f"[yellow]partial cache[/] {cache_path} has {len(rows)} < {n_target}; extending"
        )
        existing = rows
    else:
        # Seed from EXP-003a cache.
        if not seed_cache_path.exists():
            raise RuntimeError(f"seed cache missing: {seed_cache_path}")
        existing = [
            json.loads(line) for line in seed_cache_path.read_text().splitlines() if line.strip()
        ]
        console.print(f"[dim]seeded[/] {len(existing)} queries from {seed_cache_path}")

    used_ids = {r["origin_chunk_id"] for r in existing}

    # Pull KB rows, exclude already-used origin chunks, sample stratified.
    db = lancedb.connect(str(kb_dir / "index"))
    if TABLE_NAME not in db.list_tables().tables:
        raise RuntimeError(f"no index at {kb_dir}/index — build the KB first")
    rows_arrow = db.open_table(TABLE_NAME).to_arrow().to_pylist()
    avail = [r for r in rows_arrow if r["chunk_id"] not in used_ids]
    n_needed = n_target - len(existing)
    if n_needed <= 0:
        # Trim back to n_target.
        return [
            SyntheticQuery(
                question=r["question"],
                origin_chunk_id=r["origin_chunk_id"],
                origin_doc_path=r["origin_doc_path"],
                origin_section=list(r.get("origin_section") or []),
            )
            for r in existing[:n_target]
        ]
    # Use a different seed than EXP-003a's (which was 0) so we don't re-sample
    # the same chunks even after the dedupe; EXP-003a's _sample_chunks used
    # seed=0.
    chosen = _sample_chunks(avail, n_needed, seed=42)
    console.print(
        f"[bold yellow]q-gen[/] generating {len(chosen)} new queries "
        f"(have {len(existing)}, target {n_target}, model={question_model})"
    )

    # Pre-flight: ensure Ollama embedder AND the rerank service's model are
    # unloaded so the question-gen model (qwen3:14b-q4_K_M, ~9 GB) has VRAM.
    try:
        import httpx as _httpx

        with _httpx.Client(timeout=10.0) as cli:
            cli.post(
                "http://localhost:11434/api/generate",
                json={"model": DEFAULT_EMBED_MODEL, "prompt": "", "keep_alive": 0},
            )
    except Exception:
        pass
    rerank_url_env = os.environ.get("LAB_RAG_RERANKER_URL", "http://127.0.0.1:8401")
    try:
        import httpx as _httpx

        with _httpx.Client(timeout=15.0) as cli:
            cli.post(f"{rerank_url_env.rstrip('/')}/unload")
        time.sleep(1.0)
    except Exception:
        pass

    client = Client(host="http://localhost:11434")
    new_rows: list[dict[str, Any]] = []
    skipped = 0
    t0 = time.time()
    for i, row in enumerate(chosen, 1):
        try:
            q = _gen_question(
                client, row["text"], list(row.get("section_path") or []), question_model
            )
        except Exception as e:
            console.print(f"[red]q-gen failed[/] {row['chunk_id']}: {e}")
            skipped += 1
            continue
        if not q or len(q) < 8:
            console.print(f"[yellow]q-gen empty[/] {row['chunk_id']}")
            skipped += 1
            continue
        new_rows.append(
            {
                "question": q,
                "origin_chunk_id": row["chunk_id"],
                "origin_doc_path": row["doc_path"],
                "origin_section": list(row.get("section_path") or []),
            }
        )
        if i % 10 == 0 or i == len(chosen):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1.0)
            eta = (len(chosen) - i) / max(rate, 0.001)
            console.print(
                f"[dim]q-gen {i}/{len(chosen)} (skipped={skipped}, rate={rate:.2f}/s, eta={eta/60:.1f}m): {q[:80]}[/]"
            )

    merged = existing + new_rows
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    console.print(
        f"[green]wrote[/] {len(merged)} queries to {cache_path} "
        f"(new={len(new_rows)}, skipped={skipped})"
    )
    return [
        SyntheticQuery(
            question=r["question"],
            origin_chunk_id=r["origin_chunk_id"],
            origin_doc_path=r["origin_doc_path"],
            origin_section=list(r.get("origin_section") or []),
        )
        for r in merged
    ]


def run_per_cell_sweep(cfg: dict[str, Any]) -> int:
    """EXP-004a / EXP-004c — run a list of (cell) configurations, with optional rerank.

    EXP-004c extension: cells may carry `rerank_model`, `truncation`
    (int chars or null=no-trunc), and `mode` (rpc | inproc). The runner
    dispatches per-cell to the host-side service (rpc) or constructs a
    :class:`LabReranker` directly (inproc) — exercising the no-RPC path.
    """

    kb_name = cfg["kb"]["name"]
    kb_dir = KB_ROOT / kb_name
    cache_path = REPO_ROOT / cfg["queries"]["cache_path"]
    cells_cfg = cfg["cells"]
    rerank_cfg = cfg.get("rerank") or {}
    rerank_url = rerank_cfg.get("url") or os.environ.get("LAB_RAG_RERANKER_URL")
    rerank_model = rerank_cfg.get("model")
    slug = cfg.get("experiment", {}).get("slug", "EXP-004?")

    # Detect EXP-004c shape by presence of new per-cell fields.
    is_exp004c = any(
        ("mode" in c) or ("truncation" in c) or ("rerank_model" in c) for c in cells_cfg
    )

    out_summary = REPO_ROOT / cfg["outputs"]["summary"]
    out_verdicts = REPO_ROOT / cfg["outputs"]["verdicts"]
    out_raw = REPO_ROOT / cfg["outputs"]["raw"]
    out_per_cell = REPO_ROOT / cfg["outputs"]["per_cell"]
    out_rerank_stats = REPO_ROOT / cfg["outputs"]["rerank_stats"]

    console.print(f"[bold]{slug}[/] per-cell sweep on KB={kb_name} ({kb_dir})")

    # Pre-flight: any RPC rerank cell requires the service to be reachable.
    # Inproc rerank cells don't — they construct LabReranker directly.
    rpc_cells = [c for c in cells_cfg if c.get("rerank") and c.get("mode", "rpc") == "rpc"]
    inproc_cells = [c for c in cells_cfg if c.get("rerank") and c.get("mode") == "inproc"]
    any_rerank = bool(rpc_cells) or bool(inproc_cells)
    any_rpc = bool(rpc_cells)
    if any_rpc:
        if not rerank_url:
            console.print("[red]KILL[/] rpc rerank cells configured but no rerank.url set")
            return 2
        import httpx as _httpx

        try:
            with _httpx.Client(timeout=5.0) as cli:
                hresp = cli.get(f"{rerank_url.rstrip('/')}/healthz")
            if hresp.status_code != 200:
                console.print(
                    f"[red]KILL[/] rerank service health check returned {hresp.status_code}"
                )
                return 2
            health = hresp.json()
            server_model = health.get("model")
            console.print(
                f"[green]rerank service[/] healthy: model={server_model} loaded={health.get('loaded')}"
            )
            # Check per-cell rerank_model compatibility with the running server.
            for c in rpc_cells:
                cell_model = c.get("rerank_model") or rerank_model
                if cell_model and cell_model != server_model:
                    console.print(
                        f"[red]KILL[/] cell {c['name']} requests rpc rerank with model "
                        f"{cell_model} but server runs {server_model}; "
                        f"swap cell mode to inproc or restart rerank service with that model"
                    )
                    return 2
        except Exception as exc:
            console.print(f"[red]KILL[/] rerank service unreachable at {rerank_url}: {exc}")
            return 2

    # Load queries.
    # EXP-004c extension: if `seed_cache_path` + `n_target` are set, seed +
    # generate up to n_target. EXP-004a behaviour preserved otherwise.
    queries_cfg = cfg["queries"]
    seed_cache_raw = queries_cfg.get("seed_cache_path")
    n_target_raw = queries_cfg.get("n_target")
    question_model_raw = queries_cfg.get("question_model")
    if seed_cache_raw and n_target_raw:
        seed_cache_path = REPO_ROOT / seed_cache_raw
        queries = _extend_query_cache(
            cache_path=cache_path,
            seed_cache_path=seed_cache_path,
            kb_dir=kb_dir,
            n_target=int(n_target_raw),
            question_model=str(question_model_raw),
        )
    else:
        if not cache_path.exists():
            console.print(f"[red]KILL[/] query cache missing at {cache_path}")
            return 2
        rows = [json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
        queries = [
            SyntheticQuery(
                question=r["question"],
                origin_chunk_id=r["origin_chunk_id"],
                origin_doc_path=r["origin_doc_path"],
                origin_section=list(r.get("origin_section") or []),
            )
            for r in rows
        ]
    # EXP-004c min-N kill criterion is 100; EXP-004a's was 25.
    min_n = 100 if is_exp004c else 25
    if len(queries) < min_n:
        console.print(f"[red]KILL[/] only {len(queries)} queries loaded (< {min_n})")
        return 2
    console.print(f"[green]queries[/] {len(queries)} loaded from {cache_path}")

    # KB
    db = lancedb.connect(str(kb_dir / "index"))
    tbl = db.open_table(TABLE_NAME)
    all_rows = tbl.to_arrow().to_pylist()
    console.print(f"[dim]KB table: {len(all_rows)} rows")

    # Embed queries once.
    t_emb_start = time.time()
    texts = [q.question for q in queries]
    res = embed_texts(texts, model=DEFAULT_EMBED_MODEL, batch_size=8)
    if len(res.vectors) != len(queries):
        console.print(f"[red]KILL[/] embedding count mismatch {len(res.vectors)} vs {len(queries)}")
        return 2
    for q, v in zip(queries, res.vectors, strict=True):
        q.qvec = v
    console.print(f"[green]embeddings[/] {len(queries)} done ({time.time() - t_emb_start:.1f}s)")

    # If any cell uses the host-side reranker, the embedder + reranker fight
    # for the 12 GB VRAM. The embedding model holds ~8.8 GB; the reranker
    # needs ~2.5 GB at load. We have all qvecs cached in memory, so unload
    # the Ollama embedding model NOW (via keep_alive=0). This is the standard
    # VRAM-coexistence pattern from the Phase 7 plan.
    if any_rerank:
        try:
            import httpx as _httpx

            with _httpx.Client(timeout=10.0) as cli:
                # POST /api/generate with keep_alive=0 unloads cleanly.
                cli.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": DEFAULT_EMBED_MODEL,
                        "prompt": "",
                        "keep_alive": 0,
                    },
                )
            time.sleep(2.0)
            with _httpx.Client(timeout=5.0) as cli:
                ps = cli.get("http://localhost:11434/api/ps").json()
            resident = [m.get("model") for m in ps.get("models", [])]
            console.print(f"[dim]ollama resident after unload: {resident}")
        except Exception as exc:
            console.print(f"[yellow]warn[/] could not unload embedder: {exc}")

    # Pre-compute stage-1 results per (fusion, top_k_stage1) per query to
    # amortise across rerank-on/rerank-off pairs that share stage-1.
    stage1_keys: set[tuple[str, float | None, int]] = set()
    for c in cells_cfg:
        stage1_keys.add(
            (
                c["fusion"],
                float(c.get("alpha")) if c.get("alpha") is not None else None,
                int(c["top_k_stage1"]),
            )
        )

    stage1_cache: dict[
        tuple[str, float | None, int], list[list[tuple[str, float, float, float, dict[str, Any]]]]
    ] = {}
    for key in stage1_keys:
        fusion, alpha, top_k = key
        out_list: list[list[tuple[str, float, float, float, dict[str, Any]]]] = []
        t_s1 = time.time()
        for q in queries:
            assert q.qvec is not None
            try:
                s1 = _stage1_only(
                    tbl,
                    all_rows,
                    q.qvec,
                    q.question,
                    fusion=fusion,
                    alpha=alpha,
                    top_k_stage1=top_k,
                )
            except Exception as exc:
                console.print(
                    f"[red]stage-1 err[/] fusion={fusion} alpha={alpha} k={top_k} q={q.question[:40]}: {exc}"
                )
                s1 = []
            out_list.append(s1)
        stage1_cache[key] = out_list
        console.print(
            f"[dim]stage-1 cached[/] fusion={fusion} alpha={alpha} top_k={top_k} "
            f"({time.time() - t_s1:.1f}s, {len(out_list)} queries)"
        )

    # Per-cell run.
    rerank_call_count = 0
    rerank_err_count = 0
    rerank_latencies: list[float] = []
    per_cell_latencies: dict[str, list[float]] = {}
    per_cell_errors_count: dict[str, int] = {}

    per_cell_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    cell_recalls_for_h2: dict[str, list[int]] = {}  # cell_name -> per-query 0/1

    # Cache of in-process LabReranker instances by model_name, so cells that
    # share a model don't re-load weights. Lazy-construct only when needed.
    inproc_rerankers: dict[str, Any] = {}

    def _get_inproc(model_name: str) -> Any:
        if model_name in inproc_rerankers:
            return inproc_rerankers[model_name]
        from lab.rag.rerank import LabReranker

        console.print(
            f"[bold yellow]inproc rerank load[/] {model_name} "
            f"(may take 30-120s on first call, cold download for non-cached models)"
        )
        # Clear LAB_RAG_RERANKER_URL in process env so LabReranker.rerank()
        # doesn't try to dispatch over HTTP — we want the in-process predict
        # path exercised.
        os.environ.pop("LAB_RAG_RERANKER_URL", None)
        r = LabReranker(model_name=model_name, idle_unload_sec=0)
        inproc_rerankers[model_name] = r
        return r

    # Track when we transition from rpc cells to inproc cells, to free the
    # rerank service's VRAM before constructing in-process rerankers.
    rpc_cell_names = {c["name"] for c in rpc_cells}
    inproc_cell_names = {c["name"] for c in inproc_cells}
    last_rpc_idx = -1
    for ci, c in enumerate(cells_cfg):
        if c["name"] in rpc_cell_names:
            last_rpc_idx = ci

    for cell_idx, cell in enumerate(cells_cfg):
        name = cell["name"]
        fusion = cell["fusion"]
        alpha_v = float(cell.get("alpha")) if cell.get("alpha") is not None else None
        top_k_stage1 = int(cell["top_k_stage1"])
        do_rerank = bool(cell.get("rerank"))
        final_k = int(cell["final_k"])
        cell_mode = cell.get("mode", "rpc") if do_rerank else None
        cell_rerank_model = cell.get("rerank_model") or rerank_model

        # Before the first inproc cell, ask the rerank service to unload its
        # weights so the in-process reranker has VRAM headroom.
        if (
            do_rerank
            and cell_mode == "inproc"
            and name in inproc_cell_names
            and last_rpc_idx >= 0
            and cell_idx > last_rpc_idx
            and rerank_url
        ):
            already_unloaded = any(pc.get("cell") in inproc_cell_names for pc in per_cell_rows)
            if not already_unloaded:
                try:
                    import httpx as _httpx

                    with _httpx.Client(timeout=15.0) as cli:
                        u = cli.post(f"{rerank_url.rstrip('/')}/unload")
                    console.print(
                        f"[dim]rerank service /unload[/] -> {u.status_code} {u.text[:80]}"
                    )
                    time.sleep(2.0)
                except Exception as exc:
                    console.print(f"[yellow]warn[/] rerank /unload failed: {exc}")
        # Truncation handling:
        #   - explicit key "truncation": null  -> no truncation (full passage)
        #   - integer N                         -> text[:N]
        #   - missing key                       -> default 1500 (EXP-004a behaviour)
        if "truncation" in cell:
            cell_trunc = cell["truncation"]
            cell_trunc_int = int(cell_trunc) if cell_trunc is not None else None
        else:
            cell_trunc_int = 1500

        key = (fusion, alpha_v, top_k_stage1)
        stage1_all = stage1_cache[key]

        t_cell = time.time()
        recall_5: list[int] = []
        mrr_vals: list[float] = []
        ndcg_vals: list[float] = []
        cell_errors = 0
        cell_latencies: list[float] = []
        gold_in_pool_count = 0

        console.print(
            f"[bold]cell {name}[/] mode={cell_mode} model={cell_rerank_model} "
            f"truncation={cell_trunc_int if cell_trunc_int is not None else 'none'} "
            f"(rerank={do_rerank})"
        )

        # Pre-warm inproc reranker if this cell uses one — loads weights and
        # primes GPU before per-query timing starts.
        if do_rerank and cell_mode == "inproc":
            try:
                _get_inproc(cell_rerank_model)
            except Exception as exc:
                console.print(f"[red]cell {name} inproc-load failed[/] {exc}")
                cell_errors = len(queries)
                # Fall through to record empty cell rows.

        for qi, (q, stage1) in enumerate(zip(queries, stage1_all, strict=True)):
            gold = q.origin_chunk_id
            gold_in_pool = any(c[0] == gold for c in stage1)
            if gold_in_pool:
                gold_in_pool_count += 1

            if do_rerank and stage1:
                # Truncate passage text (or not) per cell config.
                if cell_trunc_int is None:
                    candidates = [
                        {
                            "chunk_id": c[0],
                            "combined": c[1],
                            "dense_score": c[2],
                            "sparse_score": c[3],
                            "text": (c[4].get("text", "") or ""),
                            "stage1_rank": idx + 1,
                        }
                        for idx, c in enumerate(stage1)
                    ]
                else:
                    candidates = [
                        {
                            "chunk_id": c[0],
                            "combined": c[1],
                            "dense_score": c[2],
                            "sparse_score": c[3],
                            "text": (c[4].get("text", "") or "")[:cell_trunc_int],
                            "stage1_rank": idx + 1,
                        }
                        for idx, c in enumerate(stage1)
                    ]

                err: str | None = None
                lat_ms = 0.0
                final_hits: list[dict[str, Any]] = []
                final_ids: list[str | None] = []
                if cell_mode == "inproc":
                    try:
                        r_inst = _get_inproc(cell_rerank_model)
                        t0_l = time.perf_counter()
                        hits_local = r_inst.rerank(q.question, list(candidates), top_n=final_k)
                        lat_ms = (time.perf_counter() - t0_l) * 1000.0
                        final_hits = hits_local
                        final_ids = [h.get("chunk_id") for h in hits_local]
                    except Exception as exc:
                        err = f"inproc: {exc}"
                else:
                    # rpc
                    hits, lat_ms, err = _rerank_via_http(
                        url=rerank_url,
                        model=cell_rerank_model,
                        query=q.question,
                        candidates=candidates,
                        top_n=final_k,
                    )
                    if not err:
                        final_hits = hits
                        final_ids = [h.get("chunk_id") for h in hits]
                rerank_call_count += 1
                rerank_latencies.append(lat_ms)
                cell_latencies.append(lat_ms)
                if err:
                    rerank_err_count += 1
                    cell_errors += 1
                    final_ids = []
                    final_hits = []
                    if qi < 5 or qi % 50 == 0:
                        console.print(f"[red]rerank err[/] cell={name} qi={qi}: {err[:200]}")
            else:
                final_hits = []
                final_ids = [c[0] for c in stage1[:final_k]]

            # Compute rank of gold in final list (1-based; 0 = miss).
            rank = 0
            for i, cid in enumerate(final_ids, start=1):
                if cid == gold:
                    rank = i
                    break

            r5 = 1 if (rank > 0 and rank <= 5) else 0
            recall_5.append(r5)
            # MRR/nDCG capped at 10 conventionally; our final_k=5 so structurally <=5.
            m = 1.0 / rank if rank > 0 else 0.0
            mrr_vals.append(m)
            d = (1.0 / math.log2(1 + rank)) if rank > 0 else 0.0
            ndcg_vals.append(d)

            per_query_rows.append(
                {
                    "cell": name,
                    "fusion": fusion,
                    "alpha": alpha_v if alpha_v is not None else "",
                    "top_k_stage1": top_k_stage1,
                    "rerank": int(do_rerank),
                    "rerank_model": cell_rerank_model or "",
                    "truncation": (cell_trunc_int if cell_trunc_int is not None else 0),
                    "mode": cell_mode or "",
                    "final_k": final_k,
                    "query_idx": qi,
                    "question": q.question.replace("\n", " ")[:300],
                    "origin_chunk_id": gold,
                    "gold_in_stage1_pool": int(gold_in_pool),
                    "stage1_size": len(stage1),
                    "rerank_top_score": (
                        float(final_hits[0]["rerank_score"])
                        if final_hits and "rerank_score" in final_hits[0]
                        else ""
                    ),
                    "rerank_bottom_score": (
                        float(final_hits[-1]["rerank_score"])
                        if final_hits and "rerank_score" in final_hits[-1]
                        else ""
                    ),
                    "rerank_latency_ms": f"{lat_ms:.1f}" if do_rerank and stage1 else "",
                    "hit_rank": rank,
                    "recall_at_5": r5,
                    "mrr10": f"{m:.6f}",
                    "ndcg_at_10": f"{d:.6f}",
                }
            )

        cell_recalls_for_h2[name] = recall_5
        per_cell_latencies[name] = cell_latencies
        per_cell_errors_count[name] = cell_errors
        mean_r5 = statistics.mean(recall_5) if recall_5 else 0.0
        mean_mrr = statistics.mean(mrr_vals) if mrr_vals else 0.0
        mean_ndcg = statistics.mean(ndcg_vals) if ndcg_vals else 0.0
        wall = time.time() - t_cell
        if cell_latencies:
            sl = sorted(cell_latencies)
            lat_p50_cell = sl[len(sl) // 2]
            lat_p95_cell = sl[min(len(sl) - 1, int(0.95 * len(sl)))]
            lat_mean_cell = sum(cell_latencies) / len(cell_latencies)
        else:
            lat_p50_cell = lat_p95_cell = lat_mean_cell = 0.0
        per_cell_rows.append(
            {
                "cell": name,
                "fusion": fusion,
                "alpha": alpha_v if alpha_v is not None else "",
                "top_k_stage1": top_k_stage1,
                "rerank": int(do_rerank),
                "rerank_model": cell_rerank_model or "",
                "truncation": (cell_trunc_int if cell_trunc_int is not None else 0),
                "mode": cell_mode or "",
                "final_k": final_k,
                "n_queries": len(queries),
                "recall_at_5": f"{mean_r5:.4f}",
                "mrr10": f"{mean_mrr:.4f}",
                "ndcg_at_10": f"{mean_ndcg:.4f}",
                "gold_in_stage1_pool_frac": f"{gold_in_pool_count / len(queries):.4f}",
                "errors": cell_errors,
                "wall_sec": f"{wall:.1f}",
                "rerank_lat_p50_ms": f"{lat_p50_cell:.1f}",
                "rerank_lat_p95_ms": f"{lat_p95_cell:.1f}",
                "rerank_lat_mean_ms": f"{lat_mean_cell:.1f}",
            }
        )
        console.print(
            f"[bold]{name}[/] recall@5={mean_r5:.3f}  mrr10={mean_mrr:.3f}  "
            f"ndcg={mean_ndcg:.3f}  gold_in_pool={gold_in_pool_count}/{len(queries)}  "
            f"errors={cell_errors}  wall={wall:.1f}s"
        )

        # Kill criterion: per-cell rerank error rate > 5% over its 50 calls.
        if do_rerank and cell_errors > max(1, int(0.05 * len(queries))):
            console.print(
                f"[red]KILL[/] cell {name} rerank errors {cell_errors}/{len(queries)} > 5%"
            )
            # Continue running other cells but mark.

    # Write outputs.
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    with out_raw.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        w.writeheader()
        for r in per_query_rows:
            w.writerow(r)
    with out_per_cell.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_cell_rows[0].keys()))
        w.writeheader()
        for r in per_cell_rows:
            w.writerow(r)

    # Rerank stats.
    if rerank_latencies:
        sorted_lat = sorted(rerank_latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))]
    else:
        p50 = p95 = 0.0
    stats = {
        "rerank_call_count": rerank_call_count,
        "rerank_error_count": rerank_err_count,
        "rerank_error_rate": (rerank_err_count / rerank_call_count) if rerank_call_count else 0.0,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "latency_ms_mean": (sum(rerank_latencies) / len(rerank_latencies))
        if rerank_latencies
        else 0.0,
        "latency_ms_max": max(rerank_latencies) if rerank_latencies else 0.0,
    }
    out_rerank_stats.write_text(json.dumps(stats, indent=2) + "\n")

    # Verdicts.
    per_cell_by_name = {r["cell"]: r for r in per_cell_rows}

    if is_exp004c:
        # EXP-004c verdicts: H1 replication, H2 truncation monotone,
        # H3 RPC overhead, H4 model winner.
        b0 = float(per_cell_by_name["B0_alpha_baseline"]["recall_at_5"])
        q1 = float(per_cell_by_name["Q1_qwen3_1500c"]["recall_at_5"])
        q2 = float(per_cell_by_name["Q2_qwen3_2500c"]["recall_at_5"])
        q3 = float(per_cell_by_name["Q3_qwen3_notrunc"]["recall_at_5"])
        q4 = float(per_cell_by_name["Q4_qwen3_1500c_inproc"]["recall_at_5"])
        b1 = float(per_cell_by_name["B1_bge_1500c_inproc"]["recall_at_5"])

        # Q3 may have errored out entirely (OOM); guard.
        q3_errored = int(per_cell_by_name["Q3_qwen3_notrunc"].get("errors", 0)) >= len(queries)

        best_rerank = max(q1, q2, q3, q4, b1)
        h1_verdict = "CONFIRMED" if best_rerank >= 0.92 else "REFUTED"

        # H2: strict monotone Q3 > Q2 > Q1
        if q3_errored:
            h2_verdict = "REFUTED"
            h2_reason = "Q3 errored (no-truncation infeasible) — strict monotone undefined"
        elif (q3 > q2) and (q2 > q1):
            h2_verdict = "CONFIRMED"
            h2_reason = f"Q3({q3:.3f}) > Q2({q2:.3f}) > Q1({q1:.3f})"
        else:
            h2_verdict = "REFUTED"
            h2_reason = f"Q3={q3:.3f}, Q2={q2:.3f}, Q1={q1:.3f} — not strictly increasing"

        # H3: |Q4 - Q1| <= 0.02
        h3_delta = q4 - q1
        h3_verdict = "CONFIRMED" if abs(h3_delta) <= 0.02 else "REFUTED"

        # H4: Q4 vs B1 (both inproc)
        h4_delta = q4 - b1
        if q4 - b1 >= 0.05:
            h4_winner = "Qwen3-Reranker-0.6B"
        elif b1 - q4 >= 0.05:
            h4_winner = "bge-reranker-v2-m3"
        else:
            h4_winner = "tie (keep Qwen3)"

        # Paired Wilcoxon p-values for each rerank cell vs B0 (descriptive).
        wilcoxons: dict[str, dict[str, Any]] = {}
        for cell_name in (
            "Q1_qwen3_1500c",
            "Q2_qwen3_2500c",
            "Q3_qwen3_notrunc",
            "Q4_qwen3_1500c_inproc",
            "B1_bge_1500c_inproc",
        ):
            if cell_name not in cell_recalls_for_h2:
                continue
            deltas = [
                cell_recalls_for_h2[cell_name][i] - cell_recalls_for_h2["B0_alpha_baseline"][i]
                for i in range(len(queries))
            ]
            p = _wilcoxon_one_sided_greater([float(x) for x in deltas])
            n_pos = sum(1 for d in deltas if d > 0)
            n_neg = sum(1 for d in deltas if d < 0)
            wilcoxons[cell_name] = {"p": p, "pos": n_pos, "neg": n_neg}

        # SUMMARY.md
        lines: list[str] = []
        lines.append(f"# {slug} — reranker validation at higher N — SUMMARY\n")
        lines.append(f"N queries: {len(queries)}  KB: {kb_name}\n")
        lines.append("\n## Per-cell metrics\n")
        lines.append(
            "| cell | mode | rerank model | trunc | recall@5 | MRR@10 | nDCG@10 | gold-in-pool | errors | wall (s) | rerank p50 (ms) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in per_cell_rows:
            trunc_s = str(r.get("truncation", "")) if r.get("truncation") else "none"
            if r.get("truncation") == 0 and r["rerank"]:
                trunc_s = "none"
            lines.append(
                f"| {r['cell']} | {r.get('mode') or '—'} | "
                f"{r.get('rerank_model') or '—'} | {trunc_s if r['rerank'] else '—'} | "
                f"{r['recall_at_5']} | {r['mrr10']} | {r['ndcg_at_10']} | "
                f"{r['gold_in_stage1_pool_frac']} | {r['errors']} | {r['wall_sec']} | "
                f"{r.get('rerank_lat_p50_ms', '0.0')} |"
            )
        lines.append("\n## Hypothesis verdicts\n")
        lines.append(
            f"- **H1** (best reranked ≥ 0.92, +10pp over B0={b0:.3f}): **{h1_verdict}**  "
            f"max(rerank cells) = {best_rerank:.3f}; delta over B0 = {best_rerank - b0:+.3f}"
        )
        lines.append(f"- **H2** (truncation monotone Q3>Q2>Q1): **{h2_verdict}**  {h2_reason}")
        lines.append(
            f"- **H3** (|Q4-Q1| ≤ 0.02 — RPC overhead): **{h3_verdict}**  "
            f"Q4={q4:.3f}, Q1={q1:.3f}, delta={h3_delta:+.3f}"
        )
        lines.append(
            f"- **H4** (Qwen3 vs BGE, inproc): winner = **{h4_winner}**  "
            f"Q4={q4:.3f}, B1={b1:.3f}, delta={h4_delta:+.3f}"
        )
        lines.append("\n## Wilcoxon vs B0 (one-sided, treat > control)\n")
        for cell_name, w in wilcoxons.items():
            ties = len(queries) - w["pos"] - w["neg"]
            lines.append(
                f"- {cell_name}: +{w['pos']} / -{w['neg']} / ties={ties}; " f"p = {w['p']:.4f}"
            )
        lines.append("\n## Rerank-service stats (RPC cells only)\n")
        lines.append(f"- calls: {stats['rerank_call_count']}")
        lines.append(
            f"- errors: {stats['rerank_error_count']} ({100*stats['rerank_error_rate']:.1f}%)"
        )
        lines.append(f"- latency p50: {stats['latency_ms_p50']:.1f} ms")
        lines.append(f"- latency p95: {stats['latency_ms_p95']:.1f} ms")
        lines.append(f"- latency mean: {stats['latency_ms_mean']:.1f} ms")
        lines.append(f"- latency max: {stats['latency_ms_max']:.1f} ms")
        out_summary.write_text("\n".join(lines) + "\n")

        # verdicts.md
        vlines: list[str] = []
        vlines.append(f"# {slug} — verdicts\n")
        vlines.append(f"Pre-registered in `docs/exp/{slug}.md`.\n")

        vlines.append("\n## H1 — best reranked cell ≥ 0.92 (i.e. +10pp over B0)\n")
        vlines.append(f"**Verdict: {h1_verdict}**\n")
        vlines.append(f"- B0 (alpha=0.75, no rerank) recall@5: {b0:.3f}")
        vlines.append(f"- Q1 (Qwen3 + 1500c, rpc) recall@5: {q1:.3f}")
        vlines.append(f"- Q2 (Qwen3 + 2500c, rpc) recall@5: {q2:.3f}")
        if q3_errored:
            vlines.append("- Q3 (Qwen3 + no-trunc, rpc) recall@5: ERRORED (OOM expected)")
        else:
            vlines.append(f"- Q3 (Qwen3 + no-trunc, rpc) recall@5: {q3:.3f}")
        vlines.append(f"- Q4 (Qwen3 + 1500c, inproc) recall@5: {q4:.3f}")
        vlines.append(f"- B1 (BGE + 1500c, inproc) recall@5: {b1:.3f}")
        vlines.append(f"- max(rerank cells) = {best_rerank:.3f}; threshold = 0.920")
        vlines.append(f"- delta over B0: {best_rerank - b0:+.3f}")

        vlines.append("\n## H2 — truncation monotone: Q3 > Q2 > Q1\n")
        vlines.append(f"**Verdict: {h2_verdict}**\n")
        vlines.append(f"- {h2_reason}")

        vlines.append("\n## H3 — RPC overhead: |Q4 - Q1| ≤ 0.02\n")
        vlines.append(f"**Verdict: {h3_verdict}**\n")
        vlines.append(f"- Q4 (in-process) recall@5: {q4:.3f}")
        vlines.append(f"- Q1 (rpc) recall@5: {q1:.3f}")
        vlines.append(f"- delta (Q4 - Q1): {h3_delta:+.3f}  (threshold ±0.020)")

        vlines.append("\n## H4 — rerank-model comparison: Q4 (Qwen3 inproc) vs B1 (BGE inproc)\n")
        vlines.append(f"**Winner: {h4_winner}**\n")
        vlines.append(f"- Q4 (Qwen3-Reranker-0.6B): {q4:.3f}")
        vlines.append(f"- B1 (bge-reranker-v2-m3): {b1:.3f}")
        vlines.append(f"- delta (Q4 - B1): {h4_delta:+.3f}  (threshold ±0.050)")

        vlines.append("\n## Wilcoxon vs B0 (one-sided, treat > control)\n")
        for cell_name, w in wilcoxons.items():
            ties = len(queries) - w["pos"] - w["neg"]
            vlines.append(
                f"- {cell_name}: +{w['pos']} / -{w['neg']} / ties={ties}; " f"p = {w['p']:.4f}"
            )

        out_verdicts.write_text("\n".join(vlines) + "\n")

        # Console summary
        total_errors = sum(int(r["errors"]) for r in per_cell_rows)
        console.print(
            f"\n[bold]DONE[/] {slug} cells={len(per_cell_rows)} queries={len(queries)} "
            f"rerank_calls={rerank_call_count} rerank_errs={rerank_err_count} "
            f"total_errors={total_errors}\n"
            f"  H1 (≥0.92): {h1_verdict}  best={best_rerank:.3f} (B0={b0:.3f})\n"
            f"  H2 (Q3>Q2>Q1): {h2_verdict}  {h2_reason}\n"
            f"  H3 (|Q4-Q1|≤0.02): {h3_verdict}  Q4={q4:.3f} Q1={q1:.3f} d={h3_delta:+.3f}\n"
            f"  H4 winner: {h4_winner}  (Q4={q4:.3f}, B1={b1:.3f})\n"
        )

    else:
        # EXP-004a verdicts (legacy path).
        c0 = float(per_cell_by_name["C0_alpha_baseline"]["recall_at_5"])
        c1 = float(per_cell_by_name["C1_rrf_baseline"]["recall_at_5"])
        c2 = float(per_cell_by_name["C2_alpha_rerank"]["recall_at_5"])
        c3 = float(per_cell_by_name["C3_rrf_rerank"]["recall_at_5"])
        best_rerank = max(c2, c3)
        h1_verdict = "CONFIRMED" if best_rerank >= 0.92 else "REFUTED"

        deltas_c2_vs_c0 = [
            cell_recalls_for_h2["C2_alpha_rerank"][i] - cell_recalls_for_h2["C0_alpha_baseline"][i]
            for i in range(len(queries))
        ]
        deltas_c3_vs_c1 = [
            cell_recalls_for_h2["C3_rrf_rerank"][i] - cell_recalls_for_h2["C1_rrf_baseline"][i]
            for i in range(len(queries))
        ]
        p_c2 = _wilcoxon_one_sided_greater([float(x) for x in deltas_c2_vs_c0])
        p_c3 = _wilcoxon_one_sided_greater([float(x) for x in deltas_c3_vs_c1])
        h2_verdict = "CONFIRMED" if (p_c2 < 0.05 and p_c3 < 0.05) else "REFUTED"

        h3_delta_alpha = c1 - c0
        h3_delta_rerank_arm = c3 - c2

        # SUMMARY.md
        lines = []
        lines.append(f"# {slug} — reranker validation — SUMMARY\n")
        lines.append(f"N queries: {len(queries)}  KB: {kb_name}  Rerank model: {rerank_model}\n")
        lines.append("\n## Per-cell metrics\n")
        lines.append(
            "| cell | fusion | alpha | stage-1 top-k | rerank | final-k | recall@5 | MRR@10 | nDCG@10 | gold-in-pool | errors | wall (s) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in per_cell_rows:
            lines.append(
                f"| {r['cell']} | {r['fusion']} | {r['alpha']} | {r['top_k_stage1']} | "
                f"{'yes' if r['rerank'] else 'no'} | {r['final_k']} | "
                f"{r['recall_at_5']} | {r['mrr10']} | {r['ndcg_at_10']} | "
                f"{r['gold_in_stage1_pool_frac']} | {r['errors']} | {r['wall_sec']} |"
            )
        lines.append("\n## Hypothesis verdicts\n")
        lines.append(
            f"- H1 (aggressive, ≥0.92 best reranked): **{h1_verdict}**  "
            f"max(C2,C3)={best_rerank:.3f}; C0 baseline={c0:.3f}; delta={best_rerank-c0:+.3f}"
        )
        lines.append(
            f"- H2 (rerank always improves, paired Wilcoxon both p<0.05): **{h2_verdict}**  "
            f"C2 vs C0 p={p_c2:.4f}; C3 vs C1 p={p_c3:.4f}"
        )
        lines.append(
            f"- H3 (informational): delta_alpha (C1-C0)={h3_delta_alpha:+.3f}; "
            f"delta_rerank_arm (C3-C2)={h3_delta_rerank_arm:+.3f}"
        )
        lines.append("\n## Rerank-service stats\n")
        lines.append(f"- calls: {stats['rerank_call_count']}")
        lines.append(
            f"- errors: {stats['rerank_error_count']} ({100*stats['rerank_error_rate']:.1f}%)"
        )
        lines.append(f"- latency p50: {stats['latency_ms_p50']:.1f} ms")
        lines.append(f"- latency p95: {stats['latency_ms_p95']:.1f} ms")
        lines.append(f"- latency mean: {stats['latency_ms_mean']:.1f} ms")
        lines.append(f"- latency max: {stats['latency_ms_max']:.1f} ms")
        out_summary.write_text("\n".join(lines) + "\n")

        # verdicts.md
        vlines = []
        vlines.append(f"# {slug} — verdicts\n")
        vlines.append(f"Pre-registered in `docs/exp/{slug}.md`.\n")
        vlines.append("\n## H1 — best reranked cell ≥ 0.92 (i.e. +10pp over C0=0.820)\n")
        vlines.append(f"**Verdict: {h1_verdict}**\n")
        vlines.append(f"- C0 (alpha=0.75, no rerank) recall@5: {c0:.3f}")
        vlines.append(f"- C1 (RRF, no rerank) recall@5: {c1:.3f}")
        vlines.append(f"- C2 (alpha=0.75 + rerank) recall@5: {c2:.3f}")
        vlines.append(f"- C3 (RRF + rerank) recall@5: {c3:.3f}")
        vlines.append(f"- max(C2, C3) = {best_rerank:.3f}; threshold = 0.920")
        vlines.append(f"- delta over C0: {best_rerank - c0:+.3f}")

        vlines.append(
            "\n## H2 — rerank always improves (paired Wilcoxon, one-sided, both p<0.05)\n"
        )
        vlines.append(f"**Verdict: {h2_verdict}**\n")
        n_pos_c2 = sum(1 for d in deltas_c2_vs_c0 if d > 0)
        n_neg_c2 = sum(1 for d in deltas_c2_vs_c0 if d < 0)
        n_pos_c3 = sum(1 for d in deltas_c3_vs_c1 if d > 0)
        n_neg_c3 = sum(1 for d in deltas_c3_vs_c1 if d < 0)
        vlines.append(
            f"- C2 vs C0: +{n_pos_c2} / -{n_neg_c2} / ties={len(queries)-n_pos_c2-n_neg_c2}; "
            f"Wilcoxon one-sided p = {p_c2:.4f}"
        )
        vlines.append(
            f"- C3 vs C1: +{n_pos_c3} / -{n_neg_c3} / ties={len(queries)-n_pos_c3-n_neg_c3}; "
            f"Wilcoxon one-sided p = {p_c3:.4f}"
        )

        vlines.append("\n## H3 — RRF beats alpha-blend as stage-1 (informational)\n")
        vlines.append(f"- delta_alpha (C1 - C0): {h3_delta_alpha:+.3f}")
        vlines.append(f"- delta_rerank_arm (C3 - C2): {h3_delta_rerank_arm:+.3f}")
        if h3_delta_alpha < 0 or h3_delta_rerank_arm < 0:
            vlines.append("- NOTE: at least one comparison favors alpha-blend; see SUMMARY.md")

        out_verdicts.write_text("\n".join(vlines) + "\n")

        # Final console summary.
        total_errors = sum(int(r["errors"]) for r in per_cell_rows)
        console.print(
            f"\n[bold]DONE[/] cells={len(per_cell_rows)} queries={len(queries)} "
            f"rerank_calls={rerank_call_count} rerank_errs={rerank_err_count} "
            f"rerank_p50={p50:.0f}ms total_errors={total_errors}\n"
            f"  H1: {h1_verdict}  (best reranked recall@5 = {best_rerank:.3f}; threshold 0.920)\n"
            f"  H2: {h2_verdict}  (p_C2={p_c2:.4f}, p_C3={p_c3:.4f})\n"
            f"  H3 informational: delta_alpha={h3_delta_alpha:+.3f}, "
            f"delta_rerank_arm={h3_delta_rerank_arm:+.3f}"
        )

    # Honor kill criterion overall: > 5% of all rerank calls failing.
    if rerank_call_count and (rerank_err_count / rerank_call_count) > 0.05:
        console.print("[red]KILL criterion tripped: rerank error rate > 5% overall[/]")
        return 3

    return 0


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main(config_path: str) -> int:
    cfg = yaml.safe_load(Path(config_path).read_text())
    # Dispatch on shape: per-cell configs (EXP-004a) vs alpha x k matrix.
    if isinstance(cfg.get("cells"), list):
        return run_per_cell_sweep(cfg)
    kb_name = cfg["kb"]["name"]
    kb_dir = KB_ROOT / kb_name
    n_target = int(cfg["queries"]["n_target"])
    question_model = cfg["queries"]["question_model"]
    cache_path = REPO_ROOT / cfg["queries"]["cache_path"]

    out_summary = REPO_ROOT / cfg["outputs"]["summary"]
    out_verdicts = REPO_ROOT / cfg["outputs"]["verdicts"]
    out_raw = REPO_ROOT / cfg["outputs"]["raw"]
    out_best = REPO_ROOT / cfg["outputs"]["best_configs"]

    pilot_cells = cfg.get("pilot", {}).get("cells", [])
    pilot_n_queries = int(cfg.get("pilot", {}).get("n_queries", PILOT_QUERIES))

    console.print(f"[bold]EXP-003a[/] starting on KB={kb_name} ({kb_dir})")
    t0 = time.time()

    # Step 1: queries
    queries = generate_or_load_queries(cache_path, kb_dir, n_target, question_model)
    if len(queries) < 25:
        console.print(f"[red]KILL[/] only {len(queries)} valid queries (< 25 threshold)")
        return 2
    console.print(f"[green]queries[/] {len(queries)} ready  ({time.time() - t0:.0f}s)")

    # Step 2: open KB once
    db = lancedb.connect(str(kb_dir / "index"))
    tbl = db.open_table(TABLE_NAME)
    all_rows = tbl.to_arrow().to_pylist()
    console.print(f"[dim]KB table loaded: {len(all_rows)} rows")

    # Step 3: embed all queries once (cached for all 20 cells)
    console.print("[bold]embedding[/] queries (once, reused across cells)…")
    t_emb = time.time()
    texts = [q.question for q in queries]
    res = embed_texts(texts, model=DEFAULT_EMBED_MODEL, batch_size=8)
    if len(res.vectors) != len(queries):
        console.print(f"[red]KILL[/] embedding count mismatch {len(res.vectors)} vs {len(queries)}")
        return 2
    for q, v in zip(queries, res.vectors, strict=True):
        q.qvec = v
    console.print(f"[green]embeddings[/] {len(queries)} done ({time.time() - t_emb:.0f}s)")

    # Step 4: PILOT
    pilot_queries = queries[:pilot_n_queries]
    pilot_alphas = sorted({c["alpha"] for c in pilot_cells})
    pilot_ks = sorted({c["k"] for c in pilot_cells})
    console.print(
        f"[bold yellow]PILOT[/] {len(pilot_queries)} queries × "
        f"{len(pilot_alphas)} alphas × {len(pilot_ks)} ks"
    )
    pilot_metrics, pilot_rows = run_cells(pilot_queries, tbl, all_rows, pilot_alphas, pilot_ks)
    pilot_csv = out_raw.parent / "pilot_raw.csv"
    write_raw_csv(pilot_rows, pilot_csv)
    console.print(f"[green]pilot[/] CSV written to {pilot_csv}")
    for m in pilot_metrics:
        if m.recall <= 0.0 and m.n_errors == 0:
            console.print(
                f"[yellow]pilot warning[/] alpha={m.alpha} k={m.k} "
                f"recall=0 across {m.n_queries} queries — sanity-check origin alignment"
            )

    # Step 5: FULL SWEEP
    console.print(f"[bold green]FULL SWEEP[/] {len(ALL_ALPHAS)} alphas × {len(ALL_KS)} ks")
    full_metrics, full_rows = run_cells(queries, tbl, all_rows, ALL_ALPHAS, ALL_KS)

    write_raw_csv(full_rows, out_raw)
    write_best_configs(full_metrics, out_best)
    write_summary_md(full_metrics, out_summary)
    verdicts = compute_verdicts(full_metrics)
    write_verdicts_md(verdicts, out_verdicts)

    total_errors = sum(m.n_errors for m in full_metrics)
    console.print(
        f"\n[bold]DONE[/] wall={(time.time() - t0):.0f}s  cells={len(full_metrics)}  "
        f"errors={total_errors}  verdicts: "
        f"H1={verdicts['H1']['verdict']}  "
        f"H2={verdicts['H2']['verdict']}  "
        f"H3={verdicts['H3']['verdict']}"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: retrieval_sweep.py <path-to-EXP-003a.yaml>", file=sys.stderr)
        sys.exit(2)
    raise SystemExit(main(sys.argv[1]))
