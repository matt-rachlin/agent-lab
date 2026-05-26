"""EXP-003a: bash KB retrieval-quality sweep -- alpha x k.

Reads conf/sweep/EXP-003a.yaml directly (not via lab.sweep.config -- that
schema is for agent sweeps; this is a pure-retrieval sweep).

Pipeline:
  1. Generate / load cached synthetic queries (~50) via the lab.rag
     eval_retrieval module style (sample chunks, ask qwen3:14b-q4_K_M
     to write a realistic question). Cached at queries.jsonl.
  2. Embed each query ONCE (qwen3-embedding:8b-q8_0) -- embeddings reused
     across all 20 cells, so the sweep cost is ~50 embeddings + 1000
     sparse + LanceDB lookups (not 1000 embeddings).
  3. Pilot: alpha=0.5/k=5 and alpha=0.0/k=5 x 5 queries. Sanity-check
     CSV shape + scorer numerics before committing 20 cells x 50 queries.
  4. Full sweep: 5 alpha x 4 k = 20 cells. For each cell + query: rank
     of origin chunk in the result (via hybrid_query at k=max(10,k)),
     compute recall@k_target, MRR@10, nDCG@10.
  5. Write raw.csv, SUMMARY.md, best_configs.csv, verdicts.md.

Hypotheses (pre-registered in docs/exp/EXP-003a.md):
  H1: argmax_alpha mean(recall@5) in {0.25, 0.5, 0.75}  (hybrid beats endpoints)
  H2: mean(recall@10) - mean(recall@5) >= 0.10  (top-k matters)
  H3: at best k per endpoint, BM25 (alpha=0.0) beats dense (alpha=1.0)
      on at least one of {recall@5, MRR@10}.
"""

from __future__ import annotations

import csv
import json
import math
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
                "content": (
                    f"Section: {sec}\nPassage:\n---\n{chunk_text[:1500]}\n---\nQuestion:"
                ),
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
            q = _gen_question(client, row["text"], list(row.get("section_path") or []), question_model)
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
            scored.append(RankedHit(chunk_id=cid, score=combined, dense_score=d_score, sparse_score=s_score))
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


def _bootstrap_ci(values: list[float], n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
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
            val = (
                cm.recall
                if "recall" in name
                else cm.mrr10
                if "mrr" in name
                else cm.ndcg
            )
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
        "MIXED"
        if tied_with_endpoint
        else ("CONFIRMED" if h1_alpha_star is not None else "REFUTED")
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
# main
# ----------------------------------------------------------------------


def main(config_path: str) -> int:
    cfg = yaml.safe_load(Path(config_path).read_text())
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
