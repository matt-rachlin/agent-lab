"""LanceDB writer + hybrid retrieval router.

Vendored from kb_builder.index.

Layout per KB: <kb_dir>/index/chunks.lance/  (a LanceDB dataset directory).
We store dense vectors and a JSON-encoded sparse vector. Hybrid query
combines a vector search with a BM25 score computed over the chunks'
sparse vectors at query time (in-process; corpus is small enough to scan).

Phase 7 (2026-05-26): the default fusion strategy is now Reciprocal Rank
Fusion (RRF, Cormack et al. 2009) — rank-based, score-normalization-free,
zero hyperparameters per query. Alpha-blend stays reachable via
``fusion="alpha"`` for back-compat / ablation runs. A cross-encoder
reranker (see :mod:`lab.rag.rerank`) is layered on top by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lancedb
import pyarrow as pa

from lab.rag import RRF_K
from lab.rag.embedder import embed_texts, tokenize_for_bm25

logger = logging.getLogger(__name__)

TABLE_NAME = "chunks"

#: Fusion strategies recognised by :func:`hybrid_query`. Default is ``"rrf"``.
FusionStrategy = Literal["rrf", "alpha"]


def _schema(dims: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("source_id", pa.string()),
            pa.field("source_url", pa.string()),
            pa.field("source_sha256", pa.string()),
            pa.field("retrieved_at", pa.string()),
            pa.field("doc_path", pa.string()),
            pa.field("section_path", pa.list_(pa.string())),
            pa.field("byte_start", pa.int64()),
            pa.field("byte_end", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("title", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("prerequisites", pa.list_(pa.string())),
            pa.field("vector", pa.list_(pa.float32(), dims)),
            pa.field("sparse_json", pa.string()),
            pa.field("tokens", pa.int64()),
            pa.field("chunk_format_version", pa.int32()),
            pa.field("authority", pa.string()),
        ]
    )


def open_table(kb_dir: Path, dims: int) -> Any:
    db = lancedb.connect(str(kb_dir / "index"))
    if TABLE_NAME in db.list_tables().tables:
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=_schema(dims))


def write_rows(kb_dir: Path, rows: list[dict[str, Any]], dims: int) -> None:
    tbl = open_table(kb_dir, dims)
    tbl.add(rows)


def replace_table(kb_dir: Path, rows: list[dict[str, Any]], dims: int) -> None:
    db = lancedb.connect(str(kb_dir / "index"))
    if TABLE_NAME in db.list_tables().tables:
        db.drop_table(TABLE_NAME)
    tbl = db.create_table(TABLE_NAME, schema=_schema(dims))
    if rows:
        tbl.add(rows)


@dataclass
class Hit:
    chunk_id: str
    text: str
    title: str
    summary: str
    source_url: str
    retrieved_at: str
    section_path: list[str]
    score: float
    dense_score: float
    sparse_score: float
    authority: str
    #: Stage-2 cross-encoder score (raw float). ``None`` when reranking was
    #: skipped (e.g. ``rerank=False`` or :data:`LAB_RAG_RERANKER=none`).
    rerank_score: float | None = None
    #: Rank in the stage-1 candidate set (1-based) — useful for telemetry.
    stage1_rank: int | None = None


def _dense_search(tbl: Any, query_vector: list[float], k: int) -> list[dict[str, Any]]:
    res: list[dict[str, Any]] = tbl.search(query_vector).limit(k).to_list()
    return res


def _bm25_scan(rows: list[dict[str, Any]], query_text: str, k: int) -> list[tuple[int, float]]:
    """Score every row by BM25 overlap between query tokens and the row's
    pre-computed sparse vector (term -> weight).
    """
    import json

    q = tokenize_for_bm25(query_text)
    if not q:
        return []
    scores: list[tuple[int, float]] = []
    for idx, row in enumerate(rows):
        sj = row.get("sparse_json") or "{}"
        try:
            sp = json.loads(sj)
        except Exception:
            sp = {}
        s = sum(sp.get(tok, 0.0) for tok in q)
        if s > 0:
            scores.append((idx, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def _rrf_fuse(
    dense_ranking: list[str],
    sparse_ranking: list[str],
    k_const: int = RRF_K,
) -> dict[str, float]:
    """Reciprocal Rank Fusion (Cormack et al. 2009).

    Returns ``{chunk_id: score}`` where ``score = sum(1 / (k + rank))`` across
    the two rankings. ``rank`` is 1-based; missing candidates contribute 0.
    """
    fused: dict[str, float] = {}
    for rank, cid in enumerate(dense_ranking, start=1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k_const + rank)
    for rank, cid in enumerate(sparse_ranking, start=1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k_const + rank)
    return fused


def _stage1_candidates(
    *,
    tbl: Any,
    query_text: str,
    qvec: list[float],
    pool_size: int,
    fusion: FusionStrategy,
    alpha: float | None,
    filter_authority: str | None,
) -> list[tuple[str, float, float, float, dict[str, Any]]]:
    """Run stage-1 retrieval and return scored candidates ordered best-first.

    Returned tuple shape: ``(chunk_id, combined, dense_score, sparse_score, row)``.
    """
    dense = _dense_search(tbl, qvec, pool_size)
    if filter_authority:
        dense = [r for r in dense if r.get("authority") == filter_authority]

    if not dense:
        return []

    # Dense-only path: no need to pull all rows for BM25.
    if fusion == "alpha" and alpha is not None and alpha >= 1.0:
        scored: list[tuple[str, float, float, float, dict[str, Any]]] = []
        max_d = max((1.0 / (1.0 + float(r.get("_distance", 1.0))) for r in dense), default=1.0)
        for r in dense[:pool_size]:
            d_sim = (1.0 / (1.0 + float(r.get("_distance", 1.0)))) / max_d
            scored.append((r["chunk_id"], d_sim, d_sim, 0.0, r))
        return scored

    all_rows = tbl.to_arrow().to_pylist()
    if filter_authority:
        all_rows = [r for r in all_rows if r.get("authority") == filter_authority]
    sparse_top = _bm25_scan(all_rows, query_text, pool_size)

    # Build candidate union by chunk_id.
    seen: dict[str, dict[str, Any]] = {}
    for r in dense:
        seen[r["chunk_id"]] = r
    sparse_score_by_id: dict[str, float] = {}
    for idx, s in sparse_top:
        r = all_rows[idx]
        seen.setdefault(r["chunk_id"], r)
        sparse_score_by_id[r["chunk_id"]] = s

    # Per-candidate dense similarity (used for telemetry on every path).
    dense_distances = {r["chunk_id"]: float(r.get("_distance", 1.0)) for r in dense}
    d_sims_raw = {cid: 1.0 / (1.0 + d) for cid, d in dense_distances.items()}
    max_dsim = max(d_sims_raw.values()) if d_sims_raw else 1.0
    d_sims = {cid: v / max_dsim for cid, v in d_sims_raw.items()}

    max_sparse = max(sparse_score_by_id.values()) if sparse_score_by_id else 1.0
    s_norms = {cid: v / max_sparse for cid, v in sparse_score_by_id.items()}

    if fusion == "rrf":
        dense_ranking = [r["chunk_id"] for r in dense]
        sparse_ranking = [all_rows[idx]["chunk_id"] for idx, _ in sparse_top]
        fused = _rrf_fuse(dense_ranking, sparse_ranking)
        scored = []
        for cid, row in seen.items():
            combined = fused.get(cid, 0.0)
            scored.append(
                (cid, combined, d_sims.get(cid, 0.0), s_norms.get(cid, 0.0), row)
            )
    else:
        # alpha-blend (legacy) with explicit alpha
        a = 0.5 if alpha is None else alpha
        scored = []
        for cid, row in seen.items():
            d_score = d_sims.get(cid, 0.0)
            s_score = s_norms.get(cid, 0.0)
            combined = a * d_score + (1.0 - a) * s_score
            scored.append((cid, combined, d_score, s_score, row))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _row_to_hit(
    cid: str,
    combined: float,
    d_score: float,
    s_score: float,
    row: dict[str, Any],
    *,
    rerank_score: float | None = None,
    stage1_rank: int | None = None,
) -> Hit:
    return Hit(
        chunk_id=cid,
        text=row.get("text", "") or "",
        title=row.get("title", "") or "",
        summary=row.get("summary", "") or "",
        source_url=row.get("source_url", "") or "",
        retrieved_at=row.get("retrieved_at", "") or "",
        section_path=list(row.get("section_path") or []),
        score=combined,
        dense_score=d_score,
        sparse_score=s_score,
        authority=row.get("authority", "") or "",
        rerank_score=rerank_score,
        stage1_rank=stage1_rank,
    )


def hybrid_query(
    kb_dir: Path,
    query_text: str,
    *,
    k: int = 5,
    fusion: FusionStrategy = "rrf",
    rerank: bool = True,
    top_k_stage1: int = 50,
    alpha: float | None = None,
    model: str | None = None,
    filter_authority: str | None = None,
) -> list[Hit]:
    """Hybrid dense + sparse retrieval with optional cross-encoder rerank.

    Pipeline:
      1. **Stage-1** retrieves a pool of ``top_k_stage1`` candidates fused by
         ``fusion`` (``"rrf"`` — default — or ``"alpha"`` with explicit
         ``alpha`` weight).
      2. **Stage-2** (optional, on by default) re-scores stage-1 with a
         cross-encoder reranker and returns the top ``k``.

    When ``rerank=False`` or ``LAB_RAG_RERANKER=none`` the reranker is bypassed
    and the top ``k`` come straight from stage-1. Phase 10 hooks (skip
    heuristics + cosine dedupe) live in this same call site.

    Back-compat: callers that pass ``alpha=`` and not ``fusion=`` get the
    legacy alpha-blend (no surprise behaviour change for ablations or older
    scripts). Pass ``fusion="rrf"`` explicitly to force RRF even with an alpha
    value supplied for telemetry.

    Returns ``[]`` (without calling the embedding model) if the KB has no
    index yet — so callers can smoke-test against in-progress KBs without
    burning GPU time.
    """
    from lab.rag import DEFAULT_EMBED_MODEL

    model = model or DEFAULT_EMBED_MODEL

    # Early-return on missing/empty index: avoids hitting Ollama for an empty KB.
    if not (kb_dir / "index").exists():
        return []
    db = lancedb.connect(str(kb_dir / "index"))
    if TABLE_NAME not in db.list_tables().tables:
        return []
    tbl = db.open_table(TABLE_NAME)
    total_rows = tbl.count_rows()
    if total_rows == 0:
        return []

    # Honour the legacy contract: when a caller passes alpha=... but no
    # explicit fusion, fall back to alpha-blend for back-compat.
    if alpha is not None and fusion == "rrf":
        # Caller passed alpha but didn't override fusion — they're on the
        # legacy path. Switch to alpha-blend silently.
        fusion = "alpha"

    pool = max(top_k_stage1, k, 40)
    qvec = embed_texts([query_text], model=model, batch_size=1).vectors[0]

    stage1 = _stage1_candidates(
        tbl=tbl,
        query_text=query_text,
        qvec=qvec,
        pool_size=pool,
        fusion=fusion,
        alpha=alpha,
        filter_authority=filter_authority,
    )
    if not stage1:
        return []

    # Phase 10 hooks: skip + dedupe live in their own module so they're easy
    # to unit-test without spinning up LanceDB.
    from lab.rag.skip import compute_skip_decision, dedupe_candidates

    skip_decision = compute_skip_decision(
        candidates=[
            {
                "chunk_id": cid,
                "score": combined,
                "dense_score": d,
                "sparse_score": s,
            }
            for cid, combined, d, s, _ in stage1
        ],
        total_kb_rows=total_rows,
        rerank_requested=rerank,
    )

    use_reranker = skip_decision.use_reranker

    if use_reranker:
        # Dedupe before sending to the reranker — keeps the cross-encoder
        # call list short.
        candidate_dicts: list[dict[str, Any]] = []
        for stage_rank, (cid, combined, d_score, s_score, row) in enumerate(stage1, start=1):
            candidate_dicts.append(
                {
                    "chunk_id": cid,
                    "combined": combined,
                    "dense_score": d_score,
                    "sparse_score": s_score,
                    "row": row,
                    "stage1_rank": stage_rank,
                    "text": row.get("text", "") or "",
                    "vector": row.get("vector"),
                }
            )
        deduped, dupe_clusters = dedupe_candidates(candidate_dicts)

        from lab.rag.rerank import get_default_reranker

        reranker = get_default_reranker()
        if reranker.disabled:
            # Honour the env-level disable even when rerank=True — easier to
            # turn off globally without code edits.
            hits = [
                _row_to_hit(
                    c["chunk_id"],
                    c["combined"],
                    c["dense_score"],
                    c["sparse_score"],
                    c["row"],
                    rerank_score=None,
                    stage1_rank=c["stage1_rank"],
                )
                for c in deduped[:k]
            ]
        else:
            reranked = reranker.rerank(query_text, deduped, top_n=k)
            hits = [
                _row_to_hit(
                    c["chunk_id"],
                    c["combined"],
                    c["dense_score"],
                    c["sparse_score"],
                    c["row"],
                    rerank_score=float(c.get("rerank_score", 0.0)),
                    stage1_rank=c["stage1_rank"],
                )
                for c in reranked
            ]
        # Phase 10 low-confidence alert.
        from lab.rag.skip import maybe_emit_low_confidence

        maybe_emit_low_confidence(hits, kb_dir=kb_dir)
        _ = dupe_clusters  # surface kept for future telemetry hookup
    else:
        logger.debug("rerank skipped: %s", skip_decision.reason)
        hits = [
            _row_to_hit(cid, combined, d, s, row, rerank_score=None, stage1_rank=rank)
            for rank, (cid, combined, d, s, row) in enumerate(stage1[:k], start=1)
        ]
    return hits


def count_rows(kb_dir: Path) -> int:
    if not (kb_dir / "index").exists():
        return 0
    db = lancedb.connect(str(kb_dir / "index"))
    if TABLE_NAME not in db.list_tables().tables:
        return 0
    rows: int = db.open_table(TABLE_NAME).count_rows()
    return rows


def index_bytes(kb_dir: Path) -> int:
    p = kb_dir / "index"
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total
