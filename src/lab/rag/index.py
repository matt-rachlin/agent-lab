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
import math
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
            # ---- Phase 9 (v2) parent-child fields. ---------------------
            # All three are nullable / default-false so a v1 KB migrated by
            # ``scripts/migrate_kb_schema_v2.py`` reads cleanly and a freshly
            # built v2 KB populates them per chunk.
            pa.field("parent_chunk_id", pa.string()),
            pa.field("child_index", pa.int32()),
            pa.field("is_parent", pa.bool_()),
            # ---- Phase 11 (v3) HyPE fields. ----------------------------
            # Both nullable so v1/v2 rows survive deserialisation under the
            # v3 reader. ``hype_questions`` is a parallel list to
            # ``hype_vectors`` (same N per row); when both are non-null and
            # ``use_hype=True`` at query time, the dense score becomes
            # ``max(content_cos, max(question_cos))``.
            pa.field("hype_questions", pa.list_(pa.string())),
            pa.field("hype_vectors", pa.list_(pa.list_(pa.float32(), dims))),
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
    #: Phase 9: when ``expand_to_parent=True`` and the matched chunk is a
    #: child, ``text`` is replaced with the parent's text and ``child_offset``
    #: points at where the child's text starts inside the parent. On v1 KBs
    #: (FLAT) and on chunks that are already parents, ``child_offset`` stays
    #: ``None`` and ``text`` is unchanged.
    child_offset: int | None = None
    #: Parent chunk id if this hit comes from a parent-child KB and the row
    #: was a child. None for FLAT KBs and for parent rows.
    parent_chunk_id: str | None = None
    #: True when ``text`` is parent text expanded around the matched child.
    expanded_to_parent: bool = False


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


def _resolve_cache_key(kb_dir: Path, *, top_k: int) -> tuple[str, int] | None:
    """Return ``(kb_version, top_k)`` if the KB manifest is readable, else None.

    Failure to load the manifest is non-fatal — we just bypass the rerank
    cache for this call.
    """
    try:
        from lab.rag.cache import kb_version_token
        from lab.rag.manifest import load_manifest

        manifest = load_manifest(kb_dir / "manifest.yaml")
        return (kb_version_token(manifest), int(top_k))
    except Exception:
        return None


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


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Plain cosine between two float vectors.

    Defensive: returns 0.0 on missing inputs, mismatched dims, or zero norm.
    Pure-Python so test fakes don't have to pull in numpy.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _row_has_hype(row: dict[str, Any]) -> bool:
    """True iff the row carries at least one usable HyPE vector."""
    vecs = row.get("hype_vectors")
    if not vecs:
        return False
    # ``vecs`` is a list of vectors; the row is "hype-enabled" if any
    # individual entry is a non-empty sequence.
    return any(v for v in vecs)


def _hype_boost_dsim(
    row: dict[str, Any],
    qvec: list[float],
    content_dsim: float,
) -> float:
    """Return ``max(content_dsim, max_question_cos)`` for a row.

    ``content_dsim`` is the row's pre-normalised dense similarity (already
    in the range used by the rest of the stage-1 pipeline). The HyPE
    component is the raw maximum cosine across the row's stored question
    vectors; we keep raw cosine here because HyPE is meant to *lift* the
    dsim of an on-question match, not just compete with it. Callers that
    don't want this behaviour pass ``use_hype=False``.
    """
    vecs = row.get("hype_vectors") or []
    if not vecs:
        return content_dsim
    best = content_dsim
    for v in vecs:
        if not v:
            continue
        c = _cosine(qvec, v)
        if c > best:
            best = c
    return best


def _stage1_candidates(
    *,
    tbl: Any,
    query_text: str,
    qvec: list[float],
    pool_size: int,
    fusion: FusionStrategy,
    alpha: float | None,
    filter_authority: str | None,
    use_hype: bool = False,
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
            if use_hype and _row_has_hype(r):
                # HyPE lift: take the best of (content-vector cosine,
                # question-vector cosine). Operates on the normalised
                # ``d_sim`` so we stay in the same dynamic range the rest of
                # the pipeline expects.
                d_sim = _hype_boost_dsim(r, qvec, d_sim)
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

    # Phase 11 HyPE: lift each per-candidate raw dense similarity by the best
    # question-vector cosine when the row carries HyPE vectors. We apply the
    # boost BEFORE normalisation so a strong question match can override a
    # weak content match (otherwise the post-normalisation cap of 1.0 would
    # let multiple rows tie at the top). No-op when ``use_hype=False`` or
    # the row has no hype vectors.
    if use_hype:
        for cid, row in seen.items():
            if not _row_has_hype(row):
                continue
            d_sims_raw[cid] = _hype_boost_dsim(row, qvec, d_sims_raw.get(cid, 0.0))

    max_dsim = max(d_sims_raw.values()) if d_sims_raw else 1.0
    d_sims = {cid: v / max_dsim for cid, v in d_sims_raw.items()}

    max_sparse = max(sparse_score_by_id.values()) if sparse_score_by_id else 1.0
    s_norms = {cid: v / max_sparse for cid, v in sparse_score_by_id.items()}

    if fusion == "rrf":
        if use_hype:
            # Reorder the dense ranking by the boosted ``d_sims`` so HyPE
            # actually flips ranks at the RRF level (RRF is rank-only, so
            # the *score* lift doesn't help unless we surface it as a
            # position change). Rows missing from ``d_sims`` (shouldn't
            # happen, but defensively) get a score of 0.0.
            dense_ranking = sorted(
                (r["chunk_id"] for r in dense),
                key=lambda c: d_sims.get(c, 0.0),
                reverse=True,
            )
        else:
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
    parent_id = row.get("parent_chunk_id") or None
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
        parent_chunk_id=parent_id,
    )


def _dedupe_by_parent(
    candidates: list[tuple[str, float, float, float, dict[str, Any]]],
) -> list[tuple[str, float, float, float, dict[str, Any]]]:
    """Collapse children sharing the same parent to a single representative.

    Keeps the highest-scoring child in each parent group and rewrites its
    combined score to ``max(child_scores)``. Rows with no ``parent_chunk_id``
    pass through unchanged (FLAT KBs and standalone parents).

    Order preserved from the input.
    """
    by_parent: dict[str, int] = {}
    out: list[tuple[str, float, float, float, dict[str, Any]]] = []
    for cand in candidates:
        cid, combined, d, s, row = cand
        pid = row.get("parent_chunk_id") or None
        if not pid:
            out.append(cand)
            continue
        if pid in by_parent:
            idx = by_parent[pid]
            existing = out[idx]
            if combined > existing[1]:
                # Keep this child as the representative — better child rank.
                out[idx] = (cid, combined, d, s, row)
        else:
            by_parent[pid] = len(out)
            out.append(cand)
    return out


def _load_parent_text(tbl: Any, parent_chunk_id: str) -> tuple[str, int] | None:
    """Fetch a parent row's ``(text, byte_start)`` so we can build
    ``child_offset`` and replace the child's body with parent text.

    Returns None if the parent isn't in the table (legacy KB, broken link).
    """
    try:
        rows = (
            tbl.search()
            .where(f"chunk_id = '{parent_chunk_id}' AND is_parent = true")
            .limit(1)
            .to_list()
        )
    except Exception:
        # Some LanceDB versions don't accept SQL-style .where on .search();
        # fall back to a full-scan filter (KB is small, this stays O(N)).
        try:
            rows_all = tbl.to_arrow().to_pylist()
        except Exception:
            return None
        rows = [
            r
            for r in rows_all
            if r.get("chunk_id") == parent_chunk_id and r.get("is_parent")
        ][:1]
    if not rows:
        return None
    r = rows[0]
    return (r.get("text", "") or "", int(r.get("byte_start", 0) or 0))


def _expand_to_parent_in_hits(tbl: Any, hits: list[Hit]) -> None:
    """In-place: replace each child hit's ``text`` with parent text and set
    ``child_offset`` to the byte index of the child inside the parent.

    Falls back to leaving the hit unchanged when the parent can't be looked
    up. Hits that are already parent rows (``parent_chunk_id is None``) are
    skipped.
    """
    # Cache parent fetches across hits — multiple children may share a parent
    # even after dedupe (rerank may bring them back in).
    cache: dict[str, tuple[str, int] | None] = {}
    for h in hits:
        pid = h.parent_chunk_id
        if not pid:
            continue
        if pid not in cache:
            cache[pid] = _load_parent_text(tbl, pid)
        parent_info = cache[pid]
        if parent_info is None:
            continue
        ptext, _pbyte_start = parent_info
        # We don't store the child's body-relative byte_start on Hit; pull
        # it from the row dict via a fresh lookup. Use a defensive
        # text-search fallback when byte ranges aren't reliable.
        # The child's text is a substring of the parent's by construction.
        child_text = h.text
        offset = ptext.find(child_text) if child_text else 0
        if offset < 0:
            offset = 0
        h.text = ptext
        h.child_offset = offset
        h.expanded_to_parent = True


def _table_has_hype(tbl: Any) -> bool:
    """True iff the table schema declares HyPE columns."""
    try:
        names = set(tbl.schema.names)
    except Exception:
        return False
    return "hype_vectors" in names and "hype_questions" in names


def hybrid_query(
    kb_dir: Path,
    query_text: str,
    *,
    k: int = 5,
    fusion: FusionStrategy = "rrf",
    rerank: bool = False,
    top_k_stage1: int = 50,
    alpha: float | None = None,
    model: str | None = None,
    filter_authority: str | None = None,
    expand_to_parent: bool = True,
    dedupe_by_parent: bool = True,
    use_hype: bool | None = None,
    multi_query: bool = False,
) -> list[Hit]:
    """Hybrid dense + sparse retrieval with optional cross-encoder rerank.

    Pipeline:
      1. **Stage-1** retrieves a pool of ``top_k_stage1`` candidates fused by
         ``fusion`` (``"rrf"`` — default — or ``"alpha"`` with explicit
         ``alpha`` weight).
      2. **Stage-2** (optional, **OFF by default** post-EXP-004c — see
         F-007 amendment) re-scores stage-1 with a cross-encoder reranker
         and returns the top ``k``. Set ``rerank=True`` to opt in: +5pp
         recall@5 lift on bash KB, ~700ms additional latency per call.

    When ``rerank=False`` (the default) or ``LAB_RAG_RERANKER=none`` the
    reranker is bypassed and the top ``k`` come straight from stage-1.
    Phase 10 hooks (skip heuristics + cosine dedupe) live in this same
    call site.

    Phase 9 parent-child:
      * ``dedupe_by_parent`` collapses multiple child hits sharing the same
        parent down to one representative (max-of-children score). On a
        FLAT v1 KB no rows carry ``parent_chunk_id`` so the call is a no-op.
      * ``expand_to_parent`` rewrites each child hit's ``text`` with the
        parent's text and exposes ``child_offset`` on the :class:`Hit` so the
        model sees a wider context without an extra round-trip. No-op for
        FLAT KBs and for chunks that are already parents.

    Back-compat: callers that pass ``alpha=`` and not ``fusion=`` get the
    legacy alpha-blend (no surprise behaviour change for ablations or older
    scripts). Pass ``fusion="rrf"`` explicitly to force RRF even with an alpha
    value supplied for telemetry.

    Phase 11 HyPE:
      * ``use_hype`` controls whether each candidate's dense similarity is
        lifted to ``max(content_cos, max(question_cos))`` using the row's
        stored hypothetical-question vectors. ``None`` (the default) means
        "auto" — turn it on when the KB schema declares HyPE columns.
        Pass ``False`` to force-disable (ablation), ``True`` to force-enable
        (will be a no-op on a v1/v2 KB with empty hype_vectors).

    Phase 12 multi-query:
      * ``multi_query`` expands the query into N alternate phrasings via a
        local LLM, runs ``hybrid_query`` once per phrasing, and RRF-fuses
        the result lists. Default OFF — adds one LLM call + N extra
        retrievals per top-level question. Inner calls always run with
        ``multi_query=False`` to prevent recursive expansion.

    Returns ``[]`` (without calling the embedding model) if the KB has no
    index yet — so callers can smoke-test against in-progress KBs without
    burning GPU time.
    """
    from lab.rag import DEFAULT_EMBED_MODEL

    model = model or DEFAULT_EMBED_MODEL

    # Phase 12: when ``multi_query`` is set, fan out to alternate phrasings
    # and RRF-fuse the result lists. The inner call(s) always run with
    # ``multi_query=False`` so we can't recurse.
    if multi_query:
        return _hybrid_query_multi(
            kb_dir=kb_dir,
            query_text=query_text,
            k=k,
            fusion=fusion,
            rerank=rerank,
            top_k_stage1=top_k_stage1,
            alpha=alpha,
            model=model,
            filter_authority=filter_authority,
            expand_to_parent=expand_to_parent,
            dedupe_by_parent=dedupe_by_parent,
            use_hype=use_hype,
        )

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

    # Phase 11: resolve use_hype. ``None`` (default) → auto-on iff the table
    # carries HyPE columns. Explicit True/False overrides the auto-detect.
    effective_use_hype = (
        _table_has_hype(tbl) if use_hype is None else bool(use_hype)
    )

    stage1 = _stage1_candidates(
        tbl=tbl,
        query_text=query_text,
        qvec=qvec,
        pool_size=pool,
        fusion=fusion,
        alpha=alpha,
        filter_authority=filter_authority,
        use_hype=effective_use_hype,
    )
    if not stage1:
        return []

    # Phase 9 parent-dedupe: collapse children of the same parent before the
    # reranker sees them. On FLAT KBs the rows have no parent_chunk_id, so
    # _dedupe_by_parent is a no-op.
    if dedupe_by_parent:
        stage1 = _dedupe_by_parent(stage1)

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
            # Resolve kb_version from the manifest for cache namespacing.
            cache_key = _resolve_cache_key(kb_dir, top_k=k)
            reranked = reranker.rerank(
                query_text, deduped, top_n=k, cache_key=cache_key
            )
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

    # Phase 9 parent expansion: rewrite child hits with parent text. No-op
    # for FLAT KBs (no child carries a parent_chunk_id) and for hits that
    # are already parent rows.
    if expand_to_parent and any(h.parent_chunk_id for h in hits):
        _expand_to_parent_in_hits(tbl, hits)
    return hits


def _hybrid_query_multi(
    *,
    kb_dir: Path,
    query_text: str,
    k: int,
    fusion: FusionStrategy,
    rerank: bool,
    top_k_stage1: int,
    alpha: float | None,
    model: str | None,
    filter_authority: str | None,
    expand_to_parent: bool,
    dedupe_by_parent: bool,
    use_hype: bool | None,
) -> list[Hit]:
    """Phase 12: multi-query expansion with RRF fusion across phrasings.

    The original ``query_text`` is always retained as the first phrasing —
    if LLM expansion fails we degrade to a single ``hybrid_query`` call
    (no extra cost vs. ``multi_query=False``).
    """
    from lab.rag.expand import multi_query as _expand_multi_query

    phrasings = _expand_multi_query(query_text)
    if not phrasings:
        phrasings = [query_text]

    per_query_hits: list[list[Hit]] = []
    for phrasing in phrasings:
        hits = hybrid_query(
            kb_dir,
            phrasing,
            k=k,
            fusion=fusion,
            rerank=rerank,
            top_k_stage1=top_k_stage1,
            alpha=alpha,
            model=model,
            filter_authority=filter_authority,
            expand_to_parent=expand_to_parent,
            dedupe_by_parent=dedupe_by_parent,
            use_hype=use_hype,
            multi_query=False,  # guard against recursion
        )
        if hits:
            per_query_hits.append(hits)
    if not per_query_hits:
        return []

    # RRF-fuse the hit lists by chunk_id. We keep the best :class:`Hit`
    # object seen for each chunk_id (highest fused score becomes ``score``).
    fused_scores: dict[str, float] = {}
    best_hit: dict[str, Hit] = {}
    for hits in per_query_hits:
        for rank, h in enumerate(hits, start=1):
            fused_scores[h.chunk_id] = fused_scores.get(h.chunk_id, 0.0) + 1.0 / (
                RRF_K + rank
            )
            if h.chunk_id not in best_hit:
                best_hit[h.chunk_id] = h
    ordered = sorted(
        fused_scores.items(), key=lambda kv: kv[1], reverse=True
    )
    out: list[Hit] = []
    for cid, fused_score in ordered[:k]:
        h = best_hit[cid]
        # Rewrite ``score`` to the fused score so downstream sorts behave.
        h.score = fused_score
        out.append(h)
    return out


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
