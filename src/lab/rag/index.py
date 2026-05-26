"""LanceDB writer + hybrid retrieval router.

Vendored from kb_builder.index.

Layout per KB: <kb_dir>/index/chunks.lance/  (a LanceDB dataset directory).
We store dense vectors and a JSON-encoded sparse vector. Hybrid query
combines a vector search with a BM25 score computed over the chunks'
sparse vectors at query time (in-process; corpus is small enough to scan).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from lab.rag.embedder import embed_texts, tokenize_for_bm25

TABLE_NAME = "chunks"


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


def hybrid_query(
    kb_dir: Path,
    query_text: str,
    *,
    k: int = 5,
    alpha: float = 0.5,
    model: str | None = None,
    filter_authority: str | None = None,
) -> list[Hit]:
    """Hybrid dense + sparse retrieval.

    alpha=1.0 -> pure dense; alpha=0.0 -> pure sparse.

    Returns [] (without calling the embedding model) if the KB has no index
    yet — so callers can smoke-test against in-progress KBs without burning
    GPU time.
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
    if tbl.count_rows() == 0:
        return []

    # Embed query
    qvec = embed_texts([query_text], model=model, batch_size=1).vectors[0]

    # Dense candidates (over-fetch so we have plenty to merge with sparse)
    pool = max(k * 8, 40)
    dense = _dense_search(tbl, qvec, pool)
    if filter_authority:
        dense = [r for r in dense if r.get("authority") == filter_authority]

    # Sparse over the same dense pool (cheap) + a broader scan over all rows
    if alpha < 1.0:
        all_rows = tbl.to_arrow().to_pylist()
        if filter_authority:
            all_rows = [r for r in all_rows if r.get("authority") == filter_authority]
        sparse_top = _bm25_scan(all_rows, query_text, pool)
        # collect candidate set: union of dense + sparse-top by chunk_id
        seen: dict[str, dict[str, Any]] = {}
        for r in dense:
            seen[r["chunk_id"]] = r
        sparse_score_by_id: dict[str, float] = {}
        for idx, s in sparse_top:
            r = all_rows[idx]
            seen.setdefault(r["chunk_id"], r)
            sparse_score_by_id[r["chunk_id"]] = s

        # Build dense score map
        dense_distances = {r["chunk_id"]: float(r.get("_distance", 1.0)) for r in dense}
        # Convert L2 distance to similarity (1 / (1 + d)), normalized within candidates
        d_sims = {cid: 1.0 / (1.0 + d) for cid, d in dense_distances.items()}
        max_dsim = max(d_sims.values()) if d_sims else 1.0
        d_sims = {cid: v / max_dsim for cid, v in d_sims.items()}

        max_sparse = max(sparse_score_by_id.values()) if sparse_score_by_id else 1.0
        s_norms = {cid: v / max_sparse for cid, v in sparse_score_by_id.items()}

        scored: list[tuple[str, float, float, float, dict[str, Any]]] = []
        for cid, row in seen.items():
            d_score = d_sims.get(cid, 0.0)
            s_score = s_norms.get(cid, 0.0)
            combined = alpha * d_score + (1.0 - alpha) * s_score
            scored.append((cid, combined, d_score, s_score, row))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:k]
    else:
        scored = []
        max_d = max((1.0 / (1.0 + float(r.get("_distance", 1.0))) for r in dense), default=1.0)
        for r in dense[:k]:
            d_sim = (1.0 / (1.0 + float(r.get("_distance", 1.0)))) / max_d
            scored.append((r["chunk_id"], d_sim, d_sim, 0.0, r))

    hits: list[Hit] = []
    for cid, combined, d_score, s_score, row in scored:
        hits.append(
            Hit(
                chunk_id=cid,
                text=row.get("text", ""),
                title=row.get("title", "") or "",
                summary=row.get("summary", "") or "",
                source_url=row.get("source_url", "") or "",
                retrieved_at=row.get("retrieved_at", "") or "",
                section_path=list(row.get("section_path") or []),
                score=combined,
                dense_score=d_score,
                sparse_score=s_score,
                authority=row.get("authority", "") or "",
            )
        )
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
