"""Phase 9 — parent dedupe + schema-v2 columns on lab.rag.index.

We mock LanceDB / embedder calls; these are pure-Python unit tests.
"""

from __future__ import annotations

from typing import Any

from lab.rag.index import _dedupe_by_parent, _schema


def _cand(
    cid: str, score: float, parent: str | None = None
) -> tuple[str, float, float, float, dict[str, Any]]:
    row = {
        "chunk_id": cid,
        "text": cid,
        "section_path": [],
        "parent_chunk_id": parent,
        "is_parent": parent is None,
    }
    return (cid, score, score, 0.0, row)


def test_dedupe_by_parent_collapses_children() -> None:
    """Children sharing a parent collapse to a single representative whose
    score is max(child_scores)."""
    cands = [
        _cand("c1", 0.9, parent="p1"),
        _cand("c2", 0.7, parent="p1"),
        _cand("c3", 0.8, parent="p2"),
    ]
    out = _dedupe_by_parent(cands)
    # Two parents — one entry per parent.
    pids = {row.get("parent_chunk_id") for _, _, _, _, row in out}
    assert pids == {"p1", "p2"}
    # The kept p1 row is c1 (higher-scoring child), score preserved.
    p1_row = next(c for c in out if c[4]["parent_chunk_id"] == "p1")
    assert p1_row[0] == "c1"
    assert p1_row[1] == 0.9


def test_dedupe_by_parent_passes_through_flat_rows() -> None:
    """Rows with no parent_chunk_id (FLAT KB) are unchanged."""
    cands = [
        _cand("a", 0.5),
        _cand("b", 0.4),
        _cand("c", 0.3),
    ]
    out = _dedupe_by_parent(cands)
    assert [c[0] for c in out] == ["a", "b", "c"]


def test_dedupe_by_parent_keeps_higher_score_when_out_of_order() -> None:
    """If a later candidate has a higher score, it replaces the earlier one."""
    cands = [
        _cand("low", 0.2, parent="p"),
        _cand("high", 0.99, parent="p"),
    ]
    out = _dedupe_by_parent(cands)
    assert len(out) == 1
    assert out[0][0] == "high"
    assert out[0][1] == 0.99


def test_schema_has_v2_parent_child_columns() -> None:
    import pyarrow as pa

    schema = _schema(dims=32)
    names = set(schema.names)
    assert {"parent_chunk_id", "child_index", "is_parent"}.issubset(names)
    assert schema.field("is_parent").type == pa.bool_()
    assert schema.field("child_index").type == pa.int32()
    assert schema.field("parent_chunk_id").type == pa.string()
