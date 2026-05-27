"""Unit tests for lab.rag.index.

Covers: schema shape, count_rows/index_bytes on missing/empty dirs,
hybrid_query early-return when no index exists, and _bm25_scan weighting.
"""

from __future__ import annotations

from pathlib import Path

from lab.rag.index import (
    TABLE_NAME,
    _bm25_scan,
    _schema,
    count_rows,
    hybrid_query,
    index_bytes,
)


def test_schema_has_all_columns():
    schema = _schema(dims=128)
    names = set(schema.names)
    expected = {
        "chunk_id",
        "source_id",
        "source_url",
        "source_sha256",
        "retrieved_at",
        "doc_path",
        "section_path",
        "byte_start",
        "byte_end",
        "text",
        "title",
        "summary",
        "keywords",
        "prerequisites",
        "vector",
        "sparse_json",
        "tokens",
        "chunk_format_version",
        "authority",
        # Phase 9 v2 parent-child columns
        "parent_chunk_id",
        "child_index",
        "is_parent",
        # Phase 11 v3 HyPE columns
        "hype_questions",
        "hype_vectors",
    }
    assert names == expected
    # vector field is a fixed-size list of float32 with the requested width
    vec_field = schema.field("vector")
    # list_(value_type, list_size) — list_size is exposed on the type
    assert vec_field.type.list_size == 128


def test_count_rows_no_index(tmp_path: Path):
    # No index/ dir at all
    assert count_rows(tmp_path) == 0
    # Empty index/ dir (no LanceDB table)
    (tmp_path / "index").mkdir()
    assert count_rows(tmp_path) == 0


def test_index_bytes_no_index(tmp_path: Path):
    assert index_bytes(tmp_path) == 0


def test_index_bytes_counts_files(tmp_path: Path):
    idx = tmp_path / "index"
    idx.mkdir()
    (idx / "a").write_bytes(b"x" * 100)
    (idx / "sub").mkdir()
    (idx / "sub" / "b").write_bytes(b"y" * 50)
    assert index_bytes(tmp_path) == 150


def test_hybrid_query_empty_kb_returns_no_hits_without_embedding(tmp_path: Path):
    """If the KB has no index at all, hybrid_query must early-return [] without
    constructing an Ollama Client (we'd see an import-level side-effect otherwise)."""
    hits = hybrid_query(tmp_path, "any question")
    assert hits == []


def test_bm25_scan_overlap_scoring():
    rows = [
        {"chunk_id": "a", "sparse_json": '{"foo": 1.0, "bar": 2.0}'},
        {"chunk_id": "b", "sparse_json": '{"baz": 5.0}'},
        {"chunk_id": "c", "sparse_json": "{}"},
        {"chunk_id": "d", "sparse_json": '{"foo": 0.5}'},
    ]
    # Query 'foo bar' matches 'a' best, then 'd', and 'b'/'c' not at all.
    out = _bm25_scan(rows, "foo bar", k=10)
    ranked_ids = [rows[i]["chunk_id"] for i, _ in out]
    assert ranked_ids[0] == "a"
    assert "d" in ranked_ids
    assert "b" not in ranked_ids
    assert "c" not in ranked_ids


def test_bm25_scan_empty_query():
    rows = [{"chunk_id": "a", "sparse_json": '{"x": 1.0}'}]
    assert _bm25_scan(rows, "", k=5) == []


def test_bm25_scan_malformed_sparse_json_ignored():
    rows = [{"chunk_id": "a", "sparse_json": "not-json-here"}]
    # Should not raise; just yield no score for that row.
    assert _bm25_scan(rows, "anything", k=5) == []


def test_table_name_constant():
    # Stable string used to look up the dataset in LanceDB.
    assert TABLE_NAME == "chunks"
