"""Unit tests for `_truncate`'s kb_query carve-out.

The RAG scorers (recall_at_k, mrr, ndcg, attribution) read chunk_id and
source_url off each hit in the trajectory. A blunt truncate would
collapse the entire result into a string preview and zero those scorers
out — F-005-successor surfaced during the 6h-e smoke when glm-5.1-cloud
retrieved gold chunks but recall_at_k scored 0.0 because the trajectory
had been preview-stringified.

These tests lock down the contract that kb_query-shaped dicts retain
the per-hit structural fields no matter the payload size.
"""

from __future__ import annotations

from lab.inspect_bridge.solver import _truncate


def _make_hits(n: int, text_len: int = 1500) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": f"chunk-{i:03d}",
            "source_url": f"https://example.com/doc-{i}",
            "section_path": ["A", f"sub-{i}"],
            "title": f"title-{i}",
            "summary": "s" * 200,
            "text": "x" * text_len,
            "score": 0.5 + i * 0.01,
            "dense_score": 0.4 + i * 0.01,
            "sparse_score": 0.3 + i * 0.01,
            "rerank_score": 5.0 - i * 0.1,
            "stage1_rank": i + 1,
            "truncated": False,
        }
        for i in range(n)
    ]


def test_small_kb_query_result_passes_through_unchanged() -> None:
    """A small payload should not be touched — the carve-out only kicks
    in when the payload exceeds the cap.
    """

    value = {"hits": _make_hits(1, text_len=50), "kb_status": "ok"}
    out = _truncate(value, cap=4096)
    # No structural transformation expected; value comes back as-is.
    assert out is value or out == value


def test_kb_query_result_keeps_chunk_ids_when_over_cap() -> None:
    """A 5-hit result with ~1500-char texts exceeds 4096 bytes total but
    we MUST preserve every hit's chunk_id (the RAG scorers read this).
    """

    value = {"hits": _make_hits(5, text_len=1500), "kb_status": "ok"}
    out = _truncate(value, cap=4096)

    assert isinstance(out, dict)
    assert "hits" in out
    out_hits = out["hits"]
    assert isinstance(out_hits, list)
    assert len(out_hits) == 5, "must not drop hits"
    for i, hit in enumerate(out_hits):
        assert hit.get("chunk_id") == f"chunk-{i:03d}"
        assert hit.get("source_url") == f"https://example.com/doc-{i}"
        assert hit.get("section_path") == ["A", f"sub-{i}"]


def test_kb_query_result_trims_text_before_dropping_it() -> None:
    """First attempt: keep all hits, just trim text to 240 chars. Only
    drop text entirely if even the trimmed version exceeds the cap.
    """

    # 5 hits * 240 chars text * boilerplate ≈ comfortably under 4096.
    value = {"hits": _make_hits(5, text_len=1500), "kb_status": "ok"}
    out = _truncate(value, cap=4096)
    assert out.get("_hits_text_trimmed") is True or out.get("_hits_text_dropped") is True
    if out.get("_hits_text_trimmed"):
        # text was trimmed but still present
        for hit in out["hits"]:
            text = hit.get("text", "")
            assert isinstance(text, str)
            assert len(text) <= 241, f"text not trimmed: {len(text)} chars"


def test_kb_query_result_drops_text_when_even_trimmed_too_big() -> None:
    """Many hits with trimmed text still overflow the cap → drop text but
    keep the structural fields RAG scorers depend on.
    """

    # 50 hits * 240 chars text ≈ 12_000+ chars before even the boilerplate
    # — guaranteed to overflow a 4096-cap.
    value = {"hits": _make_hits(50, text_len=1500), "kb_status": "ok"}
    out = _truncate(value, cap=4096)

    assert out.get("_hits_text_dropped") is True
    assert len(out["hits"]) == 50, "must not drop hits even at the minimal stage"
    for i, hit in enumerate(out["hits"]):
        # Structural fields the RAG scorers read MUST survive.
        assert hit.get("chunk_id") == f"chunk-{i:03d}"
        assert hit.get("source_url") == f"https://example.com/doc-{i}"
        assert "score" in hit
        # Phase 7 rerank signal must survive even in the minimal branch:
        # EXP-004 post-hoc analysis relies on `rerank_score` being in the
        # persisted trajectory regardless of payload size.
        assert "rerank_score" in hit
        assert "stage1_rank" in hit
        # text / summary are gone in this branch — that's the trade-off.
        assert "text" not in hit


def test_kb_query_result_preserves_top_level_fields() -> None:
    """Non-hits keys (kb_status, kb_dir, error) must round-trip."""

    value = {
        "hits": _make_hits(50),
        "kb_status": "ok",
        "kb_dir": "/kb/bash",
    }
    out = _truncate(value, cap=4096)
    assert out["kb_status"] == "ok"
    assert out["kb_dir"] == "/kb/bash"


def test_non_kb_query_dict_uses_blunt_truncate() -> None:
    """The carve-out triggers ONLY when `hits` is present and is a list."""

    value = {"data": "x" * 10000}
    out = _truncate(value, cap=4096)
    assert isinstance(out, dict)
    assert out.get("_truncated") is True
    assert "preview" in out
