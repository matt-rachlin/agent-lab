"""Phase 12 — hybrid_query ``multi_query=True`` recursion-guard + RRF fuse.

These tests stay surgical: we mock :mod:`lab.rag.expand.multi_query` and the
inner :func:`hybrid_query` call so the test runner doesn't touch Ollama or
LanceDB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import lab.rag.index as idx_module
from lab.rag.index import Hit, _hybrid_query_multi, hybrid_query


def _hit(cid: str, score: float) -> Hit:
    return Hit(
        chunk_id=cid,
        text=f"text {cid}",
        title="",
        summary="",
        source_url="",
        retrieved_at="",
        section_path=[],
        score=score,
        dense_score=score,
        sparse_score=0.0,
        authority="official",
    )


def test_multi_query_does_not_recurse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Inner call must run with multi_query=False — otherwise infinite loop."""

    inner_calls: list[dict[str, Any]] = []

    def fake_hybrid_query(*args: Any, **kwargs: Any) -> list[Hit]:
        inner_calls.append({"args": args, "kwargs": kwargs})
        return [_hit("c0", 0.9)]

    def fake_multi(question: str, **_kw: Any) -> list[str]:
        return [question, question + " v2", question + " v3"]

    # Patch the expand entry point used inside _hybrid_query_multi.
    monkeypatch.setattr("lab.rag.expand.multi_query", fake_multi)

    # Capture inner hybrid_query calls. We patch the module-level binding the
    # multi-query helper uses (calls hybrid_query() unqualified).
    monkeypatch.setattr(idx_module, "hybrid_query", fake_hybrid_query)

    out = _hybrid_query_multi(
        kb_dir=tmp_path,
        query_text="how do i redirect stderr",
        k=3,
        fusion="rrf",
        rerank=False,
        top_k_stage1=50,
        alpha=None,
        model=None,
        filter_authority=None,
        expand_to_parent=True,
        dedupe_by_parent=True,
        use_hype=None,
    )
    assert out
    # All 3 phrasings hit the inner function.
    assert len(inner_calls) == 3
    # And every inner call ran with multi_query=False.
    for call in inner_calls:
        assert call["kwargs"].get("multi_query") is False


def test_multi_query_rrf_fuses_per_phrasing_hits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A chunk_id that appears in many phrasings' top-3 wins the RRF race."""

    def fake_multi(question: str, **_kw: Any) -> list[str]:
        return [question, "alt1", "alt2"]

    # First phrasing ranks: c0, c1, c2.
    # Second phrasing ranks: c1, c2, c3.
    # Third phrasing ranks: c1, c0, c4.
    # c1 shows up in every list (rank 2, 1, 1) → highest RRF score.
    per_phrasing = [
        [_hit("c0", 0.9), _hit("c1", 0.8), _hit("c2", 0.7)],
        [_hit("c1", 0.9), _hit("c2", 0.8), _hit("c3", 0.7)],
        [_hit("c1", 0.9), _hit("c0", 0.8), _hit("c4", 0.7)],
    ]

    call_idx = {"i": 0}

    def fake_hybrid_query(*_args: Any, **_kwargs: Any) -> list[Hit]:
        i = call_idx["i"]
        call_idx["i"] += 1
        return per_phrasing[i]

    monkeypatch.setattr("lab.rag.expand.multi_query", fake_multi)
    monkeypatch.setattr(idx_module, "hybrid_query", fake_hybrid_query)

    out = _hybrid_query_multi(
        kb_dir=tmp_path,
        query_text="q",
        k=3,
        fusion="rrf",
        rerank=False,
        top_k_stage1=50,
        alpha=None,
        model=None,
        filter_authority=None,
        expand_to_parent=True,
        dedupe_by_parent=True,
        use_hype=None,
    )
    cids = [h.chunk_id for h in out]
    assert cids[0] == "c1"


def test_multi_query_empty_phrasings_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If expansion yields nothing, we still run a single inner call with the
    original question text."""

    inner_calls: list[str] = []

    def fake_hybrid_query(_kb: Path, question: str, **_kw: Any) -> list[Hit]:
        inner_calls.append(question)
        return [_hit("c0", 0.9)]

    monkeypatch.setattr("lab.rag.expand.multi_query", lambda _q, **_kw: [])
    monkeypatch.setattr(idx_module, "hybrid_query", fake_hybrid_query)

    out = _hybrid_query_multi(
        kb_dir=tmp_path,
        query_text="ORIGINAL",
        k=3,
        fusion="rrf",
        rerank=False,
        top_k_stage1=50,
        alpha=None,
        model=None,
        filter_authority=None,
        expand_to_parent=True,
        dedupe_by_parent=True,
        use_hype=None,
    )
    assert inner_calls == ["ORIGINAL"]
    assert [h.chunk_id for h in out] == ["c0"]


def test_hybrid_query_top_level_routes_to_multi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Top-level ``hybrid_query(multi_query=True)`` enters the multi path."""

    called = {"multi": False}

    def fake_multi(**_kw: Any) -> list[Hit]:
        called["multi"] = True
        return [_hit("ok", 1.0)]

    monkeypatch.setattr(idx_module, "_hybrid_query_multi", fake_multi)

    out = hybrid_query(tmp_path, "q", k=3, multi_query=True)
    assert called["multi"] is True
    assert [h.chunk_id for h in out] == ["ok"]
