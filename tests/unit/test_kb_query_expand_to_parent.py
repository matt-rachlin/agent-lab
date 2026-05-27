"""Phase 9 — expand_to_parent / dedupe_by_parent pass-through on kb_query.

We stub :func:`lab.rag.index.hybrid_query` so we never touch Ollama / LanceDB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from lab.agent.tools import kb_query as kb_query_mod
from lab.rag.index import Hit


def _stage(tmp_path: Path, name: str) -> Path:
    kb_dir = tmp_path / name
    kb_dir.mkdir()
    (kb_dir / "manifest.yaml").write_text("name: t\nslug: t\n", encoding="utf-8")
    return kb_dir


def test_expand_to_parent_default_true_passes_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stage(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    monkeypatch.setattr("lab.rag.index.count_rows", lambda _d: 5)
    captured: dict[str, Any] = {}

    def fake(*a: Any, **kw: Any) -> list[Hit]:
        captured.update(kw)
        return []

    monkeypatch.setattr("lab.rag.index.hybrid_query", fake)
    kb_query_mod.kb_query(kb_name="bash", question="hi")
    assert captured["expand_to_parent"] is True
    assert captured["dedupe_by_parent"] is True


def test_expand_to_parent_can_be_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    monkeypatch.setattr("lab.rag.index.count_rows", lambda _d: 5)
    captured: dict[str, Any] = {}

    def fake(*a: Any, **kw: Any) -> list[Hit]:
        captured.update(kw)
        return []

    monkeypatch.setattr("lab.rag.index.hybrid_query", fake)
    kb_query_mod.kb_query(
        kb_name="bash",
        question="hi",
        expand_to_parent=False,
        dedupe_by_parent=False,
    )
    assert captured["expand_to_parent"] is False
    assert captured["dedupe_by_parent"] is False


def test_kb_query_emits_parent_child_fields_in_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hits returned to the model include parent_chunk_id / child_offset /
    expanded_to_parent (even when None / False — keeps the schema flat)."""
    _stage(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    monkeypatch.setattr("lab.rag.index.count_rows", lambda _d: 5)

    parent_hit = Hit(
        chunk_id="child-1",
        text="full parent text including child",
        title="",
        summary="",
        source_url="",
        retrieved_at="",
        section_path=["A"],
        score=0.9,
        dense_score=0.8,
        sparse_score=0.1,
        authority="official",
        parent_chunk_id="parent-1",
        child_offset=22,
        expanded_to_parent=True,
    )
    flat_hit = Hit(
        chunk_id="flat-1",
        text="flat chunk text",
        title="",
        summary="",
        source_url="",
        retrieved_at="",
        section_path=[],
        score=0.7,
        dense_score=0.5,
        sparse_score=0.2,
        authority="official",
    )

    monkeypatch.setattr(
        "lab.rag.index.hybrid_query",
        lambda *a, **kw: [parent_hit, flat_hit],
    )
    out = kb_query_mod.kb_query(kb_name="bash", question="hi")
    assert len(out["hits"]) == 2
    h0, h1 = out["hits"]
    assert h0["parent_chunk_id"] == "parent-1"
    assert h0["child_offset"] == 22
    assert h0["expanded_to_parent"] is True
    # FLAT hit keeps None / False fields — never crashes the model side.
    assert h1["parent_chunk_id"] is None
    assert h1["child_offset"] is None
    assert h1["expanded_to_parent"] is False
