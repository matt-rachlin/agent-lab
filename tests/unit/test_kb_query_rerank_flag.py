"""Unit tests that the new ``rerank`` + ``fusion`` flags on the kb_query MCP
tool pass through to :func:`lab.rag.index.hybrid_query` unchanged.

We stub ``hybrid_query`` so we never touch Ollama / LanceDB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lab.agent.tools import kb_query as kb_query_mod


def _stage(tmp_path: Path, name: str) -> Path:
    kb_dir = tmp_path / name
    kb_dir.mkdir()
    (kb_dir / "manifest.yaml").write_text("name: t\nslug: t\n", encoding="utf-8")
    return kb_dir


def test_rerank_default_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default reverted to False post-EXP-004c (see F-007 amendment)."""

    _stage(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    monkeypatch.setattr("lab.rag.index.count_rows", lambda _d: 5)
    captured: dict[str, Any] = {}

    def fake(*a: Any, **kw: Any) -> list[Any]:
        captured.update(kw)
        return []

    monkeypatch.setattr("lab.rag.index.hybrid_query", fake)
    out = kb_query_mod.kb_query(kb_name="bash", question="hello")
    assert out["hits"] == []
    assert captured["rerank"] is False
    assert captured["fusion"] == "rrf"
    assert captured["alpha"] is None


def test_rerank_can_be_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Callers can opt into rerank explicitly."""

    _stage(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    monkeypatch.setattr("lab.rag.index.count_rows", lambda _d: 5)
    captured: dict[str, Any] = {}

    def fake(*a: Any, **kw: Any) -> list[Any]:
        captured.update(kw)
        return []

    monkeypatch.setattr("lab.rag.index.hybrid_query", fake)
    kb_query_mod.kb_query(kb_name="bash", question="hi", rerank=True)
    assert captured["rerank"] is True


def test_invalid_fusion_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    out = kb_query_mod.kb_query(kb_name="bash", question="hi", fusion="bogus")
    assert out["hits"] == []
    assert "fusion" in out["error"]
