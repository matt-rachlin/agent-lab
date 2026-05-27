"""Phase 12 — kb_query MCP tool passes multi_query through to hybrid_query."""

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


def test_multi_query_default_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stage(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    monkeypatch.setattr("lab.rag.index.count_rows", lambda _d: 5)
    captured: dict[str, Any] = {}

    def fake(*_a: Any, **kw: Any) -> list[Hit]:
        captured.update(kw)
        return []

    monkeypatch.setattr("lab.rag.index.hybrid_query", fake)
    kb_query_mod.kb_query(kb_name="bash", question="hi")
    assert captured["multi_query"] is False


def test_multi_query_flag_is_propagated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stage(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    monkeypatch.setattr("lab.rag.index.count_rows", lambda _d: 5)
    captured: dict[str, Any] = {}

    def fake(*_a: Any, **kw: Any) -> list[Hit]:
        captured.update(kw)
        return []

    monkeypatch.setattr("lab.rag.index.hybrid_query", fake)
    kb_query_mod.kb_query(kb_name="bash", question="hi", multi_query=True)
    assert captured["multi_query"] is True
