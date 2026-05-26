"""Unit tests for `lab.agent.tools.kb_query` — validation + clean failure paths.

These run on the host without the sandbox. We monkeypatch `count_rows` and
`hybrid_query` so we never reach the embedder (no Ollama, no GPU). The path
in the tool that does the real work is exercised end-to-end by the
integration test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lab.agent.tools import kb_query as kb_query_mod


def _stage_manifest(tmp_path: Path, name: str) -> Path:
    kb_dir = tmp_path / name
    kb_dir.mkdir()
    (kb_dir / "manifest.yaml").write_text("name: test\nslug: test\n", encoding="utf-8")
    return kb_dir


def test_kb_query_rejects_invalid_kb_name() -> None:
    out = kb_query_mod.kb_query(kb_name="../etc", question="hi")
    assert out["hits"] == []
    assert "invalid kb_name" in out["error"]


def test_kb_query_rejects_kb_name_with_slash() -> None:
    out = kb_query_mod.kb_query(kb_name="a/b", question="hi")
    assert out["hits"] == []
    assert "invalid kb_name" in out["error"]


def test_kb_query_rejects_empty_question() -> None:
    out = kb_query_mod.kb_query(kb_name="bash", question="   ")
    assert out["hits"] == []
    assert "non-empty" in out["error"]


def test_kb_query_rejects_alpha_out_of_range() -> None:
    out = kb_query_mod.kb_query(kb_name="bash", question="x", alpha=2.0)
    assert out["hits"] == []
    assert "alpha" in out["error"]


def test_kb_query_clamps_k_to_safe_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kb_dir = _stage_manifest(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))

    captured: dict[str, Any] = {}

    def fake_count(_dir: Path) -> int:
        return 3

    def fake_hybrid(
        kb_dir_in: Path,
        question: str,
        **kwargs: Any,
    ) -> list[Any]:
        captured["k"] = kwargs.get("k")
        return []

    monkeypatch.setattr("lab.rag.index.count_rows", fake_count)
    monkeypatch.setattr("lab.rag.index.hybrid_query", fake_hybrid)
    out = kb_query_mod.kb_query(kb_name="bash", question="hi", k=10_000)
    assert out["hits"] == []
    assert captured["k"] == 50  # clamped
    assert kb_dir.exists()


def test_kb_query_missing_kb_returns_clean_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    out = kb_query_mod.kb_query(kb_name="nope", question="hi")
    assert out["hits"] == []
    assert out["kb_status"] == "missing"


def test_kb_query_empty_kb_returns_clean_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stage_manifest(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    monkeypatch.setattr("lab.rag.index.count_rows", lambda _d: 0)

    out = kb_query_mod.kb_query(kb_name="bash", question="hi")
    assert out["hits"] == []
    assert out["kb_status"] == "empty"


def test_kb_query_truncates_long_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lab.rag.index import Hit

    _stage_manifest(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    monkeypatch.setattr("lab.rag.index.count_rows", lambda _d: 1)
    long_text = "x" * (kb_query_mod.MAX_TEXT_CHARS + 500)

    def fake_hybrid(*a: Any, **kw: Any) -> list[Hit]:
        return [
            Hit(
                chunk_id="c1",
                text=long_text,
                title="t",
                summary="s",
                source_url="https://example.com",
                retrieved_at="2026-01-01T00:00:00Z",
                section_path=["a", "b"],
                score=0.9,
                dense_score=0.8,
                sparse_score=0.7,
                authority="official",
            )
        ]

    monkeypatch.setattr("lab.rag.index.hybrid_query", fake_hybrid)
    out = kb_query_mod.kb_query(kb_name="bash", question="hi", k=1)
    assert len(out["hits"]) == 1
    hit = out["hits"][0]
    assert hit["truncated"] is True
    assert len(hit["text"]) == kb_query_mod.MAX_TEXT_CHARS
    assert hit["chunk_id"] == "c1"
    assert hit["section_path"] == ["a", "b"]
    assert hit["authority"] == "official"


def test_kb_query_short_text_not_truncated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lab.rag.index import Hit

    _stage_manifest(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    monkeypatch.setattr("lab.rag.index.count_rows", lambda _d: 1)

    def fake_hybrid(*a: Any, **kw: Any) -> list[Hit]:
        return [
            Hit(
                chunk_id="c2",
                text="short",
                title="",
                summary="",
                source_url="",
                retrieved_at="",
                section_path=[],
                score=0.5,
                dense_score=0.5,
                sparse_score=0.0,
                authority="",
            )
        ]

    monkeypatch.setattr("lab.rag.index.hybrid_query", fake_hybrid)
    out = kb_query_mod.kb_query(kb_name="bash", question="hi")
    assert out["kb_status"] == "ok"
    assert out["hits"][0]["truncated"] is False
    assert out["hits"][0]["text"] == "short"


def test_kb_query_hybrid_error_returns_clean_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stage_manifest(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    monkeypatch.setattr("lab.rag.index.count_rows", lambda _d: 5)

    def boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("ollama down")

    monkeypatch.setattr("lab.rag.index.hybrid_query", boom)
    out = kb_query_mod.kb_query(kb_name="bash", question="hi")
    assert out["hits"] == []
    assert "ollama down" in out["error"]


def test_kb_query_passes_authority_filter_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stage_manifest(tmp_path, "bash")
    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))
    monkeypatch.setattr("lab.rag.index.count_rows", lambda _d: 5)

    captured: dict[str, Any] = {}

    def fake_hybrid(
        kb_dir_in: Path,
        question: str,
        **kwargs: Any,
    ) -> list[Any]:
        captured["authority"] = kwargs.get("filter_authority")
        captured["alpha"] = kwargs.get("alpha")
        captured["fusion"] = kwargs.get("fusion")
        captured["rerank"] = kwargs.get("rerank")
        return []

    monkeypatch.setattr("lab.rag.index.hybrid_query", fake_hybrid)
    out = kb_query_mod.kb_query(
        kb_name="bash", question="x", authority="official", alpha=0.3
    )
    assert out["hits"] == []
    assert captured["authority"] == "official"
    assert captured["alpha"] == pytest.approx(0.3)
    # Default fusion stays "rrf" — but when alpha is passed, hybrid_query
    # itself flips to alpha-blend on the inside. The MCP tool surfaces both:
    # caller passes fusion explicitly to control it.
    assert captured["fusion"] == "rrf"
    # Default rerank flipped to False post-EXP-004c (see F-007 amendment).
    assert captured["rerank"] is False


def test_kb_query_permission_denied_on_manifest_stat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the sandbox can't stat the manifest, the tool returns a clean
    error string instead of crashing — rootless podman uid-mapping is a
    realistic source of PermissionError on `~/db/kb/`.
    """

    monkeypatch.setenv("LAB_KB_ROOT", str(tmp_path))

    real_is_file = Path.is_file

    def is_file_with_perm_error(self: Path, *args: Any, **kw: Any) -> bool:
        if self.name == "manifest.yaml":
            raise PermissionError(13, "Permission denied", str(self))
        return real_is_file(self, *args, **kw)

    monkeypatch.setattr(Path, "is_file", is_file_with_perm_error)
    out = kb_query_mod.kb_query(kb_name="bash", question="hi")
    assert out["hits"] == []
    assert "permission denied" in out["error"].lower()


def test_task_needs_kb_mount_heuristic() -> None:
    from lab.agent.tools import task_needs_kb_mount

    assert task_needs_kb_mount(None) is False
    assert task_needs_kb_mount([]) is False
    assert task_needs_kb_mount([{"name": "fs_read"}]) is False
    assert task_needs_kb_mount([{"name": "kb_query"}]) is True
    assert task_needs_kb_mount([{"name": "fs_read"}, {"name": "kb_query"}]) is True
    # Tolerate bare-string specs in case a caller uses that shape.
    assert task_needs_kb_mount(["kb_query"]) is True
    # Garbage entries are ignored, not crashed on.
    assert task_needs_kb_mount([None, {"foo": "bar"}]) is False
