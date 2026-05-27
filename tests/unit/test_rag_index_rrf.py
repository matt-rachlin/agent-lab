"""Unit tests for RRF fusion math + hybrid_query stage-1 wiring.

We mock out the LanceDB table interactions so the tests stay CPU-only and
deterministic. The cross-encoder is bypassed via ``rerank=False`` (no
sentence-transformers import needed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lab.rag import RRF_K
from lab.rag.index import _rrf_fuse, _stage1_candidates, hybrid_query


def test_rrf_fuse_default_k_60() -> None:
    """RRF score for rank 1 in both lists == 2/(60+1) = 0.0327..."""
    dense = ["a", "b", "c"]
    sparse = ["a", "x", "y"]
    fused = _rrf_fuse(dense, sparse)
    # 'a' appears at rank 1 in both lists
    assert fused["a"] == pytest.approx(2.0 / (RRF_K + 1))
    # 'b' only in dense at rank 2
    assert fused["b"] == pytest.approx(1.0 / (RRF_K + 2))
    # 'x' only in sparse at rank 2
    assert fused["x"] == pytest.approx(1.0 / (RRF_K + 2))


def test_rrf_fuse_rewards_consistent_ranking() -> None:
    """A doc ranked top in both heads beats a doc only top in one."""
    fused = _rrf_fuse(["doc_a", "doc_b"], ["doc_a", "doc_c"])
    # doc_a wins because it appears in both.
    ordered = sorted(fused, key=lambda c: fused[c], reverse=True)
    assert ordered[0] == "doc_a"
    # b and c are tied (each only in one head at rank 2)
    assert fused["doc_b"] == fused["doc_c"]


def test_rrf_fuse_empty_inputs() -> None:
    assert _rrf_fuse([], []) == {}
    fused = _rrf_fuse(["only"], [])
    assert fused == {"only": 1.0 / (RRF_K + 1)}


def test_rrf_fuse_custom_k() -> None:
    """k_const=10 makes rank gaps weigh more."""
    fused_default = _rrf_fuse(["a", "b"], [])
    fused_k10 = _rrf_fuse(["a", "b"], [], k_const=10)
    # rank-1 score is higher with smaller k_const
    assert fused_k10["a"] > fused_default["a"]


def test_rrf_preserves_top_n_in_stage1(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_stage1_candidates`` with fusion='rrf' returns candidates ordered by RRF."""

    rows = [
        {
            "chunk_id": f"c{i}",
            "text": f"doc {i}",
            "section_path": [],
            "authority": "official",
            "sparse_json": json.dumps({"foo": 1.0}) if i in (1, 4) else "{}",
            "title": "",
            "summary": "",
            "source_url": "",
            "retrieved_at": "",
            "vector": [0.0] * 4,
        }
        for i in range(6)
    ]

    class FakeArrow:
        def to_pylist(self) -> list[dict[str, Any]]:
            return rows

    class FakeSearch:
        def __init__(self, results: list[dict[str, Any]]) -> None:
            self._results = results

        def limit(self, n: int) -> FakeSearch:
            self._results = self._results[:n]
            return self

        def to_list(self) -> list[dict[str, Any]]:
            return self._results

    class FakeTable:
        def __init__(self, dense_order: list[int]) -> None:
            self.dense_order = dense_order

        def search(self, _vec: list[float]) -> FakeSearch:
            dense_results: list[dict[str, Any]] = []
            for distance, idx in enumerate(self.dense_order):
                r = dict(rows[idx])
                r["_distance"] = float(distance)
                dense_results.append(r)
            return FakeSearch(dense_results)

        def to_arrow(self) -> FakeArrow:
            return FakeArrow()

    # Dense ranks c2 first; sparse ('foo') only matches c1 and c4. RRF should
    # blend so c1 and c4 climb above c0,c3,c5 even though dense ranked them
    # mid-pack.
    tbl = FakeTable(dense_order=[2, 0, 1, 3, 4, 5])

    scored = _stage1_candidates(
        tbl=tbl,
        query_text="foo",
        qvec=[0.0] * 4,
        pool_size=10,
        fusion="rrf",
        alpha=None,
        filter_authority=None,
    )
    cids = [t[0] for t in scored]
    # Dense top-1 still ought to lead (rank-1 in dense, never in sparse, but
    # rank-1 contributes 1/(60+1)). c1 contributes from sparse rank-1 too,
    # which compounds.
    assert cids[0] in {"c2", "c1"}
    # Both sparse-matched candidates are present.
    assert "c1" in cids
    assert "c4" in cids


def test_hybrid_query_skips_embedding_on_empty_kb(tmp_path: Path) -> None:
    """If the KB has no index dir, no Ollama call is attempted."""
    hits = hybrid_query(tmp_path, "anything", k=3, rerank=True)
    assert hits == []


def test_hybrid_query_alpha_backcompat_routes_through_alpha_blend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Passing ``alpha=`` with the default fusion still hits alpha-blend."""

    # Stage the minimum filesystem layout so the early-returns don't fire.
    (tmp_path / "index").mkdir()

    captured: dict[str, Any] = {}

    def fake_stage1(**kwargs: Any) -> list[tuple[str, float, float, float, dict[str, Any]]]:
        captured["fusion"] = kwargs["fusion"]
        captured["alpha"] = kwargs["alpha"]
        return [
            ("c0", 0.7, 0.6, 0.5, {"text": "x", "vector": [0.0]}),
            ("c1", 0.5, 0.4, 0.3, {"text": "y", "vector": [0.0]}),
        ]

    # Replace the LanceDB-heavy plumbing with stubs.
    import lab.rag.index as idx

    class FakeTable:
        def count_rows(self) -> int:
            return 5

    class FakeDB:
        def list_tables(self) -> Any:
            class _T:
                tables: tuple[str, ...] = ("chunks",)

            return _T()

        def open_table(self, _name: str) -> FakeTable:
            return FakeTable()

    monkeypatch.setattr(idx.lancedb, "connect", lambda _p: FakeDB())
    monkeypatch.setattr(idx, "_stage1_candidates", fake_stage1)
    # ``hybrid_query`` calls the unqualified ``embed_texts`` name bound at
    # import time in ``lab.rag.index``; patching the source module would be
    # a no-op since the reference is already resolved.
    monkeypatch.setattr(
        idx,
        "embed_texts",
        lambda *a, **kw: type("R", (), {"vectors": [[0.0]]})(),
    )

    hits = hybrid_query(tmp_path, "q", k=2, alpha=0.75, rerank=False)
    assert captured["fusion"] == "alpha"
    assert captured["alpha"] == pytest.approx(0.75)
    assert [h.chunk_id for h in hits] == ["c0", "c1"]
