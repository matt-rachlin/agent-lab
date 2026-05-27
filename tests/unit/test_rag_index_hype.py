"""Phase 11 — HyPE schema + hybrid_query wiring.

We mock LanceDB / embedder calls; tests stay CPU-only.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from lab.rag.index import (
    _hype_boost_dsim,
    _row_has_hype,
    _schema,
    _stage1_candidates,
    _table_has_hype,
)

# ---------- schema ----------------------------------------------------------


def test_schema_has_hype_fields() -> None:
    """Phase 11 fields are present on the canonical schema."""
    s = _schema(4)
    names = set(s.names)
    assert "hype_questions" in names
    assert "hype_vectors" in names


# ---------- _row_has_hype / _hype_boost_dsim ---------------------------------


def test_row_has_hype_negative_cases() -> None:
    assert _row_has_hype({}) is False
    assert _row_has_hype({"hype_vectors": None}) is False
    assert _row_has_hype({"hype_vectors": []}) is False
    # All empty vectors → still no hype.
    assert _row_has_hype({"hype_vectors": [None, [], None]}) is False


def test_row_has_hype_positive() -> None:
    assert _row_has_hype({"hype_vectors": [[0.1, 0.2]]}) is True
    assert _row_has_hype({"hype_vectors": [None, [0.1, 0.2]]}) is True


def test_hype_boost_dsim_takes_max_of_content_and_questions() -> None:
    row = {"hype_vectors": [[1.0, 0.0], [0.0, 1.0]]}
    # Query aligned with second hype vector → cosine 1.0 dominates content 0.2.
    boosted = _hype_boost_dsim(row, qvec=[0.0, 1.0], content_dsim=0.2)
    assert boosted == pytest.approx(1.0)


def test_hype_boost_dsim_preserves_content_when_hype_worse() -> None:
    row = {"hype_vectors": [[1.0, 0.0]]}
    # Query orthogonal to the hype vec → cosine 0. Content stays at 0.9.
    boosted = _hype_boost_dsim(row, qvec=[0.0, 1.0], content_dsim=0.9)
    assert boosted == pytest.approx(0.9)


def test_hype_boost_dsim_no_op_when_row_lacks_vectors() -> None:
    assert _hype_boost_dsim({}, qvec=[1.0], content_dsim=0.5) == pytest.approx(0.5)


# ---------- _table_has_hype --------------------------------------------------


def test_table_has_hype_detects_columns() -> None:
    class FakeSchema:
        names = ("chunk_id", "vector", "hype_vectors", "hype_questions")

    class FakeTable:
        schema = FakeSchema()

    assert _table_has_hype(FakeTable()) is True


def test_table_has_hype_negative() -> None:
    class FakeSchema:
        names = ("chunk_id", "vector")  # legacy v1/v2 schema

    class FakeTable:
        schema = FakeSchema()

    assert _table_has_hype(FakeTable()) is False


# ---------- _stage1_candidates with use_hype --------------------------------


def _make_fake_table(
    rows: list[dict[str, Any]],
    dense_order: list[int],
    distances: list[float] | None = None,
) -> Any:
    """Minimal LanceDB-shaped fake. ``dense_order`` is the row index in
    ascending-distance order; ``distances`` lets a test override the per-
    rank distance values (defaults to ``[0.0, 1.0, 2.0, ...]``).
    """

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
        def search(self, _vec: list[float]) -> FakeSearch:
            dense_results: list[dict[str, Any]] = []
            for rank, idx in enumerate(dense_order):
                r = dict(rows[idx])
                if distances is not None:
                    r["_distance"] = float(distances[rank])
                else:
                    r["_distance"] = float(rank)
                dense_results.append(r)
            return FakeSearch(dense_results)

        def to_arrow(self) -> FakeArrow:
            return FakeArrow()

    return FakeTable()


def test_stage1_use_hype_promotes_question_match() -> None:
    """A chunk whose hype-vec matches the query should climb the ranking
    when HyPE is on, even though its raw content vector lost the dense
    head race.
    """
    rows = [
        {
            "chunk_id": "c0",
            # c0 wins the content dense race (will be dense-rank #1).
            "vector": [1.0, 0.0],
            "section_path": [],
            "authority": "official",
            "sparse_json": json.dumps({}),
            "hype_vectors": [[1.0, 0.0]],  # aligned with c0's content
            "hype_questions": ["q0?"],
        },
        {
            "chunk_id": "c1",
            "vector": [1.0, 0.0],
            "section_path": [],
            "authority": "official",
            "sparse_json": json.dumps({}),
            # c1's hype-vec matches the *query* (not its own content).
            "hype_vectors": [[0.0, 1.0]],
            "hype_questions": ["q1?"],
        },
        {
            "chunk_id": "c2",
            "vector": [1.0, 0.0],
            "section_path": [],
            "authority": "official",
            "sparse_json": json.dumps({}),
            "hype_vectors": None,
            "hype_questions": None,
        },
    ]
    # Dense head ranks c0 first (distance 0.1 → raw d_sim ~0.91),
    # then c1 (distance 0.5 → raw d_sim ~0.67), then c2 (distance 1.0 →
    # raw d_sim 0.5). c1's hype-vec matches the query exactly (cosine
    # 1.0) so its boosted raw d_sim climbs above c0's 0.91.
    tbl = _make_fake_table(rows, dense_order=[0, 1, 2], distances=[0.1, 0.5, 1.0])

    # Query vector pointing orthogonal to the row content vectors. The fake
    # table returns precomputed _distance values regardless of qvec, so the
    # *only* signal that distinguishes c0 from c1 in the hype path is the
    # hype-vector cosine.
    qvec = [0.0, 1.0]

    scored_no_hype = _stage1_candidates(
        tbl=tbl,
        query_text="some query",
        qvec=qvec,
        pool_size=10,
        fusion="rrf",
        alpha=None,
        filter_authority=None,
        use_hype=False,
    )
    cids_no_hype = [t[0] for t in scored_no_hype]
    # Without HyPE: dense ranks govern → c0 wins.
    assert cids_no_hype[0] == "c0"

    scored_hype = _stage1_candidates(
        tbl=tbl,
        query_text="some query",
        qvec=qvec,
        pool_size=10,
        fusion="rrf",
        alpha=None,
        filter_authority=None,
        use_hype=True,
    )
    cids_hype = [t[0] for t in scored_hype]
    # With HyPE: c1's question vec matches qvec exactly → its boosted dsim
    # exceeds c0's whose hype vec is orthogonal to qvec. c1 must show up
    # ahead of c0 in the boosted dense ranking.
    pos_c0 = cids_hype.index("c0")
    pos_c1 = cids_hype.index("c1")
    assert pos_c1 < pos_c0


def test_stage1_use_hype_no_op_when_rows_lack_vectors() -> None:
    """use_hype=True on a KB with no hype columns leaves ranks unchanged."""
    rows = [
        {
            "chunk_id": f"c{i}",
            "vector": [1.0, 0.0],
            "section_path": [],
            "authority": "official",
            "sparse_json": json.dumps({}),
            # No hype_vectors.
        }
        for i in range(3)
    ]
    tbl = _make_fake_table(rows, dense_order=[2, 1, 0])

    scored = _stage1_candidates(
        tbl=tbl,
        query_text="anything",
        qvec=[1.0, 0.0],
        pool_size=10,
        fusion="rrf",
        alpha=None,
        filter_authority=None,
        use_hype=True,
    )
    cids = [t[0] for t in scored]
    # Dense ordering preserved.
    assert cids[0] == "c2"
