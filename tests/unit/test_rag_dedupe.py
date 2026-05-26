"""Unit tests for the Phase 10 cosine-dedupe helper.

The dedupe step runs *before* the cross-encoder so we don't waste compute on
near-duplicate stage-1 candidates. It must keep the highest-ranked candidate
in each cluster and surface the dropped IDs for telemetry.
"""

from __future__ import annotations

from typing import Any

from lab.rag.skip import DEDUPE_COSINE, dedupe_candidates


def _cand(cid: str, vec: list[float]) -> dict[str, Any]:
    return {"chunk_id": cid, "vector": vec, "text": cid}


def test_dedupe_keeps_highest_ranked_in_cluster() -> None:
    # c1, c2 share an identical vector (cosine=1). c3 is orthogonal.
    cands = [
        _cand("c1", [1.0, 0.0]),
        _cand("c2", [1.0, 0.0]),  # near-duplicate of c1
        _cand("c3", [0.0, 1.0]),  # unrelated
    ]
    kept, clusters = dedupe_candidates(cands)
    kept_ids = [c["chunk_id"] for c in kept]
    assert kept_ids == ["c1", "c3"]
    assert any(set(cluster) == {"c1", "c2"} for cluster in clusters)


def test_dedupe_threshold_respected() -> None:
    # Two vectors with cosine just *below* threshold should both be kept.
    import math

    # Construct vectors with cosine = 0.85 (below 0.92 default).
    theta = math.acos(0.85)
    v1 = [1.0, 0.0]
    v2 = [math.cos(theta), math.sin(theta)]
    cands = [_cand("c1", v1), _cand("c2", v2)]
    kept, clusters = dedupe_candidates(cands)
    assert len(kept) == 2
    assert clusters == []  # nothing collapsed


def test_dedupe_no_vector_kept_as_is() -> None:
    """Candidates missing a usable vector are kept (we can't measure cosine)."""
    cands = [
        {"chunk_id": "c1", "vector": None, "text": "x"},
        {"chunk_id": "c2", "vector": [], "text": "y"},
        _cand("c3", [1.0, 0.0]),
    ]
    kept, _clusters = dedupe_candidates(cands)
    kept_ids = [c["chunk_id"] for c in kept]
    assert set(kept_ids) == {"c1", "c2", "c3"}


def test_dedupe_custom_threshold() -> None:
    # cosine ~= 0.948 between these two unit-ish vectors.
    cands = [
        _cand("c1", [1.0, 0.0]),
        _cand("c2", [1.0, 0.33]),
    ]
    # Very strict threshold keeps both.
    kept, _clusters = dedupe_candidates(cands, cosine_threshold=0.99)
    assert len(kept) == 2
    # Default threshold (0.92) collapses them.
    kept2, _ = dedupe_candidates(cands, cosine_threshold=DEDUPE_COSINE)
    assert len(kept2) == 1
