"""Unit tests for the Phase 10 skip heuristics in lab.rag.skip.

We exercise each branch of compute_skip_decision with hand-crafted candidate
lists and total-row counts so the contract is locked down independently of
the index/retrieval plumbing.
"""

from __future__ import annotations

from typing import Any

import pytest
from lab.rag import skip
from lab.rag.skip import (
    HIGH_CONFIDENCE_TOP1,
    LOW_CONFIDENCE_TOP2,
    SMALL_CANDIDATE_SET,
    SMALL_KB_CHUNKS,
    compute_skip_decision,
    get_low_confidence_counters,
    get_skip_counters,
    maybe_emit_low_confidence,
    reset_counters,
)


@pytest.fixture(autouse=True)
def _reset_counters() -> Any:
    reset_counters()
    yield
    reset_counters()


def _cands(n: int, top1: float = 0.5, top2: float = 0.4) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(n):
        score = top1 if i == 0 else top2 if i == 1 else max(0.0, top2 - 0.1 * (i - 1))
        out.append({"chunk_id": f"c{i}", "score": score})
    return out


def test_caller_disabled_overrides_everything() -> None:
    d = compute_skip_decision(
        candidates=_cands(50, top1=0.5, top2=0.4),
        total_kb_rows=5000,
        rerank_requested=False,
    )
    assert d.use_reranker is False
    assert d.reason == "caller_disabled"
    # Counters NOT incremented for the explicit-disable path (the skip
    # heuristic counters track auto-skips only).
    assert get_skip_counters() == {}


def test_small_kb_skip() -> None:
    d = compute_skip_decision(
        candidates=_cands(50),
        total_kb_rows=SMALL_KB_CHUNKS - 1,
        rerank_requested=True,
    )
    assert d.use_reranker is False
    assert d.reason == "small_kb"
    assert get_skip_counters().get("small_kb") == 1


def test_small_candidate_set_skip() -> None:
    d = compute_skip_decision(
        candidates=_cands(SMALL_CANDIDATE_SET - 1),
        total_kb_rows=5000,
        rerank_requested=True,
    )
    assert d.use_reranker is False
    assert d.reason == "small_candidate_set"
    assert get_skip_counters().get("small_candidate_set") == 1


def test_high_confidence_top1_skip() -> None:
    cands = _cands(50, top1=HIGH_CONFIDENCE_TOP1 + 0.01, top2=LOW_CONFIDENCE_TOP2 - 0.01)
    d = compute_skip_decision(candidates=cands, total_kb_rows=5000, rerank_requested=True)
    assert d.use_reranker is False
    assert d.reason == "high_confidence_top1"
    assert get_skip_counters().get("high_confidence_top1") == 1


def test_rerank_runs_when_no_skip_triggers() -> None:
    cands = _cands(50, top1=0.6, top2=0.55)  # gap too small for high-conf skip
    d = compute_skip_decision(candidates=cands, total_kb_rows=5000, rerank_requested=True)
    assert d.use_reranker is True
    assert d.reason == "rerank"
    assert get_skip_counters() == {}


def test_low_confidence_alert_logs_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """If max rerank_score < 0.30, we increment the counter."""
    from pathlib import Path

    hits = [type("H", (), {"rerank_score": 0.15})()]
    maybe_emit_low_confidence(hits, kb_dir=Path("/tmp/bash"))
    assert get_low_confidence_counters().get("bash") == 1


def test_low_confidence_no_alert_when_score_high() -> None:
    from pathlib import Path

    hits = [type("H", (), {"rerank_score": 0.80})()]
    maybe_emit_low_confidence(hits, kb_dir=Path("/tmp/bash"))
    assert get_low_confidence_counters() == {}


def test_low_confidence_skipped_when_rerank_score_missing() -> None:
    """When the reranker didn't run, rerank_score is None — no alert."""
    from pathlib import Path

    hits = [type("H", (), {"rerank_score": None})()]
    maybe_emit_low_confidence(hits, kb_dir=Path("/tmp/bash"))
    assert get_low_confidence_counters() == {}


def test_counter_snapshot_is_a_copy() -> None:
    _ = compute_skip_decision(
        candidates=_cands(50, top1=HIGH_CONFIDENCE_TOP1 + 0.01, top2=0.0),
        total_kb_rows=5000,
        rerank_requested=True,
    )
    snap_a = get_skip_counters()
    snap_a["high_confidence_top1"] = 999
    snap_b = get_skip_counters()
    assert snap_b["high_confidence_top1"] == 1
    _ = skip  # silence import-only warnings
