"""Phase 10 — smart reranker skip + dedupe + low-confidence alert.

The skip heuristics live here so stage-2 (cross-encoder rerank) can be
avoided in cases where the stage-1 fusion already gives a confident answer.
Dedupe sits next door so the reranker, when it does run, isn't wasted on
near-duplicate candidates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Stage-1 candidate-count threshold below which reranking adds little value.
SMALL_CANDIDATE_SET = 10

#: Top-1 stage-1 score threshold above which reranking is skipped iff top-2
#: lags by a large margin.
HIGH_CONFIDENCE_TOP1 = 0.92
LOW_CONFIDENCE_TOP2 = 0.80

#: Small-KB cutoff (in indexed chunk count) — reranking rarely pays for
#: itself when the corpus is this small.
SMALL_KB_CHUNKS = 1000

#: Cosine similarity above which two candidates count as near-duplicates.
DEDUPE_COSINE = 0.92

#: Low-confidence threshold on the reranker's top score.
LOW_CONFIDENCE_RERANK = 0.30


@dataclass(slots=True)
class SkipDecision:
    """Result of :func:`compute_skip_decision`."""

    use_reranker: bool
    reason: str


def compute_skip_decision(
    *,
    candidates: list[dict[str, Any]],
    total_kb_rows: int,
    rerank_requested: bool,
) -> SkipDecision:
    """Decide whether to run the stage-2 cross-encoder reranker.

    Skip paths (any one wins):
      * caller asked for ``rerank=False``;
      * KB has fewer than :data:`SMALL_KB_CHUNKS` indexed rows;
      * stage-1 returned fewer than :data:`SMALL_CANDIDATE_SET` candidates;
      * stage-1 top-1 ≫ top-2 — large gap = high confidence.

    Counters emitted via :func:`_emit_skip_counter` so the existing exporter
    pattern can scrape them.
    """
    if not rerank_requested:
        return SkipDecision(use_reranker=False, reason="caller_disabled")
    if total_kb_rows < SMALL_KB_CHUNKS:
        _emit_skip_counter("small_kb")
        return SkipDecision(use_reranker=False, reason="small_kb")
    if len(candidates) < SMALL_CANDIDATE_SET:
        _emit_skip_counter("small_candidate_set")
        return SkipDecision(use_reranker=False, reason="small_candidate_set")
    if len(candidates) >= 2:
        top1 = float(candidates[0].get("score", 0.0))
        top2 = float(candidates[1].get("score", 0.0))
        if top1 > HIGH_CONFIDENCE_TOP1 and top2 < LOW_CONFIDENCE_TOP2:
            _emit_skip_counter("high_confidence_top1")
            return SkipDecision(use_reranker=False, reason="high_confidence_top1")
    return SkipDecision(use_reranker=True, reason="rerank")


def dedupe_candidates(
    candidates: list[dict[str, Any]],
    *,
    cosine_threshold: float = DEDUPE_COSINE,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    """Cluster near-duplicates by cosine similarity to a higher-ranked candidate.

    Each candidate must carry a ``"vector"`` (the dense embedding) and a
    ``"chunk_id"``. We keep the first (highest-ranked) entry in each cluster
    and drop the rest. The returned ``dupe_clusters`` lists drop-rank groups
    (``[kept_id, dropped_id, ...]``) so callers can surface provenance.

    Candidates without a usable vector are kept as-is (we can't compute
    similarity on them).
    """
    if len(candidates) <= 1:
        return list(candidates), []

    kept: list[dict[str, Any]] = []
    clusters: dict[str, list[str]] = {}

    for cand in candidates:
        v = _as_float_list(cand.get("vector"))
        cid = str(cand.get("chunk_id"))
        if v is None:
            kept.append(cand)
            continue
        duplicate_of: str | None = None
        for prior in kept:
            pv = _as_float_list(prior.get("vector"))
            if pv is None or len(pv) != len(v):
                continue
            if _cosine(v, pv) >= cosine_threshold:
                duplicate_of = str(prior.get("chunk_id"))
                break
        if duplicate_of is None:
            kept.append(cand)
            clusters.setdefault(cid, [cid])
        else:
            clusters.setdefault(duplicate_of, [duplicate_of]).append(cid)
            cand["dupe_of"] = duplicate_of

    dupe_clusters = [v for v in clusters.values() if len(v) > 1]
    return kept, dupe_clusters


def maybe_emit_low_confidence(hits: list[Any], *, kb_dir: Path) -> None:
    """If the top rerank score is below :data:`LOW_CONFIDENCE_RERANK`, log + count.

    ``hits`` items may be :class:`lab.rag.index.Hit` instances or dicts; both
    expose a ``rerank_score`` attribute / key.
    """
    if not hits:
        return
    top = hits[0]
    raw = getattr(top, "rerank_score", None)
    if raw is None and isinstance(top, dict):
        raw = top.get("rerank_score")
    if raw is None:
        return
    if float(raw) < LOW_CONFIDENCE_RERANK:
        kb_name = kb_dir.name
        logger.warning(
            "low_confidence_rerank kb=%s top_score=%.4f threshold=%.2f",
            kb_name,
            float(raw),
            LOW_CONFIDENCE_RERANK,
        )
        _emit_low_confidence_counter(kb_name)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_float_list(v: Any) -> list[float] | None:
    if v is None:
        return None
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return out or None


def _cosine(a: list[float], b: list[float]) -> float:
    s = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        s += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return s / (math.sqrt(na) * math.sqrt(nb))


# Telemetry hooks — kept process-local for now (no Prometheus dep at import
# time). The exporter scrapes via :func:`get_skip_counters` etc.

_SKIP_COUNTERS: dict[str, int] = {}
_LOW_CONFIDENCE_COUNTERS: dict[str, int] = {}


def _emit_skip_counter(reason: str) -> None:
    _SKIP_COUNTERS[reason] = _SKIP_COUNTERS.get(reason, 0) + 1


def _emit_low_confidence_counter(kb: str) -> None:
    _LOW_CONFIDENCE_COUNTERS[kb] = _LOW_CONFIDENCE_COUNTERS.get(kb, 0) + 1


def get_skip_counters() -> dict[str, int]:
    """Snapshot of skip counters keyed by reason (process-local)."""
    return dict(_SKIP_COUNTERS)


def get_low_confidence_counters() -> dict[str, int]:
    """Snapshot of low-confidence counters keyed by KB name."""
    return dict(_LOW_CONFIDENCE_COUNTERS)


def reset_counters() -> None:
    """Clear telemetry counters (mostly for tests)."""
    _SKIP_COUNTERS.clear()
    _LOW_CONFIDENCE_COUNTERS.clear()
