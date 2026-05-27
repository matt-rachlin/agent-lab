"""Bench: ``LabReranker.rerank`` p50/p95 on a 50-candidate fixture.

Times the host-side rerank service (cross-encoder) over HTTP — the
in-process path requires sentence-transformers in the venv, which the
default `dev` extra doesn't ship.

Skips when:

- The rerank service at ``127.0.0.1:8401/healthz`` doesn't respond 200
"""

from __future__ import annotations

import os
import statistics
import time

from benchmarks import BenchmarkSkipped

RERANK_BASE = "http://127.0.0.1:8401"
RERANK_HEALTHZ = f"{RERANK_BASE}/healthz"
RERANK_URL_ENV = "LAB_RAG_RERANKER_URL"
# rerank_client.rerank_via_http appends "/rerank" — pass the base URL only.
RERANK_URL = RERANK_BASE
DEFAULT_N = 20
DEFAULT_TOP_N = 10

QUERY = "how do I redirect stderr to stdout"
CAND_TEXTS = [
    # 50 synthetic candidates — mix of relevant and irrelevant lines so the
    # cross-encoder actually does work (a fully degenerate set of identical
    # candidates would short-circuit some implementations).
    f"candidate {i}: "
    + (
        "redirect stderr 2>&1 to combine streams in bash"
        if i % 3 == 0
        else "tar czf archive.tar.gz some/dir to compress directories"
        if i % 3 == 1
        else "use awk '{print $1}' to extract the first column from a file"
    )
    for i in range(50)
]


def _check_rerank_alive() -> None:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise BenchmarkSkipped(f"httpx not importable: {exc}") from exc

    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(RERANK_HEALTHZ)
    except Exception as exc:
        raise BenchmarkSkipped(f"rerank service unreachable: {exc}") from exc

    if resp.status_code != 200:
        raise BenchmarkSkipped(f"rerank /healthz returned {resp.status_code}")


def run(n: int = DEFAULT_N, top_n: int = DEFAULT_TOP_N) -> dict[str, float]:
    """Time n rerank calls. Returns p50/p95/mean."""
    _check_rerank_alive()

    try:
        from lab.rag.rerank import LabReranker
    except ImportError as exc:  # pragma: no cover
        raise BenchmarkSkipped(f"lab.rag.rerank not importable: {exc}") from exc

    # Force the HTTP path so we don't accidentally try to load the
    # cross-encoder model in the bench process.
    os.environ[RERANK_URL_ENV] = RERANK_URL
    reranker = LabReranker(idle_unload_sec=0)

    if reranker.disabled:
        # Default reranker model is the "none" sentinel post-EXP-004c;
        # use an explicit model id so the host service does real work.
        reranker = LabReranker(model_name="Qwen/Qwen3-Reranker-0.6B", idle_unload_sec=0)

    candidates = [{"text": t} for t in CAND_TEXTS]

    # Warmup + sanity: the rerank must actually inject "rerank_score".
    # LabReranker silently pass-throughs on HTTP failure — without this
    # check we'd record a pass-through latency as "rerank latency".
    try:
        warm = reranker.rerank(QUERY, list(candidates), top_n=top_n)
    except Exception as exc:
        raise BenchmarkSkipped(f"warmup rerank failed: {exc}") from exc
    if not warm or "rerank_score" not in warm[0]:
        raise BenchmarkSkipped(
            "rerank returned pass-through results (no rerank_score field); "
            "the host service is probably misrouted — check LAB_RAG_RERANKER_URL"
        )

    timings: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            reranker.rerank(QUERY, list(candidates), top_n=top_n)
        except Exception as exc:
            raise BenchmarkSkipped(f"rerank raised: {exc}") from exc
        timings.append(time.perf_counter() - t0)

    timings.sort()
    return {
        "p50_sec": statistics.median(timings),
        "p95_sec": timings[int(0.95 * len(timings)) - 1],
        "mean_sec": statistics.fmean(timings),
        "n": float(len(timings)),
        "candidates": float(len(CAND_TEXTS)),
    }
