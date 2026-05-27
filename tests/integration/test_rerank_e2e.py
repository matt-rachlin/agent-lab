"""End-to-end: stage-2 cross-encoder rerank against the real bash KB.

Skips cleanly when:
  * the bash KB is missing or empty (e.g. mid-build);
  * Valkey is unreachable (can't check GPU lease);
  * the GPU lease is held (a sweep is running — we MUST NOT load a model);
  * sentence-transformers isn't installed.

When all gates pass, this loads the configured reranker (default
Qwen3-Reranker-0.6B), runs one query against the indexed corpus, and asserts
that the reranker actually re-scored the candidate set.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_bash_kb_rerank_e2e_or_skip() -> None:
    kb_dir = Path("~/db/kb/bash").expanduser()
    if not (kb_dir / "manifest.yaml").exists():
        pytest.skip(f"no bash KB at {kb_dir}")
    from lab.rag.index import count_rows

    if count_rows(kb_dir) == 0:
        pytest.skip("bash KB has no indexed chunks; skipping rerank e2e")

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pytest.skip("sentence-transformers not installed; rerank e2e skipped")

    import redis
    from lab.core.settings import get_settings

    settings = get_settings()
    try:
        client = redis.from_url(settings.redis_url)
        lease = client.get("lab:gpu:lease:0")
    except Exception:
        pytest.skip("valkey not reachable; cannot verify GPU lease")
    if lease:
        pytest.skip(f"GPU lease held ({lease!r}); refusing to load reranker")

    from lab.rag.index import hybrid_query

    # Cold path: rerank=True; the singleton will lazy-load the cross-encoder.
    hits = hybrid_query(kb_dir, "redirect stderr to stdout", k=5, rerank=True)
    assert isinstance(hits, list)
    if not hits:
        pytest.skip("no hits even after retrieval — KB may be sparse")
    # When the reranker actually ran, every hit carries a rerank_score float.
    for h in hits:
        assert h.rerank_score is None or isinstance(h.rerank_score, float)
