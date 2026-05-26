"""Integration test: query the real ~/db/kb/bash/ index.

Skips cleanly if the bash KB is empty or `enrichment_pending` so the test
suite stays green during in-progress KB builds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.rag.index import count_rows
from lab.rag.manifest import load_manifest


@pytest.mark.integration
def test_bash_kb_query_or_skip():
    kb_dir = Path("~/db/kb/bash").expanduser()
    if not (kb_dir / "manifest.yaml").exists():
        pytest.skip(f"no bash KB at {kb_dir}")
    manifest = load_manifest(kb_dir / "manifest.yaml")
    n_rows = count_rows(kb_dir)
    if n_rows == 0:
        pytest.skip(
            f"bash KB is {manifest.status!r} with 0 indexed chunks; "
            f"smoke-skip until indexing completes"
        )
    # If we get here the KB is real and indexed — exercise hybrid_query.
    # We import lazily so the test loads even when ollama isn't reachable.
    import redis

    from lab.rag.index import hybrid_query
    from lab.settings import get_settings

    settings = get_settings()
    # Refuse to query if a sweep is holding the GPU lease.
    try:
        client = redis.from_url(settings.redis_url)
        lease = client.get("lab:gpu:lease:0")
    except Exception:
        pytest.skip("valkey not reachable; cannot verify GPU lease state")
    if lease:
        pytest.skip(f"GPU lease held ({lease!r}); refusing to query during sweep")

    hits = hybrid_query(kb_dir, "how do I redirect stderr to stdout", k=5)
    assert isinstance(hits, list)
    # Don't assert exact contents (depends on the live KB), just shape.
    for h in hits:
        assert h.chunk_id
        assert isinstance(h.section_path, list)
        assert 0.0 <= h.score <= 1.0 + 1e-6
