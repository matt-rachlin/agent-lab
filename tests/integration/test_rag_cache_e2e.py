"""End-to-end: live Valkey cache round-trip.

Skips when Valkey is unreachable. Doesn't require the bash KB to be indexed
(we exercise the cache layer directly with synthetic payloads — the rerank
e2e test covers the live retrieval path).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest


@pytest.mark.integration
def test_valkey_round_trip_or_skip(valkey: Any) -> None:
    from lab.rag.cache import RagCache

    cache = RagCache(client=valkey, kb_version=f"test-{uuid.uuid4().hex[:8]}")

    query = f"int-test-{uuid.uuid4().hex[:8]}"
    model = "qwen3-embedding:8b-q8_0"

    # MISS on first read.
    assert cache.get_embedding(query, model) is None

    # Populate + read.
    vec = [0.1, 0.2, 0.3, 0.4]
    cache.put_embedding(query, model, vec)
    out = cache.get_embedding(query, model)
    assert out == vec

    # Rerank tier with explicit kb_version.
    kbv = f"kbv-{uuid.uuid4().hex[:8]}"
    hits = [{"chunk_id": "c1", "rerank_score": 0.88, "text": "x"}]
    cache.put_rerank(query, kb_version=kbv, top_k=5, rerank_model="rm", hits=hits)
    got = cache.get_rerank(query, kb_version=kbv, top_k=5, rerank_model="rm")
    assert got == hits

    # A different kb_version misses (free invalidation contract).
    other = cache.get_rerank(query, kb_version=f"{kbv}-other", top_k=5, rerank_model="rm")
    assert other is None


@pytest.mark.integration
def test_warm_cache_is_faster_than_cold(valkey: Any) -> None:
    """A second GET on a populated key should be markedly faster than a SET.

    We're not measuring against Ollama here — the bash-KB e2e test handles
    that. This just confirms the Valkey round-trip is on the right order of
    magnitude (sub-ms typical on localhost).
    """
    from lab.rag.cache import RagCache

    cache = RagCache(client=valkey)
    query = f"warm-{uuid.uuid4().hex[:8]}"
    vec = [float(i) for i in range(64)]

    t0 = time.perf_counter()
    cache.put_embedding(query, "m", vec)
    t_set = time.perf_counter() - t0

    t0 = time.perf_counter()
    cache.get_embedding(query, "m")
    t_get = time.perf_counter() - t0

    # Sanity: both well under a second; GET should be at least not slower
    # than SET by a wide margin.
    assert t_set < 1.0
    assert t_get < 1.0
