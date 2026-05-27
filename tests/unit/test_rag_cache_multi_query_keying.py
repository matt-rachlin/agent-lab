"""Phase 12 — rerank cache key must include the ``multi_query`` flag so
multi-query and single-query results never collide for the same
``(query, kb_version, top_k, rerank_model)``.
"""

from __future__ import annotations

from lab.rag.cache import RagCache, rerank_key


def test_rerank_key_changes_with_multi_query() -> None:
    a = rerank_key("q", kb_version="v1", top_k=5, rerank_model="m", multi_query=False)
    b = rerank_key("q", kb_version="v1", top_k=5, rerank_model="m", multi_query=True)
    assert a != b


def test_rag_cache_multi_query_isolates_payloads() -> None:
    """A multi-query write must not be returned by a single-query read."""

    class _StubRedis:
        def __init__(self) -> None:
            self.store: dict[str, bytes] = {}

        def get(self, key: str) -> bytes | None:
            return self.store.get(key)

        def set(self, key: str, value: bytes, ex: int | None = None) -> bool:
            self.store[key] = value
            return True

        def ping(self) -> bool:
            return True

        def close(self) -> None:
            return None

    cache = RagCache(client=_StubRedis())
    multi_hits = [{"chunk_id": "from-multi", "rerank_score": 0.9}]
    single_hits = [{"chunk_id": "from-single", "rerank_score": 0.8}]

    cache.put_rerank(
        "q",
        kb_version="v1",
        top_k=5,
        rerank_model="m",
        hits=multi_hits,
        multi_query=True,
    )
    cache.put_rerank(
        "q",
        kb_version="v1",
        top_k=5,
        rerank_model="m",
        hits=single_hits,
        multi_query=False,
    )

    out_multi = cache.get_rerank("q", kb_version="v1", top_k=5, rerank_model="m", multi_query=True)
    out_single = cache.get_rerank(
        "q", kb_version="v1", top_k=5, rerank_model="m", multi_query=False
    )
    assert out_multi == multi_hits
    assert out_single == single_hits
