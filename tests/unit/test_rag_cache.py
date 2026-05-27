"""Unit tests for lab.rag.cache.RagCache.

We use a hand-rolled in-process Redis stub (no fakeredis dep needed) — Valkey
GET/SET/EXPIRE semantics are simple enough that a dict + TTL bookkeeping
covers the cases we care about. Real Valkey integration lives in
tests/integration/test_rag_cache_e2e.py.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from lab.rag.cache import (
    EMBED_TTL_SEC,
    KEY_HASH_LEN,
    RERANK_TTL_SEC,
    RagCache,
    RagCacheStats,
    embed_key,
    kb_version_token,
    rerank_key,
)


class _StubRedis:
    """Tiny redis-protocol-ish stub. Supports get/set/ping/close + TTLs."""

    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, float | None]] = {}
        self.set_calls: list[tuple[str, int | None]] = []

    def get(self, key: str) -> bytes | None:
        entry = self.store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.monotonic() > expires_at:
            del self.store[key]
            return None
        return value

    def set(self, key: str, value: bytes, ex: int | None = None) -> bool:
        self.set_calls.append((key, ex))
        expires_at = time.monotonic() + ex if ex else None
        self.store[key] = (value, expires_at)
        return True

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset_stats() -> Any:
    RagCache.stats = RagCacheStats()
    yield
    RagCache.stats = RagCacheStats()


def test_embed_key_shape_and_length() -> None:
    k = embed_key("How do I redirect stderr?", "qwen3-embedding:8b-q8_0")
    assert k.startswith("emb:")
    assert len(k) == len("emb:") + KEY_HASH_LEN


def test_embed_key_normalises_whitespace_and_case() -> None:
    a = embed_key("  How  Do I redirect stderr?", "m")
    b = embed_key("how do i redirect stderr?", "m")
    assert a == b


def test_embed_key_changes_with_model() -> None:
    a = embed_key("q", "qwen3-embedding:8b-q8_0")
    b = embed_key("q", "qwen3-embedding:4b")
    assert a != b


def test_rerank_key_namespaces_by_kb_version() -> None:
    a = rerank_key("q", kb_version="v1", top_k=5, rerank_model="m")
    b = rerank_key("q", kb_version="v2", top_k=5, rerank_model="m")
    assert a != b


def test_kb_version_token_prefers_explicit() -> None:
    class M:
        kb_version = "abc123"

    assert kb_version_token(M()) == "abc123"


def test_kb_version_token_falls_back_to_hash() -> None:
    class M:
        kb_version = None

        def model_dump(self, mode: str = "json") -> dict[str, Any]:
            return {"name": "bash", "status": "sealed"}

    tok = kb_version_token(M())
    # Stable + deterministic length.
    assert len(tok) == KEY_HASH_LEN
    # Same payload -> same token.
    assert kb_version_token(M()) == tok


def test_get_embedding_miss_records_counter() -> None:
    stub = _StubRedis()
    cache = RagCache(client=stub)
    assert cache.get_embedding("q", "m") is None
    snap = cache.stats_snapshot()
    assert snap["emb_miss"] == 1
    assert snap["emb_hit"] == 0


def test_put_then_get_embedding_round_trips() -> None:
    stub = _StubRedis()
    cache = RagCache(client=stub)
    vec = [0.1, 0.2, 0.3]
    cache.put_embedding("redirect stderr", "m", vec)
    out = cache.get_embedding("redirect stderr", "m")
    assert out == vec
    snap = cache.stats_snapshot()
    assert snap["emb_hit"] == 1
    # And the underlying SET had the right TTL.
    key, ex = stub.set_calls[-1]
    assert key.startswith("emb:")
    assert ex == EMBED_TTL_SEC


def test_put_rerank_uses_correct_ttl() -> None:
    stub = _StubRedis()
    cache = RagCache(client=stub)
    cache.put_rerank(
        "q",
        kb_version="v1",
        top_k=5,
        rerank_model="m",
        hits=[{"chunk_id": "c1", "rerank_score": 0.9}],
    )
    key, ex = stub.set_calls[-1]
    assert key.startswith("rrk:")
    assert ex == RERANK_TTL_SEC


def test_rerank_round_trip_returns_same_payload() -> None:
    stub = _StubRedis()
    cache = RagCache(client=stub)
    hits = [
        {"chunk_id": "c1", "rerank_score": 0.91, "text": "hi"},
        {"chunk_id": "c2", "rerank_score": 0.55, "text": "world"},
    ]
    cache.put_rerank("q", kb_version="v1", top_k=2, rerank_model="m", hits=hits)
    out = cache.get_rerank("q", kb_version="v1", top_k=2, rerank_model="m")
    assert out == hits


def test_kb_version_mismatch_treated_as_miss() -> None:
    stub = _StubRedis()
    cache = RagCache(client=stub)
    cache.put_rerank("q", kb_version="v1", top_k=2, rerank_model="m", hits=[{"x": 1}])
    out = cache.get_rerank("q", kb_version="v2", top_k=2, rerank_model="m")
    assert out is None


def test_empty_embedding_does_not_persist() -> None:
    stub = _StubRedis()
    cache = RagCache(client=stub)
    cache.put_embedding("q", "m", [])
    assert stub.set_calls == []


def test_malformed_payload_returns_miss() -> None:
    stub = _StubRedis()
    cache = RagCache(client=stub)
    # Plant a key with garbage and try to read it.
    key = embed_key("q", "m")
    stub.store[key] = (b"not json at all", None)
    assert cache.get_embedding("q", "m") is None
