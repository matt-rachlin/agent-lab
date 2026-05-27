"""Two-tier Valkey/Redis cache for the RAG pipeline.

Tier 1 — **query-embedding cache** (``emb:``): caches the dense embedding
vector for a (query, model_id) pair. TTL 24h. Saves ~20-40 ms of Ollama
round-trip per duplicate query.

Tier 2 — **rerank-result cache** (``rrk:``): caches the reranker's ordered
hits for a (query, kb_version, top_k, rerank_model) tuple. TTL 1h. Saves
~100-250 ms of cross-encoder compute per duplicate.

KB-version namespacing is what makes invalidation free: when a KB rebuilds,
its version token changes, so the old keys become unreachable and ``allkeys-lru``
evicts them as new keys are inserted. We never need to ``SCAN+DEL``.

Cache misses degrade to a non-cached call cleanly — Valkey unreachable, key
parse failure, or empty payload all return ``None`` so callers can fall back
without exception handling at every call site.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import redis as _redis

logger = logging.getLogger(__name__)

#: Tier 1 TTL — 24 hours.
EMBED_TTL_SEC = 86_400
#: Tier 2 TTL — 1 hour.
RERANK_TTL_SEC = 3_600
#: Hash prefix length (in hex chars). 16 chars = 64 bits — collision odds
#: are ~1 in 4 billion for any pair of keys at our scale.
KEY_HASH_LEN = 16


# ---------------------------------------------------------------------------
# key construction (pure helpers — exposed for tests)
# ---------------------------------------------------------------------------


def _normalize_query(q: str) -> str:
    """Collapse whitespace + lowercase. Cache keys treat 'How do I?' and
    'how do i' as the same query — slightly aggressive but matches user
    expectations and BM25 already lower-cases.
    """
    return re.sub(r"\s+", " ", q.strip()).lower()


def _sha256_short(s: str) -> str:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h[:KEY_HASH_LEN]


def embed_key(query: str, model_id: str) -> str:
    """Stable key for the tier-1 embedding cache."""
    payload = f"{model_id}|{_normalize_query(query)}"
    return f"emb:{_sha256_short(payload)}"


def rerank_key(
    query: str,
    kb_version: str,
    top_k: int,
    rerank_model: str,
    *,
    multi_query: bool = False,
) -> str:
    """Stable key for the tier-2 rerank-result cache.

    ``kb_version`` is the namespace token — see :func:`kb_version_token`.

    ``multi_query`` (Phase 12) is folded into the hash so that multi-query
    expansion results don't collide with single-query results for the same
    (query, kb_version, top_k, rerank_model). Defaults to False to keep
    keys stable for callers that don't yet pass the flag.
    """
    mq_tag = "mq1" if multi_query else "mq0"
    payload = f"{kb_version}|{rerank_model}|{top_k}|{mq_tag}|{_normalize_query(query)}"
    return f"rrk:{_sha256_short(payload)}"


def kb_version_token(manifest: Any) -> str:
    """Derive a stable ``kb_version`` token from a manifest.

    Prefers an explicit ``kb_version`` field on the manifest; falls back to a
    hash of the manifest body (mode='json' dump) so old manifests still get a
    deterministic, content-addressed token without a schema change.
    """
    if manifest is None:
        return "unknown"
    explicit = getattr(manifest, "kb_version", None)
    if explicit:
        return str(explicit)
    if hasattr(manifest, "model_dump"):
        payload = json.dumps(manifest.model_dump(mode="json"), sort_keys=True)
    elif isinstance(manifest, dict):
        payload = json.dumps(manifest, sort_keys=True, default=str)
    else:
        payload = str(manifest)
    return _sha256_short(payload)


# ---------------------------------------------------------------------------
# RagCache
# ---------------------------------------------------------------------------


class RagCacheStats:
    """In-process counter pair (hits, misses) keyed by tier name."""

    __slots__ = ("hits", "misses")

    def __init__(self) -> None:
        self.hits: dict[str, int] = {"emb": 0, "rrk": 0}
        self.misses: dict[str, int] = {"emb": 0, "rrk": 0}

    def record_hit(self, tier: str) -> None:
        self.hits[tier] = self.hits.get(tier, 0) + 1

    def record_miss(self, tier: str) -> None:
        self.misses[tier] = self.misses.get(tier, 0) + 1

    def snapshot(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for tier, c in self.hits.items():
            out[f"{tier}_hit"] = c
        for tier, c in self.misses.items():
            out[f"{tier}_miss"] = c
        return out

    def reset(self) -> None:
        for k in list(self.hits):
            self.hits[k] = 0
        for k in list(self.misses):
            self.misses[k] = 0


class RagCache:
    """Two-tier Valkey cache.

    Construct with an explicit ``client`` (for tests + DI) or a ``valkey_url``
    (the production path — resolves :class:`lab.settings.Settings.redis_url`).
    """

    #: Per-process stats. Shared across instances so the exporter can scrape
    #: a single counter set regardless of how many ``RagCache`` instances
    #: live in the process.
    stats: RagCacheStats = RagCacheStats()

    def __init__(
        self,
        *,
        client: _redis.Redis[bytes] | _redis.Redis[str] | None = None,
        valkey_url: str | None = None,
        kb_version: str | None = None,
        kb_name: str = "unknown",
    ) -> None:
        self.kb_version: str = kb_version or "unknown"
        self.kb_name: str = kb_name
        if client is not None:
            self._client: Any = client
        else:
            self._client = self._connect(valkey_url)

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------

    @staticmethod
    def _connect(valkey_url: str | None) -> Any:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - env guard
            raise RuntimeError("redis client not installed; RagCache cannot run") from exc
        if valkey_url is None:
            from lab.core.settings import get_settings

            valkey_url = get_settings().redis_url
        try:
            client = redis.Redis.from_url(valkey_url, decode_responses=False)
            client.ping()
        except Exception as exc:
            logger.warning("RagCache: Valkey unreachable (%s); falling back to no-op", exc)
            return _NoopRedis()
        return client

    # ------------------------------------------------------------------
    # tier 1 — embedding cache
    # ------------------------------------------------------------------

    def get_embedding(self, query: str, model_id: str) -> list[float] | None:
        key = embed_key(query, model_id)
        try:
            raw = self._client.get(key)
        except Exception as exc:
            logger.debug("RagCache.get_embedding miss-by-exception: %s", exc)
            self.stats.record_miss("emb")
            return None
        if raw is None:
            self.stats.record_miss("emb")
            return None
        vec = _decode_embedding(raw)
        if vec is None:
            self.stats.record_miss("emb")
            return None
        self.stats.record_hit("emb")
        return vec

    def put_embedding(
        self,
        query: str,
        model_id: str,
        embedding: list[float],
    ) -> None:
        if not embedding:
            return
        key = embed_key(query, model_id)
        try:
            self._client.set(key, _encode_embedding(embedding), ex=EMBED_TTL_SEC)
        except Exception as exc:
            logger.debug("RagCache.put_embedding write failed: %s", exc)

    # ------------------------------------------------------------------
    # tier 2 — rerank-result cache
    # ------------------------------------------------------------------

    def get_rerank(
        self,
        query: str,
        kb_version: str,
        top_k: int,
        rerank_model: str,
        *,
        multi_query: bool = False,
    ) -> list[dict[str, Any]] | None:
        key = rerank_key(query, kb_version, top_k, rerank_model, multi_query=multi_query)
        try:
            raw = self._client.get(key)
        except Exception as exc:
            logger.debug("RagCache.get_rerank miss-by-exception: %s", exc)
            self.stats.record_miss("rrk")
            return None
        if raw is None:
            self.stats.record_miss("rrk")
            return None
        hits = _decode_rerank(raw)
        if hits is None:
            self.stats.record_miss("rrk")
            return None
        self.stats.record_hit("rrk")
        return hits

    def put_rerank(
        self,
        query: str,
        kb_version: str,
        top_k: int,
        rerank_model: str,
        hits: list[dict[str, Any]],
        *,
        multi_query: bool = False,
    ) -> None:
        if not hits:
            return
        key = rerank_key(query, kb_version, top_k, rerank_model, multi_query=multi_query)
        try:
            self._client.set(key, _encode_rerank(hits), ex=RERANK_TTL_SEC)
        except Exception as exc:
            logger.debug("RagCache.put_rerank write failed: %s", exc)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def stats_snapshot(self) -> dict[str, int]:
        return self.stats.snapshot()


# ---------------------------------------------------------------------------
# encoding helpers
# ---------------------------------------------------------------------------


def _encode_embedding(vec: Iterable[float]) -> bytes:
    """Compact JSON encoding — 4-5 bytes per dim is fine at the corpus scales
    we work with (a 4096-dim vector is ~24 KB, well under Valkey's 512 MB
    object limit and small enough that compression isn't worth the dep).
    """
    return json.dumps(list(vec)).encode("utf-8")


def _decode_embedding(raw: bytes) -> list[float] | None:
    try:
        payload = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        data = json.loads(payload)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    try:
        return [float(x) for x in data]
    except (TypeError, ValueError):
        return None


def _encode_rerank(hits: list[dict[str, Any]]) -> bytes:
    return json.dumps(hits, default=str).encode("utf-8")


def _decode_rerank(raw: bytes) -> list[dict[str, Any]] | None:
    try:
        payload = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        data = json.loads(payload)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    out: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# no-op client (used when Valkey is unreachable)
# ---------------------------------------------------------------------------


class _NoopRedis:
    """Drop-in for redis.Redis — every op is a miss / silent write.

    Lets RagCache work end-to-end even when Valkey is down; the lab keeps
    running, you just lose the cache benefit. Better than crashing the
    retrieval path on every query.
    """

    def get(self, _key: str) -> None:
        return None

    def set(self, _key: str, _value: bytes, ex: int | None = None) -> bool:
        return True

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        return None
