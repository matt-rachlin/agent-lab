"""HTTP client for the host-side rerank service.

Used by :meth:`lab.rag.rerank.LabReranker.rerank` when ``LAB_RAG_RERANKER_URL``
is set. The sandbox image no longer ships ``sentence-transformers`` / ``torch``,
so the cell calls into this client, which POSTs to the host-side service
mounted at ``host.containers.internal:8401``.

The client is deliberately stdlib-friendly: only ``httpx`` (already in the
sandbox image), no Pydantic, no shared models. We assemble plain dicts going
out and parse plain dicts coming back — :mod:`lab.rag.rerank_server` owns
the Pydantic side. Keeping the client schema-free saves an import in the
hot path and avoids drift if the server adds fields the client doesn't need.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

#: When set, :class:`LabReranker` POSTs to this URL instead of loading the
#: cross-encoder in-process. Typically
#: ``http://host.containers.internal:8401`` from inside the agent sandbox.
URL_ENV_VAR = "LAB_RAG_RERANKER_URL"
#: Hard client-side per-request timeout (seconds). Keep slightly under the
#: server's :data:`lab.rag.rerank_server.DEFAULT_TIMEOUT_SEC` so the client
#: gives up first and returns a usable error rather than getting a 504.
CLIENT_TIMEOUT_ENV_VAR = "LAB_RAG_RERANKER_CLIENT_TIMEOUT_SEC"
DEFAULT_CLIENT_TIMEOUT_SEC = 28.0


class RerankClientError(RuntimeError):
    """Raised when the rerank service is unreachable or returns a 5xx."""


def _read_timeout() -> float:
    raw = os.environ.get(CLIENT_TIMEOUT_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_CLIENT_TIMEOUT_SEC
    try:
        v = float(raw)
        return v if v > 0 else DEFAULT_CLIENT_TIMEOUT_SEC
    except ValueError:
        return DEFAULT_CLIENT_TIMEOUT_SEC


def rerank_via_http(
    *,
    url: str,
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int,
    model: str | None = None,
    cache_key: tuple[str, int] | None = None,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """POST a rerank request and return the hit list.

    Returns ``[]`` on empty inputs without making a network call (parity with
    :meth:`LabReranker.rerank`'s short-circuit).

    Args:
        url: Full base URL of the rerank server, e.g. ``"http://host.containers.internal:8401"``.
            The ``/rerank`` suffix is appended automatically.
        query: Search query string.
        candidates: Candidate dicts; each must carry a ``text`` field.
        top_n: Maximum hits to return.
        model: Optional sanity-check — the server rejects with 409 if it
            doesn't match its configured model.
        cache_key: Optional ``(kb_version, top_k)`` tuple for the Phase 8
            tier-2 Valkey cache.
        timeout: Override the client-side per-request budget (seconds).

    Raises:
        RerankClientError: connection refused, timeout, or 5xx.
    """

    if top_n <= 0 or not candidates:
        return []
    base = url.rstrip("/")
    endpoint = f"{base}/rerank"
    payload: dict[str, Any] = {
        "query": query,
        "candidates": candidates,
        "top_n": top_n,
    }
    if model is not None:
        payload["model"] = model
    if cache_key is not None:
        # JSON has no tuple type; the server's Pydantic model coerces the
        # 2-element list back into a tuple.
        payload["cache_key"] = [cache_key[0], cache_key[1]]

    t = float(timeout) if timeout is not None else _read_timeout()
    try:
        with httpx.Client(timeout=t) as client:
            resp = client.post(endpoint, json=payload)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise RerankClientError(f"rerank service unreachable at {endpoint}: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise RerankClientError(f"rerank request timed out after {t:.1f}s") from exc
    except httpx.HTTPError as exc:
        raise RerankClientError(f"rerank HTTP error: {exc}") from exc

    if resp.status_code >= 500:
        raise RerankClientError(
            f"rerank server returned {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code >= 400:
        # 409 (model mismatch), 422 (validation), etc. — surface verbatim
        # so the caller's logs make the misconfig obvious.
        raise RerankClientError(
            f"rerank server rejected request ({resp.status_code}): {resp.text[:200]}"
        )

    try:
        body = resp.json()
    except ValueError as exc:
        raise RerankClientError(f"rerank server returned non-JSON body: {exc}") from exc

    hits = body.get("hits")
    if not isinstance(hits, list):
        raise RerankClientError(f"rerank server response missing 'hits': {body!r}")
    # The server returns plain dicts already; we just narrow the type for the
    # caller and refuse anything that isn't a dict (would crash downstream).
    out: list[dict[str, Any]] = []
    for h in hits:
        if not isinstance(h, dict):
            raise RerankClientError(f"rerank hit is not a dict: {h!r}")
        out.append(h)
    return out


def get_remote_url() -> str | None:
    """Return the configured rerank URL, or None if unset/empty.

    Centralised so ``LabReranker`` and tests share one parse path.
    """

    raw = os.environ.get(URL_ENV_VAR, "").strip()
    return raw or None
