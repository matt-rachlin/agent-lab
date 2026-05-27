"""Unit tests for lab.rag.rerank_client.

We mock the network with ``httpx.MockTransport`` so no real socket opens. The
client is intentionally stdlib-typed (plain dicts in/out), so the assertions
shape-check the JSON wire format directly.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from lab.rag import rerank_client
from lab.rag.rerank_client import (
    DEFAULT_CLIENT_TIMEOUT_SEC,
    URL_ENV_VAR,
    RerankClientError,
    get_remote_url,
    rerank_via_http,
)


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: httpx.MockTransport,
) -> None:
    """Patch httpx.Client so every constructed client uses our mock."""

    real_init = httpx.Client.__init__

    def _wrapped_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = handler
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", _wrapped_init)


def test_get_remote_url_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(URL_ENV_VAR, raising=False)
    assert get_remote_url() is None
    monkeypatch.setenv(URL_ENV_VAR, "  http://host.containers.internal:8401  ")
    assert get_remote_url() == "http://host.containers.internal:8401"
    monkeypatch.setenv(URL_ENV_VAR, "")
    assert get_remote_url() is None


def test_posts_expected_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "hits": [{"chunk_id": "x", "text": "y", "rerank_score": 0.42}],
                "model": "test-model",
                "duration_ms": 12,
                "cache_hit": False,
            },
        )

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    out = rerank_via_http(
        url="http://host.containers.internal:8401",
        query="q",
        candidates=[{"chunk_id": "x", "text": "y"}],
        top_n=3,
        model="test-model",
        cache_key=("v1", 50),
    )

    assert captured["url"].endswith("/rerank")
    body = captured["body"]
    assert body["query"] == "q"
    assert body["top_n"] == 3
    assert body["model"] == "test-model"
    # cache_key crosses the wire as a 2-element list (JSON has no tuple).
    assert body["cache_key"] == ["v1", 50]
    assert out == [{"chunk_id": "x", "text": "y", "rerank_score": 0.42}]


def test_5xx_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(RerankClientError) as exc:
        rerank_via_http(
            url="http://h",
            query="q",
            candidates=[{"chunk_id": "x", "text": "y"}],
            top_n=1,
        )
    assert "503" in str(exc.value)


def test_4xx_surfaces_with_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """409 (model mismatch) should be raised so the caller logs it."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, text="wrong model")

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(RerankClientError) as exc:
        rerank_via_http(
            url="http://h",
            query="q",
            candidates=[{"chunk_id": "x", "text": "y"}],
            top_n=1,
        )
    assert "409" in str(exc.value)


def test_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(RerankClientError) as exc:
        rerank_via_http(
            url="http://h",
            query="q",
            candidates=[{"chunk_id": "x", "text": "y"}],
            top_n=1,
            timeout=0.5,
        )
    assert "timed out" in str(exc.value).lower()


def test_connect_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(RerankClientError) as exc:
        rerank_via_http(
            url="http://h",
            query="q",
            candidates=[{"chunk_id": "x", "text": "y"}],
            top_n=1,
        )
    assert "unreachable" in str(exc.value).lower()


def test_empty_inputs_short_circuit_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty candidates / top_n=0 must not hit the network."""

    called = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        called["n"] += 1
        return httpx.Response(200, json={"hits": [], "model": "m", "duration_ms": 0})

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    assert rerank_via_http(url="http://h", query="q", candidates=[], top_n=5) == []
    assert (
        rerank_via_http(
            url="http://h",
            query="q",
            candidates=[{"chunk_id": "x", "text": "y"}],
            top_n=0,
        )
        == []
    )
    assert called["n"] == 0


def test_default_client_timeout_constant_sane() -> None:
    """Guard: client timeout MUST be < server's 30s to surface 4xx cleanly."""

    assert 0 < DEFAULT_CLIENT_TIMEOUT_SEC < 30.0


def test_client_timeout_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_RAG_RERANKER_CLIENT_TIMEOUT_SEC", "5.5")
    # The internal reader is module-private; exercise indirectly via the value.
    assert rerank_client._read_timeout() == 5.5
    monkeypatch.setenv("LAB_RAG_RERANKER_CLIENT_TIMEOUT_SEC", "garbage")
    assert rerank_client._read_timeout() == DEFAULT_CLIENT_TIMEOUT_SEC
