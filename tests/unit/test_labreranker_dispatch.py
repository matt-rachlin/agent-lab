"""Decision-tree tests for LabReranker.rerank:

1. ``LAB_RAG_RERANKER="none"``     → pass-through, no network, no model load.
2. ``LAB_RAG_RERANKER_URL`` set    → HTTP client path, no in-process model load.
3. Neither set                     → load model in-process (existing behaviour).

The cross-encoder loader is stubbed throughout — none of these tests touch
sentence-transformers or torch.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest
from lab.rag import RERANKER_ENV_VAR
from lab.rag.rerank import LabReranker, reset_default_reranker
from lab.rag.rerank_client import URL_ENV_VAR


class _FakeCrossEncoder:
    instances: ClassVar[list[_FakeCrossEncoder]] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[list[tuple[str, str]]] = []
        _FakeCrossEncoder.instances.append(self)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return [float(len(t)) for _q, t in pairs]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv(RERANKER_ENV_VAR, raising=False)
    monkeypatch.delenv(URL_ENV_VAR, raising=False)
    reset_default_reranker()
    _FakeCrossEncoder.instances.clear()
    yield
    reset_default_reranker()


def test_disabled_short_circuits_before_url_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LAB_RAG_RERANKER=none`` wins even when URL is also set — defensive."""

    monkeypatch.setenv(RERANKER_ENV_VAR, "none")
    monkeypatch.setenv(URL_ENV_VAR, "http://should-not-be-hit:1")

    # If the URL path WERE taken we'd raise ConnectError. With the disabled
    # short-circuit, no network call is made and we get pass-through.
    r = LabReranker()
    cands = [{"chunk_id": "a", "text": "x"}, {"chunk_id": "b", "text": "y"}]
    out = r.rerank("q", cands, top_n=2)
    assert [c["chunk_id"] for c in out] == ["a", "b"]
    # No rerank_score injected on the pass-through path.
    assert all("rerank_score" not in c for c in out)
    assert r.loaded is False


def test_url_set_dispatches_via_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """When URL is set, LabReranker.rerank delegates to rerank_via_http."""

    monkeypatch.setenv(URL_ENV_VAR, "http://host.containers.internal:8401")

    captured: dict[str, Any] = {}

    def _fake_http(
        *,
        url: str,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
        model: str | None = None,
        cache_key: tuple[str, int] | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        captured.update(
            url=url,
            query=query,
            top_n=top_n,
            model=model,
            cache_key=cache_key,
            n_candidates=len(candidates),
        )
        return [
            {"chunk_id": "b", "text": "beta", "rerank_score": 0.9},
            {"chunk_id": "a", "text": "alpha", "rerank_score": 0.1},
        ]

    monkeypatch.setattr("lab.rag.rerank_client.rerank_via_http", _fake_http)

    # Constructor must not need a real model — set idle_unload=0 so the reaper
    # is inert. The class should never even try to import sentence_transformers.
    r = LabReranker(model_name="test-model", idle_unload_sec=0)
    cands = [
        {"chunk_id": "a", "text": "alpha"},
        {"chunk_id": "b", "text": "beta"},
    ]
    out = r.rerank("q", cands, top_n=2, cache_key=("v1", 5))

    assert captured["url"] == "http://host.containers.internal:8401"
    assert captured["query"] == "q"
    assert captured["top_n"] == 2
    assert captured["model"] == "test-model"
    assert captured["cache_key"] == ("v1", 5)
    assert captured["n_candidates"] == 2
    # No in-process model was loaded.
    assert r.loaded is False
    assert [h["chunk_id"] for h in out] == ["b", "a"]


def test_neither_set_loads_in_process_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy in-process path still works when no URL is configured."""

    # Patch the import in _ensure_loaded so we never touch real ST.
    import lab.rag.rerank as rerank_mod

    class _SentenceTransformersStub:
        CrossEncoder = _FakeCrossEncoder

    monkeypatch.setattr(
        rerank_mod,
        "__builtins__",
        {**__builtins__, "__import__": __builtins__["__import__"]},  # type: ignore[index]
        raising=False,
    )

    # Easier: patch the loader hook directly. We bypass _ensure_loaded by
    # pre-injecting the fake — that's exactly what the legacy unit tests do.
    fake = _FakeCrossEncoder("test-model")
    r = LabReranker(model_name="test-model", idle_unload_sec=0)
    r._model = fake  # type: ignore[assignment]
    r._last_used = time.monotonic()

    cands = [
        {"chunk_id": "a", "text": "alpha"},
        {"chunk_id": "b", "text": "beta-longer"},
    ]
    out = r.rerank("q", cands, top_n=2)
    # _FakeCrossEncoder.predict returns len(text) — so "beta-longer" wins.
    assert [c["chunk_id"] for c in out] == ["b", "a"]
    assert all("rerank_score" in c for c in out)
    assert fake.calls, "fake cross-encoder predict() should have been called"
