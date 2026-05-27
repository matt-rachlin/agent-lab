"""Unit tests for lab.rag.rerank_server.

We never load a real cross-encoder — the singleton :class:`LabReranker` is
patched in via the same hook the existing ``test_rag_rerank`` suite uses.
FastAPI is driven via ``TestClient`` so we don't need a live uvicorn loop.
"""

from __future__ import annotations

import time
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient
from lab.rag import RERANKER_ENV_VAR
from lab.rag.rerank import LabReranker, reset_default_reranker
from lab.rag.rerank_client import URL_ENV_VAR
from lab.rag.rerank_server import create_app


class _FakeCrossEncoder:
    """Pre-seeded scores keyed by candidate text; counts ``predict`` calls."""

    instances: ClassVar[list[_FakeCrossEncoder]] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.scores_for_text: dict[str, float] = {}
        self.calls = 0
        _FakeCrossEncoder.instances.append(self)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls += 1
        return [float(self.scores_for_text.get(t, float(len(t)))) for _q, t in pairs]


@pytest.fixture
def fake_loaded_reranker(monkeypatch: pytest.MonkeyPatch) -> _FakeCrossEncoder:
    """Yield a LabReranker singleton with a pre-loaded fake cross-encoder."""

    # Make sure nothing in the env contaminates the test.
    monkeypatch.delenv(RERANKER_ENV_VAR, raising=False)
    monkeypatch.delenv(URL_ENV_VAR, raising=False)
    reset_default_reranker()
    _FakeCrossEncoder.instances.clear()

    fake = _FakeCrossEncoder("test-model")
    # Patch the singleton getter so the server's get_default_reranker()
    # returns a reranker with our fake injected.
    from lab.rag import rerank as rerank_mod

    r = LabReranker(model_name="test-model", idle_unload_sec=0)
    r._model = fake  # type: ignore[assignment]
    r._last_used = time.monotonic()
    monkeypatch.setattr(rerank_mod, "_DEFAULT_RERANKER", r)
    yield fake
    reset_default_reranker()


def test_healthz_returns_loaded_state(fake_loaded_reranker: _FakeCrossEncoder) -> None:
    with TestClient(create_app()) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["loaded"] is True
    assert body["model"] == "test-model"
    assert "idle_unload_sec" in body


def test_rerank_round_trip_returns_scored_hits(
    fake_loaded_reranker: _FakeCrossEncoder,
) -> None:
    fake_loaded_reranker.scores_for_text = {"alpha": 0.1, "beta": 0.9, "gamma": 0.5}
    payload = {
        "query": "q",
        "candidates": [
            {"chunk_id": "a", "text": "alpha"},
            {"chunk_id": "b", "text": "beta"},
            {"chunk_id": "c", "text": "gamma"},
        ],
        "top_n": 3,
    }
    with TestClient(create_app()) as client:
        resp = client.post("/rerank", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "test-model"
    assert isinstance(body["duration_ms"], int)
    hits = body["hits"]
    assert [h["chunk_id"] for h in hits] == ["b", "c", "a"]
    assert pytest.approx(hits[0]["rerank_score"]) == 0.9


def test_rerank_top_n_clamps(fake_loaded_reranker: _FakeCrossEncoder) -> None:
    fake_loaded_reranker.scores_for_text = {f"t{i}": float(i) for i in range(8)}
    payload = {
        "query": "q",
        "candidates": [{"chunk_id": f"c{i}", "text": f"t{i}"} for i in range(8)],
        "top_n": 2,
    }
    with TestClient(create_app()) as client:
        resp = client.post("/rerank", json=payload)
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "c7"
    assert hits[1]["chunk_id"] == "c6"


def test_rerank_rejects_model_mismatch(fake_loaded_reranker: _FakeCrossEncoder) -> None:
    payload = {
        "query": "q",
        "candidates": [{"chunk_id": "a", "text": "x"}],
        "top_n": 1,
        "model": "some-other-model",
    }
    with TestClient(create_app()) as client:
        resp = client.post("/rerank", json=payload)
    assert resp.status_code == 409
    assert "test-model" in resp.text


def test_unload_drops_model(fake_loaded_reranker: _FakeCrossEncoder) -> None:
    with TestClient(create_app()) as client:
        # Healthz reports loaded=True up front.
        assert client.get("/healthz").json()["loaded"] is True
        unload = client.post("/unload")
        assert unload.status_code == 200
        body = unload.json()
        assert body["unloaded"] is True
        assert body["model"] == "test-model"
        # And now it's not loaded.
        assert client.get("/healthz").json()["loaded"] is False


def test_metrics_endpoint_reports_counters(
    fake_loaded_reranker: _FakeCrossEncoder,
) -> None:
    fake_loaded_reranker.scores_for_text = {"x": 0.3}
    payload = {
        "query": "q",
        "candidates": [{"chunk_id": "a", "text": "x"}],
        "top_n": 1,
    }
    with TestClient(create_app()) as client:
        client.post("/rerank", json=payload)
        resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "lab_rerank_requests_total 1" in text
    assert 'lab_rerank_model_loaded{model="test-model"} 1' in text


def test_request_validation_rejects_missing_query(
    fake_loaded_reranker: _FakeCrossEncoder,
) -> None:
    """Pydantic should reject a body with no ``query`` field."""

    with TestClient(create_app()) as client:
        resp = client.post("/rerank", json={"candidates": [], "top_n": 1})
    assert resp.status_code == 422


def test_empty_candidates_returns_empty_hits(
    fake_loaded_reranker: _FakeCrossEncoder,
) -> None:
    """LabReranker short-circuits on empty inputs — no model call."""

    with TestClient(create_app()) as client:
        resp = client.post(
            "/rerank",
            json={"query": "q", "candidates": [], "top_n": 5},
        )
    assert resp.status_code == 200
    assert resp.json()["hits"] == []
    assert fake_loaded_reranker.calls == 0


def test_top_n_zero_returns_empty(fake_loaded_reranker: _FakeCrossEncoder) -> None:
    with TestClient(create_app()) as client:
        resp = client.post(
            "/rerank",
            json={
                "query": "q",
                "candidates": [{"chunk_id": "a", "text": "x"}],
                "top_n": 0,
            },
        )
    assert resp.status_code == 200
    assert resp.json()["hits"] == []
    assert fake_loaded_reranker.calls == 0
