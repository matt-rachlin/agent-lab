"""Unit tests for lab.rag.rerank.LabReranker.

We never load a real cross-encoder — every test stubs the CrossEncoder via
the singleton's internal state or by monkeypatching the loader hook. That
keeps these fast and CPU-only, while exercising ordering, top-n clamping,
the disabled pass-through, and idle-unload behaviour.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest

from lab.rag import (
    DEFAULT_RERANKER_MODEL,
    RERANKER_ENV_VAR,
)
from lab.rag.rerank import (
    RERANKER_DISABLED,
    LabReranker,
    get_default_reranker,
    reset_default_reranker,
)


class _FakeCrossEncoder:
    """Stand-in for sentence-transformers.CrossEncoder.

    ``predict`` returns the pre-seeded score for each (query, text) pair —
    or ``len(text)`` as a last-resort fallback if the test didn't pre-seed.
    Counts calls so tests can assert no-double-loads.
    """

    instances: ClassVar[list[_FakeCrossEncoder]] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[list[tuple[str, str]]] = []
        self.scores_for_text: dict[str, float] = {}
        _FakeCrossEncoder.instances.append(self)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        out: list[float] = []
        for _q, t in pairs:
            out.append(float(self.scores_for_text.get(t, float(len(t)))))
        return out


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> Any:
    reset_default_reranker()
    monkeypatch.delenv(RERANKER_ENV_VAR, raising=False)
    _FakeCrossEncoder.instances.clear()
    yield
    reset_default_reranker()


def _patch_load(reranker: LabReranker, fake: _FakeCrossEncoder) -> None:
    """Inject a fake model directly, skipping the real import path."""
    reranker._model = fake  # type: ignore[assignment]
    reranker._last_used = time.monotonic()


def test_env_var_sets_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RERANKER_ENV_VAR, "BAAI/bge-reranker-v2-m3")
    r = LabReranker()
    assert r.model_name == "BAAI/bge-reranker-v2-m3"


def test_default_model_when_env_unset() -> None:
    r = LabReranker()
    assert r.model_name == DEFAULT_RERANKER_MODEL
    assert not r.disabled


def test_disabled_passthrough_clamps_top_n(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RERANKER_ENV_VAR, RERANKER_DISABLED)
    r = LabReranker()
    assert r.disabled is True

    cands = [{"chunk_id": f"c{i}", "text": f"doc {i}"} for i in range(15)]
    out = r.rerank("anything", cands, top_n=5)
    assert len(out) == 5
    # Pass-through preserves original ordering and does NOT add rerank_score.
    assert [c["chunk_id"] for c in out] == [f"c{i}" for i in range(5)]
    assert all("rerank_score" not in c for c in out)
    # And critically: no model load.
    assert r.loaded is False


def test_rerank_orders_by_score_desc() -> None:
    r = LabReranker(idle_unload_sec=0)  # no reaper interference
    fake = _FakeCrossEncoder("dummy")
    fake.scores_for_text = {"alpha": 0.10, "beta": 0.95, "gamma": 0.50}
    _patch_load(r, fake)

    cands = [
        {"chunk_id": "a", "text": "alpha"},
        {"chunk_id": "b", "text": "beta"},
        {"chunk_id": "c", "text": "gamma"},
    ]
    out = r.rerank("q", cands, top_n=3)
    assert [c["chunk_id"] for c in out] == ["b", "c", "a"]
    assert pytest.approx(out[0]["rerank_score"]) == 0.95
    assert pytest.approx(out[2]["rerank_score"]) == 0.10
    # Pairs were exactly the cross of query + each candidate's text, in order.
    assert fake.calls[-1] == [("q", "alpha"), ("q", "beta"), ("q", "gamma")]


def test_rerank_clamps_to_top_n() -> None:
    r = LabReranker(idle_unload_sec=0)
    fake = _FakeCrossEncoder("dummy")
    fake.scores_for_text = {f"t{i}": float(i) for i in range(10)}
    _patch_load(r, fake)

    cands = [{"chunk_id": f"c{i}", "text": f"t{i}"} for i in range(10)]
    out = r.rerank("q", cands, top_n=3)
    assert len(out) == 3
    # Highest-score first.
    assert out[0]["chunk_id"] == "c9"
    assert out[2]["chunk_id"] == "c7"


def test_rerank_empty_and_zero_top_n() -> None:
    r = LabReranker(idle_unload_sec=0)
    fake = _FakeCrossEncoder("dummy")
    _patch_load(r, fake)
    assert r.rerank("q", [], top_n=5) == []
    assert r.rerank("q", [{"chunk_id": "x", "text": "y"}], top_n=0) == []


def test_idle_unload_releases_model() -> None:
    r = LabReranker(idle_unload_sec=1)
    fake = _FakeCrossEncoder("dummy")
    _patch_load(r, fake)
    # Force last_used way into the past so the reaper fires.
    r._last_used = time.monotonic() - 10.0
    assert r.loaded is True
    # Calling rerank with a fresh fake-load won't unload because _ensure_loaded
    # is called next — so we trigger the reaper directly.
    r._maybe_unload()
    assert r.loaded is False


def test_unload_is_idempotent() -> None:
    r = LabReranker()
    r.unload()  # no-op
    r.unload()
    assert r.loaded is False


def test_default_singleton_reused() -> None:
    a = get_default_reranker()
    b = get_default_reranker()
    assert a is b
    reset_default_reranker()
    c = get_default_reranker()
    assert c is not a
