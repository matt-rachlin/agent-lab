"""Unit tests for lab.rag.embedder.

Covers: tokenize_for_bm25 shape, embed_texts batching/fallback behaviour
(mocked Ollama client), BM25 sparse-vector contributions, normalize_scores,
cosine. The retry decorator is exercised via the fallback path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from lab.rag.embedder import (
    DEFAULT_BATCH,
    EmbedResult,
    build_bm25,
    cosine,
    embed_texts,
    normalize_scores,
    sparse_for_text,
    tokenize_for_bm25,
)


def test_tokenize_for_bm25_basic():
    toks = tokenize_for_bm25("Hello, world! foo_bar 42")
    assert "hello" in toks
    assert "world" in toks
    assert "foo_bar" in toks
    assert "42" in toks
    # All tokens are lower-cased
    for t in toks:
        assert t == t.lower()


def test_tokenize_for_bm25_empty():
    assert tokenize_for_bm25("") == []
    assert tokenize_for_bm25("   ") == []


def test_normalize_scores_basic():
    assert normalize_scores([]) == []
    assert normalize_scores([5.0]) == [0.0]  # min == max
    out = normalize_scores([1.0, 2.0, 3.0])
    assert out == [0.0, 0.5, 1.0]


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero-norm fallback


def test_build_bm25_and_sparse_for_text():
    """rank_bm25's IDF can be negative for terms common across most of the
    corpus; we only assert structural properties here."""
    corpus = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
    bm, tokenized = build_bm25(corpus)
    assert len(tokenized) == 3
    sparse = sparse_for_text(bm, tokenized[0])
    # Keys are a subset of the document's tokens.
    assert set(sparse).issubset(set(tokenized[0]))
    # Every key maps to a finite float.
    for v in sparse.values():
        assert isinstance(v, float)
    # Dropping in an empty doc returns {}
    assert sparse_for_text(bm, []) == {}


def test_embed_texts_happy_path():
    """Mock the Ollama client and verify the result shape + batching."""
    fake_client = MagicMock()
    fake_client.embed.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]] * 5  # one vector per input
    }

    texts = ["a"] * 17  # 3 full batches of DEFAULT_BATCH + a tail
    with patch("lab.rag.embedder.Client", return_value=fake_client):
        result = embed_texts(texts, batch_size=DEFAULT_BATCH)
    assert isinstance(result, EmbedResult)
    # Number of calls = ceil(17 / 8) = 3
    assert fake_client.embed.call_count == 3
    # First call sees a batch of 8
    first_call_input = fake_client.embed.call_args_list[0].kwargs.get("input") or (
        fake_client.embed.call_args_list[0].args[0]
        if fake_client.embed.call_args_list[0].args
        else None
    )
    # The function passes input= kwarg
    assert first_call_input == texts[:DEFAULT_BATCH]


def test_embed_texts_fallback_on_first_batch_failure():
    """If the primary model fails, embed_texts should fall back to the 4B model."""
    fake_client = MagicMock()
    calls: list[str] = []

    def side_effect(*, model: str, input: list[str]):
        calls.append(model)
        if model == "qwen3-embedding:8b-q8_0":
            raise RuntimeError("model not loaded")
        return {"embeddings": [[0.0] * 2560 for _ in input]}

    fake_client.embed.side_effect = side_effect
    with patch("lab.rag.embedder.Client", return_value=fake_client):
        result = embed_texts(["x", "y"], batch_size=2)
    assert result.model == "qwen3-embedding:4b"
    assert result.dimensions == 2560
    # First attempt was the primary; subsequent calls used fallback
    assert calls[0] == "qwen3-embedding:8b-q8_0"
    assert any(c == "qwen3-embedding:4b" for c in calls)


def test_embed_texts_retries_then_succeeds():
    """The tenacity retry on transient errors should give us a successful
    result even if the underlying client raises a few times first."""
    fake_client = MagicMock()
    attempts = {"n": 0}

    def side_effect(*, model: str, input: list[str]):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("temporary")
        return {"embeddings": [[0.0] * 4096 for _ in input]}

    fake_client.embed.side_effect = side_effect
    with patch("lab.rag.embedder.Client", return_value=fake_client):
        result = embed_texts(["a"], batch_size=1)
    # The retry layer in _embed_one_batch absorbed the transient error
    assert result.dimensions == 4096
    assert len(result.vectors) == 1


def test_embed_texts_irrecoverable_after_fallback():
    """If both primary and fallback fail, embed_texts must raise RuntimeError."""
    fake_client = MagicMock()
    fake_client.embed.side_effect = RuntimeError("dead")
    with (
        patch("lab.rag.embedder.Client", return_value=fake_client),
        pytest.raises(RuntimeError),
    ):
        embed_texts(["x"], batch_size=1)
