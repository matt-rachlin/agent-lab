"""Ollama embeddings + BM25 sparse, with VRAM-aware batching.

Vendored from kb_builder.embedder.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

from ollama import Client
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential

from lab.rag import (
    DEFAULT_EMBED_DIMS,
    DEFAULT_EMBED_MODEL,
    FALLBACK_EMBED_DIMS,
    FALLBACK_EMBED_MODEL,
)

DEFAULT_BATCH = 8  # conservative for 8B Q8 on a 12 GB GPU


@dataclass
class EmbedResult:
    model: str
    dimensions: int
    vectors: list[list[float]]


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def tokenize_for_bm25(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.strip()]


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _embed_one_batch(client: Client, model: str, texts: list[str]) -> list[list[float]]:
    resp = client.embed(model=model, input=texts)
    return list(resp["embeddings"])


def embed_texts(
    texts: list[str],
    *,
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = DEFAULT_BATCH,
    progress: Callable[[int, int], None] | None = None,
) -> EmbedResult:
    """Embed a list of texts via Ollama. Auto-fallback to the 4B model if loading fails.

    Honours the standard ``OLLAMA_HOST`` env var (default
    ``http://localhost:11434``). The kb_query MCP tool runs inside the agent
    sandbox where ``localhost`` is the container, not the host — the
    harness sets ``OLLAMA_HOST=http://host.containers.internal:11434`` so
    the in-sandbox embedder reaches the host's Ollama. On the host (no env
    var set) we keep the prior behaviour.
    """
    import os
    client = Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    chosen = model
    dims: int | None = DEFAULT_EMBED_DIMS if model == DEFAULT_EMBED_MODEL else None

    out: list[list[float]] = []
    i = 0
    while i < len(texts):
        batch = texts[i : i + batch_size]
        try:
            vecs = _embed_one_batch(client, chosen, batch)
        except Exception as e:
            if chosen == DEFAULT_EMBED_MODEL:
                # fall back
                chosen = FALLBACK_EMBED_MODEL
                dims = FALLBACK_EMBED_DIMS
                vecs = _embed_one_batch(client, chosen, batch)
            else:
                raise RuntimeError(f"embedding failed irrecoverably: {e}") from e
        if dims is None and vecs:
            dims = len(vecs[0])
        out.extend(vecs)
        i += batch_size
        if progress:
            progress(min(i, len(texts)), len(texts))

    if dims is None:
        dims = 0
    return EmbedResult(model=chosen, dimensions=dims, vectors=out)


def build_bm25(corpus_texts: list[str]) -> tuple[BM25Okapi, list[list[str]]]:
    tokenized = [tokenize_for_bm25(t) for t in corpus_texts]
    bm = BM25Okapi(tokenized)
    return bm, tokenized


def sparse_for_text(bm: BM25Okapi, tokens: list[str]) -> dict[str, float]:
    """Compute the BM25 'self-score' contributions for each term in the chunk's tokens
    as a sparse vector. We store {term: idf*tf_norm}. At query time we score by
    overlap with the query's tokens.
    """
    # rank_bm25 doesn't directly expose per-term contributions; compute manually
    # using its internal stats.
    avgdl = bm.avgdl
    k1 = bm.k1
    b = bm.b
    idf = bm.idf  # dict
    dl = len(tokens)
    if dl == 0:
        return {}
    from collections import Counter

    tf = Counter(tokens)
    out: dict[str, float] = {}
    for term, freq in tf.items():
        if term not in idf:
            continue
        denom = freq + k1 * (1.0 - b + b * dl / avgdl)
        out[term] = idf[term] * (freq * (k1 + 1.0)) / denom if denom else 0.0
    return out


def query_sparse(bm: BM25Okapi, query_text: str, doc_tokens_list: list[list[str]]) -> list[float]:
    """Return BM25 score per doc for a query."""
    q = tokenize_for_bm25(query_text)
    scores: list[float] = bm.get_scores(q).tolist()
    return scores


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    mx = max(scores)
    mn = min(scores)
    if mx <= mn:
        return [0.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def cosine(a: list[float], b: list[float]) -> float:
    s = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return s / (na * nb)
