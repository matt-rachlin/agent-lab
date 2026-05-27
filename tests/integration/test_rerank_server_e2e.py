"""End-to-end test for the host-side rerank HTTP service.

Skips cleanly when:
  * the service isn't reachable on the configured port (the test isn't allowed
    to start one — that's the systemd unit's job).
  * the bash KB is missing or empty (no realistic candidates to send).

When all gates pass, this fetches stage-1 candidates from the indexed bash
corpus, POSTs them to the live rerank server, and asserts that every returned
hit carries a ``rerank_score`` float.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from lab.rag.rerank_client import (
    DEFAULT_CLIENT_TIMEOUT_SEC,
    RerankClientError,
    rerank_via_http,
)
from lab.rag.rerank_server import DEFAULT_PORT, PORT_ENV_VAR


def _server_url() -> str:
    port = os.environ.get(PORT_ENV_VAR, str(DEFAULT_PORT))
    return f"http://127.0.0.1:{port}"


def _service_up(url: str) -> bool:
    import httpx

    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{url}/healthz")
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.integration
def test_rerank_server_e2e_or_skip() -> None:
    url = _server_url()
    if not _service_up(url):
        pytest.skip(f"rerank service not reachable at {url}")

    kb_dir = Path("~/db/kb/bash").expanduser()
    if not (kb_dir / "manifest.yaml").exists():
        pytest.skip(f"no bash KB at {kb_dir}")

    from lab.rag.index import count_rows

    if count_rows(kb_dir) == 0:
        pytest.skip("bash KB has no indexed chunks")

    # Pull stage-1 candidates host-side (no rerank), then send them over HTTP.
    from lab.rag.index import hybrid_query

    hits = hybrid_query(
        kb_dir,
        "redirect stderr to stdout",
        k=10,
        rerank=False,
    )
    if not hits:
        pytest.skip("no stage-1 hits — KB may be sparse")

    candidates = [
        {
            "chunk_id": h.chunk_id,
            "text": h.text or "",
            "score": float(h.score) if h.score is not None else 0.0,
        }
        for h in hits
    ]

    try:
        ranked = rerank_via_http(
            url=url,
            query="redirect stderr to stdout",
            candidates=candidates,
            top_n=5,
            timeout=DEFAULT_CLIENT_TIMEOUT_SEC,
        )
    except RerankClientError as exc:
        pytest.fail(f"rerank_via_http raised against live server: {exc}")

    assert ranked, "live rerank server returned no hits"
    assert len(ranked) <= 5
    for h in ranked:
        assert "rerank_score" in h
        assert isinstance(h["rerank_score"], float)
    # Scores should be sorted desc.
    scores = [h["rerank_score"] for h in ranked]
    assert scores == sorted(scores, reverse=True)
