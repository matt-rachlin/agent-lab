"""`kb_query` — hybrid retrieval over a lab knowledge base under `LAB_KB_ROOT`.

Runs as a FastMCP stdio server, same shape as the other lab agent tools:

    python -m lab.agent.tools.kb_query

When this tool is part of a task's tool list, the harness bind-mounts the
host `~/db/kb/` directory read-only into the sandbox (default `/kb`, per
`LAB_KB_ROOT`). The vendored `lab.rag` modules then read manifests and
LanceDB indices straight off that mount and embed the query via the Ollama
endpoint reachable from inside the sandbox.

Failure modes:
  * KB doesn't exist at `<LAB_KB_ROOT>/<kb_name>/`: return
    `{"hits": [], "kb_status": "missing"}` — never raise.
  * KB exists but the index is empty (e.g. `enrichment_pending`): return
    `{"hits": [], "kb_status": "empty"}`.
  * Embedding service / DB throws mid-query: return
    `{"hits": [], "error": "..."}`.

The model should always see a clean dict; missing infra is a result, not an
exception, so the agent can reason about it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp: FastMCP = FastMCP("lab.kb_query")

#: Maximum chars of `text` returned per hit. Anything longer is truncated and
#: flagged. Keeps the model's context cheap while still leaving room for full
#: passages on most KB chunks (target ~512 tokens after chunking).
MAX_TEXT_CHARS = 1500

#: KB names live on the filesystem next to other KBs; keep the set tight so
#: a malicious `kb_name` can't escape `LAB_KB_ROOT`.
_KB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _resolve_kb_root() -> Path:
    """Resolve the KB root directory.

    Inside the sandbox the harness sets `LAB_KB_ROOT=/kb` (a read-only
    bind-mount of the host `~/db/kb/`). On the host, the env var either points
    at `~/db/kb` directly or is unset, in which case we fall back to the
    standard location.
    """

    raw = os.environ.get("LAB_KB_ROOT", "~/db/kb")
    return Path(raw).expanduser()


@mcp.tool()
def kb_query(
    kb_name: str,
    question: str,
    k: int = 5,
    alpha: float | None = None,
    authority: str | None = None,
    rerank: bool = True,
    fusion: str = "rrf",
) -> dict[str, Any]:
    """Search a lab knowledge base for relevant passages.

    Args:
        kb_name: Name of the KB under `LAB_KB_ROOT` (e.g. ``"bash"``). Must
            match ``[A-Za-z0-9][A-Za-z0-9._-]*`` — no path separators or
            traversal.
        question: Natural-language query. Embedded by the same model the KB
            was built with; passed through BM25 tokenisation for the sparse
            half of the hybrid score.
        k: Maximum number of results to return (default 5). Clamped to
            ``[1, 50]``.
        alpha: Legacy hybrid weight for alpha-blend fusion. ``0.0`` = pure
            BM25 sparse, ``1.0`` = pure dense. Pass this only when running
            an ablation against the legacy fusion strategy; otherwise omit
            and the default ``fusion="rrf"`` (rank fusion) wins.
        authority: Optional filter on the source's authority tag (e.g.
            ``"official"`` to restrict to authoritative documentation).
        rerank: Run the stage-2 cross-encoder reranker on the stage-1
            candidates (default True). Set False to compare retrieval-only
            quality. Honoured per call; the env var ``LAB_RAG_RERANKER=none``
            disables it process-wide.
        fusion: Stage-1 fusion strategy: ``"rrf"`` (rank-based, default) or
            ``"alpha"`` (legacy alpha-blend; requires ``alpha=...``).

    Returns:
        ``{"hits": [{chunk_id, source_url, section_path, text, score,
        rerank_score, stage1_rank, ...}]}`` on success. ``hits`` is always a
        list; each element carries an explicit ``truncated`` flag when the
        returned ``text`` was capped at :data:`MAX_TEXT_CHARS`. On
        KB-missing/empty/error paths returns the same shape with an empty
        list plus a ``kb_status`` or ``error`` key — never raises.
    """

    if not isinstance(kb_name, str) or not _KB_NAME_RE.match(kb_name):
        return {
            "hits": [],
            "error": f"invalid kb_name {kb_name!r}: must match {_KB_NAME_RE.pattern}",
        }
    if not isinstance(question, str) or not question.strip():
        return {"hits": [], "error": "question must be a non-empty string"}
    k = max(1, min(int(k), 50))
    alpha_val: float | None = None
    if alpha is not None:
        try:
            alpha_val = float(alpha)
        except (TypeError, ValueError):
            return {"hits": [], "error": f"alpha must be a number, got {alpha!r}"}
        if not (0.0 <= alpha_val <= 1.0):
            return {"hits": [], "error": f"alpha must be in [0, 1], got {alpha!r}"}
    if fusion not in ("rrf", "alpha"):
        return {"hits": [], "error": f"fusion must be 'rrf' or 'alpha', got {fusion!r}"}

    kb_root = _resolve_kb_root()
    kb_dir = kb_root / kb_name
    try:
        manifest_exists = (kb_dir / "manifest.yaml").is_file()
    except PermissionError as exc:
        # Rootless podman + host file modes may block the sandbox user from
        # stat-ing the KB even with a read-only bind mount. Surface as a
        # clean error string so the model can reason about it.
        return {
            "hits": [],
            "error": f"permission denied reading {kb_dir}: {exc}",
            "kb_dir": str(kb_dir),
        }
    if not manifest_exists:
        return {"hits": [], "kb_status": "missing", "kb_dir": str(kb_dir)}

    # Lazy import: pulls in ollama/lancedb/pyarrow, which is expensive and
    # only meaningful when the KB is actually queryable.
    try:
        from lab.rag.index import count_rows, hybrid_query
    except Exception as exc:  # pragma: no cover - defensive
        return {"hits": [], "error": f"rag modules unavailable: {exc}"}

    try:
        row_count = count_rows(kb_dir)
    except PermissionError as exc:
        return {
            "hits": [],
            "error": f"permission denied reading index under {kb_dir}: {exc}",
            "kb_dir": str(kb_dir),
        }
    if row_count == 0:
        return {"hits": [], "kb_status": "empty", "kb_dir": str(kb_dir)}

    try:
        raw_hits = hybrid_query(
            kb_dir,
            question,
            k=k,
            fusion=fusion,  # type: ignore[arg-type]
            rerank=bool(rerank),
            alpha=alpha_val,
            filter_authority=authority,
        )
    except Exception as exc:
        return {"hits": [], "error": f"hybrid_query failed: {exc}"}

    out: list[dict[str, Any]] = []
    for h in raw_hits:
        text = h.text or ""
        truncated = len(text) > MAX_TEXT_CHARS
        if truncated:
            text = text[:MAX_TEXT_CHARS]
        out.append(
            {
                "chunk_id": h.chunk_id,
                "source_url": h.source_url,
                "section_path": list(h.section_path),
                "title": h.title,
                "summary": h.summary,
                "text": text,
                "truncated": truncated,
                "score": h.score,
                "dense_score": h.dense_score,
                "sparse_score": h.sparse_score,
                "rerank_score": h.rerank_score,
                "stage1_rank": h.stage1_rank,
                "retrieved_at": h.retrieved_at,
                "authority": h.authority,
            }
        )
    return {"hits": out, "kb_status": "ok", "kb_dir": str(kb_dir)}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
