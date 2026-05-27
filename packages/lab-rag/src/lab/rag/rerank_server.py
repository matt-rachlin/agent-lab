"""HTTP rerank service — host-side singleton wrapping :class:`LabReranker`.

The sandbox image no longer ships ``sentence-transformers`` / ``torch`` (see
``containers/Containerfile.agent-sandbox``). Instead, the sandbox calls this
service over the podman-managed ``host.containers.internal`` alias.

Design constraints:

* **Single-threaded model access.** The cross-encoder is the bottleneck; serial
  request handling avoids GPU thrash and keeps idle-TTL accounting honest.
  ``uvicorn`` is invoked with ``workers=1`` and we acquire a process-wide lock
  in :func:`_rerank_endpoint` before calling :meth:`LabReranker.rerank`.
* **127.0.0.1 only by default.** This is an admin surface. Bind to
  ``0.0.0.0`` only by setting ``LAB_RAG_RERANKER_HOST=0.0.0.0`` explicitly.
* **Hard per-request timeout.** ``LAB_RAG_RERANKER_TIMEOUT_SEC`` (default 30).
  The client side enforces the same; the server uses ``anyio.fail_after`` so a
  runaway predict() doesn't pin the queue indefinitely.

Run as a module:

    LAB_RAG_RERANKER=qwen3-reranker-0.6b \
    python -m lab.rag.rerank_server

Endpoints:

* ``POST /rerank`` — body :class:`RerankRequest`, returns :class:`RerankResponse`
* ``POST /unload`` — drop the cached cross-encoder (admin)
* ``GET  /healthz`` — 200 OK + ``loaded`` flag
* ``GET  /metrics`` — Prometheus text format (minimal)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from lab.rag.rerank import get_default_reranker, reset_default_reranker
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: Host bind address. Defaults to loopback — the sandbox reaches it via
#: ``host.containers.internal`` which podman maps to the host bridge IP.
HOST_ENV_VAR = "LAB_RAG_RERANKER_HOST"
DEFAULT_HOST = "127.0.0.1"

#: TCP port. Picked clear of common services (Postgres 5432, Valkey 6379,
#: Ollama 11434, LiteLLM 4000, MinIO 9000/9001, MLflow 5000, Grafana 3000).
PORT_ENV_VAR = "LAB_RAG_RERANKER_PORT"
DEFAULT_PORT = 8401

#: Hard per-request budget (seconds). The model is the bottleneck and a stuck
#: predict() should not pin the queue. The client should cap below this.
TIMEOUT_ENV_VAR = "LAB_RAG_RERANKER_TIMEOUT_SEC"
DEFAULT_TIMEOUT_SEC = 30.0


# ----------------------------------------------------------------------------
# Schemas — kept narrow so the client mirror in :mod:`lab.rag.rerank_client`
# can re-use them without dragging FastAPI into the sandbox.
# ----------------------------------------------------------------------------


class RerankCandidate(BaseModel):
    """One candidate going into the cross-encoder.

    Extra fields (``chunk_id``, ``score``, ``stage1_rank``, ``authority``…) are
    preserved verbatim into the response — the reranker only reads ``text``.
    """

    model_config = {"extra": "allow"}

    text: str = ""


class RerankRequest(BaseModel):
    """Wire shape for POST /rerank."""

    query: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    top_n: int = 10
    #: Optional override; when None the server's configured model is used.
    model: str | None = None
    #: Optional (kb_version, top_k) pair for tier-2 Valkey cache hits.
    cache_key: tuple[str, int] | None = None


class RerankResponse(BaseModel):
    """Wire shape for POST /rerank response — ``hits`` carries ``rerank_score``."""

    hits: list[dict[str, Any]]
    model: str
    duration_ms: int
    cache_hit: bool = False


class HealthResponse(BaseModel):
    """GET /healthz response."""

    ok: bool = True
    loaded: bool
    model: str
    idle_unload_sec: int


# ----------------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------------


class _Metrics:
    """Minimal Prometheus-format counters. Kept in-process; reset on restart."""

    def __init__(self) -> None:
        self.requests_total = 0
        self.errors_total = 0
        self.timeouts_total = 0
        self.cache_hits_total = 0
        self.duration_sum_ms = 0.0
        self.last_duration_ms = 0.0
        self.loads_total = 0
        self.unloads_total = 0

    def render(self, *, loaded: bool, model: str) -> str:
        lines = [
            "# HELP lab_rerank_requests_total Reranker requests.",
            "# TYPE lab_rerank_requests_total counter",
            f"lab_rerank_requests_total {self.requests_total}",
            "# HELP lab_rerank_errors_total Reranker server errors (5xx).",
            "# TYPE lab_rerank_errors_total counter",
            f"lab_rerank_errors_total {self.errors_total}",
            "# HELP lab_rerank_timeouts_total Per-request hard timeouts.",
            "# TYPE lab_rerank_timeouts_total counter",
            f"lab_rerank_timeouts_total {self.timeouts_total}",
            "# HELP lab_rerank_cache_hits_total Tier-2 cache hits served.",
            "# TYPE lab_rerank_cache_hits_total counter",
            f"lab_rerank_cache_hits_total {self.cache_hits_total}",
            "# HELP lab_rerank_duration_ms_sum Cumulative request latency (ms).",
            "# TYPE lab_rerank_duration_ms_sum counter",
            f"lab_rerank_duration_ms_sum {self.duration_sum_ms:.3f}",
            "# HELP lab_rerank_last_duration_ms Last request latency (ms).",
            "# TYPE lab_rerank_last_duration_ms gauge",
            f"lab_rerank_last_duration_ms {self.last_duration_ms:.3f}",
            "# HELP lab_rerank_model_loaded Cross-encoder resident in memory.",
            "# TYPE lab_rerank_model_loaded gauge",
            f'lab_rerank_model_loaded{{model="{model}"}} {1 if loaded else 0}',
            "# HELP lab_rerank_loads_total Cross-encoder load events.",
            "# TYPE lab_rerank_loads_total counter",
            f"lab_rerank_loads_total {self.loads_total}",
            "# HELP lab_rerank_unloads_total Cross-encoder unload events.",
            "# TYPE lab_rerank_unloads_total counter",
            f"lab_rerank_unloads_total {self.unloads_total}",
        ]
        return "\n".join(lines) + "\n"


def _read_timeout_sec() -> float:
    raw = os.environ.get(TIMEOUT_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SEC
    try:
        v = float(raw)
        return v if v > 0 else DEFAULT_TIMEOUT_SEC
    except ValueError:
        return DEFAULT_TIMEOUT_SEC


def create_app() -> FastAPI:
    """Build the FastAPI app. Factored so tests can drive it via TestClient."""

    metrics = _Metrics()
    # Serialise model access — the GPU bound bottleneck means a queue beats
    # concurrent predict() calls fighting for VRAM.
    model_lock = asyncio.Lock()
    timeout_sec = _read_timeout_sec()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Don't preload the model — let the first /rerank call lazy-load so the
        # service can start even when the GPU is busy with a sweep.
        logger.info(
            "rerank-server lifespan up: timeout=%.1fs, port=%s",
            timeout_sec,
            os.environ.get(PORT_ENV_VAR, str(DEFAULT_PORT)),
        )
        try:
            yield
        finally:
            try:
                reset_default_reranker()
            except Exception:
                logger.warning("reset_default_reranker failed on shutdown", exc_info=True)

    app = FastAPI(title="lab-rerank-server", lifespan=lifespan)

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        r = get_default_reranker()
        return HealthResponse(
            ok=True,
            loaded=r.loaded,
            model=r.model_name,
            idle_unload_sec=r.idle_unload_sec,
        )

    @app.post("/unload")
    async def unload() -> dict[str, Any]:
        async with model_lock:
            r = get_default_reranker()
            was_loaded = r.loaded
            r.unload()
            if was_loaded:
                metrics.unloads_total += 1
            return {"unloaded": was_loaded, "model": r.model_name}

    @app.post("/rerank", response_model=RerankResponse)
    async def rerank_endpoint(req: RerankRequest) -> RerankResponse:
        metrics.requests_total += 1
        t0 = time.perf_counter()
        try:
            r = get_default_reranker()
            # The wire `model` override is ignored if it would force a swap;
            # the server is configured at startup for one model. We only
            # accept the override when it matches the configured model, so
            # callers can self-verify the singleton without surprises.
            if req.model and req.model != r.model_name:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"server configured for model {r.model_name!r}; "
                        f"request asked for {req.model!r}"
                    ),
                )
            was_loaded = r.loaded
            async with model_lock:
                # anyio.fail_after raises TimeoutError if the work exceeds the
                # budget — caught below and surfaced as 504.
                try:
                    with anyio.fail_after(timeout_sec):
                        hits = await asyncio.to_thread(
                            r.rerank,
                            req.query,
                            req.candidates,
                            req.top_n,
                            cache_key=req.cache_key,
                        )
                except TimeoutError as exc:
                    metrics.timeouts_total += 1
                    raise HTTPException(
                        status_code=504,
                        detail=f"rerank exceeded {timeout_sec:.1f}s",
                    ) from exc
                if not was_loaded and r.loaded:
                    metrics.loads_total += 1
            duration_ms = (time.perf_counter() - t0) * 1000.0
            metrics.duration_sum_ms += duration_ms
            metrics.last_duration_ms = duration_ms
            # A pass-through (disabled) reranker never adds rerank_score; we
            # report cache_hit=False there too so observability stays honest.
            cache_hit = bool(
                req.cache_key is not None
                and hits
                and all("rerank_score" in h for h in hits)
                and not was_loaded
                and not r.loaded
            )
            if cache_hit:
                metrics.cache_hits_total += 1
            return RerankResponse(
                hits=hits,
                model=r.model_name,
                duration_ms=int(duration_ms),
                cache_hit=cache_hit,
            )
        except HTTPException:
            raise
        except Exception as exc:
            metrics.errors_total += 1
            logger.exception("rerank request failed")
            raise HTTPException(status_code=500, detail=f"rerank failed: {exc}") from exc

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint() -> str:
        r = get_default_reranker()
        return metrics.render(loaded=r.loaded, model=r.model_name)

    return app


# Module-level app so `uvicorn lab.rag.rerank_server:app` works in addition
# to `python -m lab.rag.rerank_server`.
app = create_app()


def main() -> None:  # pragma: no cover - entrypoint
    """CLI: bind + serve. Called by ``python -m lab.rag.rerank_server``."""

    import uvicorn

    host = os.environ.get(HOST_ENV_VAR, DEFAULT_HOST).strip() or DEFAULT_HOST
    port_raw = os.environ.get(PORT_ENV_VAR, "").strip()
    try:
        port = int(port_raw) if port_raw else DEFAULT_PORT
    except ValueError:
        port = DEFAULT_PORT
    logging.basicConfig(
        level=os.environ.get("LAB_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    # workers=1 is load-bearing: serialise model access. uvicorn's --workers
    # would multiply the model in memory, blowing the 12 GB VRAM budget the
    # whole point of this service is to share.
    uvicorn.run(
        "lab.rag.rerank_server:app",
        host=host,
        port=port,
        workers=1,
        log_level=os.environ.get("LAB_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
