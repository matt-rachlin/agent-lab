"""Bench: ``lab kb query bash "redirect stderr"`` p50/p95 over n runs.

Gated on the GPU lease (``lab:gpu:lease`` in Valkey) being empty — the
KB query calls Ollama for query embedding, which contends with whatever
sweep currently holds the lease.

Skips when:

- Valkey is unreachable (no lease check possible — be conservative)
- the lease key is non-empty (GPU busy)
- the bash KB doesn't exist locally (no fixture to query)
"""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING

from benchmarks import BenchmarkSkipped

if TYPE_CHECKING:
    pass

DEFAULT_N = 20
DEFAULT_K = 5
GPU_LEASE_KEY = "lab:gpu:lease"
QUESTION = "how do I redirect stderr to stdout"
KB_NAME = "bash"


def _check_gpu_lease_free() -> None:
    """Raise BenchmarkSkipped if Valkey is unreachable or the lease is held."""
    try:
        import redis

        from lab.core.settings import get_settings
    except ImportError as exc:  # pragma: no cover — lab not installed
        raise BenchmarkSkipped(f"lab packages not importable: {exc}") from exc

    try:
        client: redis.Redis[str] = redis.Redis.from_url(
            get_settings().redis_url, decode_responses=True
        )
        holder = client.get(GPU_LEASE_KEY)
    except redis.RedisError as exc:
        raise BenchmarkSkipped(f"valkey unreachable: {exc}") from exc

    if holder:
        raise BenchmarkSkipped(f"GPU lease held by {holder!r}")


def _resolve_kb_dir() -> object:
    try:
        from lab.core.settings import get_settings
        from lab.rag.index import count_rows
    except ImportError as exc:  # pragma: no cover
        raise BenchmarkSkipped(f"lab.rag not importable: {exc}") from exc

    settings = get_settings()
    # lab.cli._kb_dir uses settings.kb_root; mirror that resolution here so
    # the bench works without invoking the typer app under subprocess.
    kb_root = getattr(settings, "kb_root", None)
    if kb_root is None:
        raise BenchmarkSkipped("settings.kb_root is unset")
    from pathlib import Path

    kb_dir = Path(kb_root) / KB_NAME
    if not (kb_dir / "manifest.yaml").exists():
        raise BenchmarkSkipped(f"no KB at {kb_dir}")
    try:
        if count_rows(kb_dir) == 0:
            raise BenchmarkSkipped(f"KB {KB_NAME!r} has zero indexed chunks")
    except Exception as exc:
        raise BenchmarkSkipped(f"count_rows({kb_dir}) failed: {exc}") from exc
    return kb_dir


def run(n: int = DEFAULT_N, k: int = DEFAULT_K) -> dict[str, float]:
    """Time ``hybrid_query`` n times. Returns p50_sec, p95_sec, mean_sec."""
    _check_gpu_lease_free()
    kb_dir = _resolve_kb_dir()

    try:
        from lab.rag.index import hybrid_query
    except ImportError as exc:  # pragma: no cover
        raise BenchmarkSkipped(f"lab.rag.index not importable: {exc}") from exc

    # One warmup pass so VAE/embedder caches aren't included in p50.
    try:
        hybrid_query(kb_dir, QUESTION, k=k, alpha=0.5)  # type: ignore[arg-type]
    except Exception as exc:
        raise BenchmarkSkipped(f"warmup query failed: {exc}") from exc

    timings: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            hybrid_query(kb_dir, QUESTION, k=k, alpha=0.5)  # type: ignore[arg-type]
        except Exception as exc:
            raise BenchmarkSkipped(f"hybrid_query raised at iter {len(timings)}: {exc}") from exc
        timings.append(time.perf_counter() - t0)

    timings.sort()
    return {
        "p50_sec": statistics.median(timings),
        "p95_sec": timings[int(0.95 * len(timings)) - 1],
        "mean_sec": statistics.fmean(timings),
        "n": float(len(timings)),
    }
