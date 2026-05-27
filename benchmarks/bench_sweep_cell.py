"""Bench: single-cell wall-time on ``llama3.1:8b-instruct-q4_K_M``.

Times one synthetic Ollama generate against a fixed prompt — a smoke
proxy for "is the model server alive and roughly as fast as last week?"

This is NOT a full sweep cell (no scorer, no sandbox). For a true cell
timing you'd boot the sandbox, which costs ~5s per run; that's too
heavy for a routine benchmark.

Skips when:

- Ollama service unreachable at ``localhost:11434``
- the target model isn't pulled
"""

from __future__ import annotations

import statistics
import time

from benchmarks import BenchmarkSkipped

MODEL = "llama3.1:8b-instruct-q4_K_M"
PROMPT = "Reply with exactly one word: 'ok'."
DEFAULT_N = 3  # generate is expensive; keep it short
TIMEOUT_SEC = 30.0


def _import_httpx() -> object:
    try:
        import httpx

        return httpx
    except ImportError as exc:  # pragma: no cover
        raise BenchmarkSkipped(f"httpx not importable: {exc}") from exc


def _check_ollama_alive(httpx_mod: object) -> list[str]:
    """Return list of available model names, or raise BenchmarkSkipped."""
    try:
        client = httpx_mod.Client(timeout=2.0)  # type: ignore[attr-defined]
        with client:
            resp = client.get("http://localhost:11434/api/tags")
    except Exception as exc:
        raise BenchmarkSkipped(f"ollama unreachable: {exc}") from exc

    if resp.status_code != 200:
        raise BenchmarkSkipped(f"ollama /api/tags returned {resp.status_code}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise BenchmarkSkipped(f"ollama tags payload invalid: {exc}") from exc

    names = [str(m.get("name", "")) for m in payload.get("models", [])]
    return [n for n in names if n]


def run(n: int = DEFAULT_N) -> dict[str, float]:
    """Time n /api/generate calls against MODEL. Returns p50/p95/mean."""
    httpx_mod = _import_httpx()
    available = _check_ollama_alive(httpx_mod)
    if MODEL not in available:
        raise BenchmarkSkipped(f"model {MODEL!r} not pulled (available: {available[:3]}...)")

    client = httpx_mod.Client(timeout=TIMEOUT_SEC)  # type: ignore[attr-defined]
    body = {"model": MODEL, "prompt": PROMPT, "stream": False, "options": {"num_predict": 4}}

    # Warmup — first call loads weights into VRAM.
    try:
        with client:
            client.post("http://localhost:11434/api/generate", json=body)
    except Exception as exc:
        raise BenchmarkSkipped(f"warmup generate failed: {exc}") from exc

    client = httpx_mod.Client(timeout=TIMEOUT_SEC)  # type: ignore[attr-defined]
    timings: list[float] = []
    with client:
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                resp = client.post("http://localhost:11434/api/generate", json=body)
            except Exception as exc:
                raise BenchmarkSkipped(f"generate raised: {exc}") from exc
            timings.append(time.perf_counter() - t0)
            if resp.status_code != 200:
                raise BenchmarkSkipped(f"generate returned {resp.status_code}")

    timings.sort()
    return {
        "p50_sec": statistics.median(timings),
        "p95_sec": timings[-1],  # n=3 → p95 ≈ max
        "mean_sec": statistics.fmean(timings),
        "n": float(len(timings)),
    }
