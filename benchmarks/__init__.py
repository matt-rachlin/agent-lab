"""Phase 13.3: micro-benchmarks for the lab.

Each ``bench_*.py`` module exposes a top-level ``run()`` returning a
``dict[str, float]`` mapping metric name to value (typically seconds).
The :mod:`benchmarks.runner` orchestrates timing, history capture, and
regression detection. See ``benchmarks/README.md`` for the user-facing
contract.
"""

from __future__ import annotations

__all__ = ["BenchmarkSkipped"]


class BenchmarkSkipped(RuntimeError):
    """Bench ``run()`` raises this when a precondition isn't met.

    Examples: GPU lease busy, Ollama service down, rerank service down.
    The runner catches this and records ``status=skipped`` without
    counting it as a regression.
    """
