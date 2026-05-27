"""Shared pytest fixtures for ``benchmarks/`` *should we ever pytest-collect them*.

We deliberately do NOT mark benches as pytest tests by default — they
are run via ``python -m benchmarks.runner``. This conftest exists so
that if a developer runs ``pytest benchmarks/`` they get clear
collection behavior (it short-circuits to a no-op) rather than
accidentally invoking ``run()`` under pytest's import machinery and
producing misleading "test" results.

Real unit tests for the runner live in ``tests/unit/test_benchmarks_*.py``.
"""

from __future__ import annotations

import pytest


def pytest_ignore_collect(collection_path: object, config: object) -> bool:
    """Skip pytest collection of bench_*.py files.

    These modules are invoked through :func:`benchmarks.runner.run_all`,
    not pytest. Returning True per-path tells pytest "do not import this
    file as a test module".
    """
    name = str(collection_path).rsplit("/", 1)[-1]
    return name.startswith("bench_") and name.endswith(".py")


@pytest.fixture
def bench_warmup_calls() -> int:
    """Default warmup count for bench fixtures that use the fixture system."""
    return 2


@pytest.fixture
def bench_measure_calls() -> int:
    """Default measured iteration count for bench fixtures."""
    return 20
