"""Tests for the compound `useful_latency` evaluator."""

from __future__ import annotations

from typing import Any

import pytest
from lab.eval.framework import RunRow, TaskRow, clear_registry


@pytest.fixture(autouse=True)
def _isolate_registry() -> Any:
    clear_registry()
    yield
    clear_registry()


def _run(text: str | None, latency_ms: int | None = 500) -> RunRow:
    return RunRow(
        run_id="r1",
        experiment_id=1,
        model_id=1,
        model_litellm_id="m1",
        task_id=1,
        seed=1,
        status="done",
        tokens_in=10,
        tokens_out=5,
        latency_ms=latency_ms,
        cost_usd=None,
        trace_path=None,
        response_text=text,
    )


def _task() -> TaskRow:
    return TaskRow(task_id=1, suite="t", slug="t1", category=None, difficulty=None, payload={})


def test_useful_latency_fast_and_non_empty_passes() -> None:
    from lab.eval.builtin.useful_latency import useful_latency

    assert useful_latency(_run("hello"), _task()).passed is True


def test_useful_latency_fast_but_empty_fails() -> None:
    """Regression for EXP-001: an empty response was previously passing latency."""
    from lab.eval.builtin.useful_latency import useful_latency

    res = useful_latency(_run("   ", latency_ms=100), _task())
    assert res.passed is False
    assert "not_empty" in (res.reasoning or "")


def test_useful_latency_slow_fails_even_if_non_empty() -> None:
    from lab.eval.builtin.useful_latency import useful_latency

    res = useful_latency(_run("hello", latency_ms=99999999), _task())
    assert res.passed is False
    assert "latency" in (res.reasoning or "")


def test_useful_latency_skip_when_no_latency() -> None:
    from lab.eval.builtin.useful_latency import useful_latency

    res = useful_latency(_run("hi", latency_ms=None), _task())
    assert res.skipped is True
