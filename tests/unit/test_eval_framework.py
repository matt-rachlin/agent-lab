"""Evaluator framework + built-ins — unit tests (no DB)."""

from __future__ import annotations

from typing import Any

import pytest

from lab.eval.framework import (
    EvalResult,
    RunRow,
    TaskRow,
    clear_registry,
    evaluator,
    get_registry,
)


@pytest.fixture(autouse=True)
def _isolate_registry() -> Any:
    clear_registry()
    yield
    clear_registry()


def _run(text: str | None = "hi", latency: int | None = 100, tokens_out: int | None = 5) -> RunRow:
    return RunRow(
        run_id="r1",
        experiment_id=1,
        model_id=1,
        model_litellm_id="m1",
        task_id=1,
        seed=1,
        status="done",
        tokens_in=10,
        tokens_out=tokens_out,
        latency_ms=latency,
        cost_usd=None,
        trace_path=None,
        response_text=text,
    )


def _task(payload: dict[str, Any]) -> TaskRow:
    return TaskRow(
        task_id=1,
        suite="t",
        slug="t1",
        category="x",
        difficulty="easy",
        payload=payload,
    )


def test_eval_result_helpers() -> None:
    p = EvalResult.passed_(reasoning="ok")
    assert p.passed is True
    assert p.score == 1.0
    f = EvalResult.failed(reasoning="bad")
    assert f.passed is False
    assert f.score == 0.0
    s = EvalResult.skip("n/a")
    assert s.skipped
    assert s.score is None
    sc = EvalResult.scored(0.42)
    assert sc.score == 0.42
    assert sc.passed is None


def test_decorator_registers() -> None:
    @evaluator(name="t_dummy", version="1.0", threshold=0.5)
    def _d(run: RunRow, task: TaskRow) -> EvalResult:
        return EvalResult.passed_()

    reg = get_registry()
    assert "t_dummy" in reg
    entry = reg["t_dummy"]
    assert entry.threshold == 0.5
    assert entry.category == "deterministic"


def test_exact_match() -> None:
    from lab.eval.builtin.exact_match import exact_match

    t = _task({"gold_answer": "42"})
    assert exact_match(_run("the answer is 42"), t).passed is True
    assert exact_match(_run("the answer is 43"), t).passed is False
    assert exact_match(_run(""), t).passed is False
    # missing gold
    assert exact_match(_run("anything"), _task({})).skipped


def test_regex_match() -> None:
    from lab.eval.builtin.regex_match import regex_match

    t = _task({"rubric": {"type": "regex", "pattern": r"\b\d{3}\b"}})
    assert regex_match(_run("ans=391"), t).passed is True
    assert regex_match(_run("ans=42"), t).passed is False
    # wrong rubric type
    assert regex_match(_run("x"), _task({"rubric": {"type": "exact_match"}})).skipped


def test_latency_under(monkeypatch: pytest.MonkeyPatch) -> None:
    # latency_under reads its threshold at import time; use the default 10000
    from lab.eval.builtin.latency_under import latency_under

    assert latency_under(_run(latency=500), _task({})).passed is True
    assert latency_under(_run(latency=20000), _task({})).passed is False
    assert latency_under(_run(latency=None), _task({})).skipped


def test_json_valid() -> None:
    from lab.eval.builtin.json_valid import json_valid

    json_task = _task({"input": "return JSON: {x: 1}"})
    assert json_valid(_run('{"x": 1}'), json_task).passed is True
    assert json_valid(_run('```json\n{"x": 1}\n```'), json_task).passed is True
    assert json_valid(_run("not json"), json_task).passed is False
    # task doesn't ask for JSON → skip
    assert json_valid(_run('{"x":1}'), _task({"input": "say hello"})).skipped


def test_not_empty() -> None:
    from lab.eval.builtin.not_empty import not_empty

    assert not_empty(_run("hello"), _task({})).passed is True
    assert not_empty(_run("   "), _task({})).passed is False
    assert not_empty(_run(None), _task({})).passed is False
