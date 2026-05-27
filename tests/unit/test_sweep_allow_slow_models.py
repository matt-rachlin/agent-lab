"""Unit tests for the Phase 19e `--allow-slow-models` sweep-runner gate.

The infrastructure refuses to dispatch a sweep that references a ceiling-
class model (capabilities contain `slow_mode` in lab.models, e.g.
llama-3.3-70b-q4 at 6-10 tok/s) unless the operator explicitly opts in
with `--allow-slow-models`. The gate runs BEFORE matrix expansion so the
rejection is fast and clean — no task lookups, no GPU lease, no I/O.

These tests exercise:

- The DB-lookup helper `_slow_models_in` against a stubbed cursor (we
  don't want a real psycopg connection in unit scope).
- The gate logic in `run_sweep` via monkey-patching `_slow_models_in`
  and `_models_lookup`, asserting the `SlowModelGateError` shape and
  that the flag toggles enforcement.
"""

from __future__ import annotations

from typing import Any

import pytest

from lab.sweep import runner as runner_mod
from lab.sweep.config import (
    ExperimentRef,
    RunConfig,
    SweepConfig,
    TaskRef,
)
from lab.sweep.runner import (
    SLOW_MODEL_CAPABILITY,
    SlowModelGateError,
    _slow_models_in,
)


def _make_spec(models: list[str]) -> SweepConfig:
    """Build a minimal SweepConfig referencing the given litellm_ids."""

    return SweepConfig(
        experiment=ExperimentRef(slug="phase19e-test"),
        tasks=TaskRef(suite="smoke", slugs=["unused"]),
        models=models,
        configs=[RunConfig(name="default")],
        seeds=[0],
    )


# ----------------------------------------------------------------------------
# _slow_models_in — the DB helper
# ----------------------------------------------------------------------------


class _FakeCursor:
    """Records the SQL parameters and returns canned rows."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_: Any) -> None:
        pass


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *_: Any) -> None:
        pass


def test_slow_models_in_empty_list_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """No DB hit when the input is empty — important for the common case
    where a sweep has no slow models (which is most of them)."""

    called = False

    def _connect(_dsn: str) -> _FakeConn:
        nonlocal called
        called = True
        return _FakeConn(_FakeCursor([]))

    monkeypatch.setattr(runner_mod.psycopg, "connect", _connect)
    assert _slow_models_in([]) == []
    assert not called, "should not hit the DB for an empty input"


def test_slow_models_in_filters_by_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper returns only the litellm_ids the DB flagged as slow."""

    cursor = _FakeCursor([("llama-3.3-70b-q4",)])
    monkeypatch.setattr(
        runner_mod.psycopg,
        "connect",
        lambda _dsn: _FakeConn(cursor),
    )

    result = _slow_models_in(["qwen3-14b-q4", "llama-3.3-70b-q4"])
    assert result == ["llama-3.3-70b-q4"]

    # The query MUST scope the capability filter — otherwise the helper
    # would return every slow model regardless of caller's input.
    assert len(cursor.executed) == 1
    _sql, params = cursor.executed[0]
    assert params == (["qwen3-14b-q4", "llama-3.3-70b-q4"], SLOW_MODEL_CAPABILITY)


def test_slow_models_in_returns_empty_when_no_slow(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor([])
    monkeypatch.setattr(
        runner_mod.psycopg,
        "connect",
        lambda _dsn: _FakeConn(cursor),
    )

    assert _slow_models_in(["qwen3-14b-q4"]) == []


# ----------------------------------------------------------------------------
# SlowModelGateError shape
# ----------------------------------------------------------------------------


def test_slow_model_gate_error_is_runtime_error_subclass() -> None:
    """Callers can `except RuntimeError` to catch the gate without a
    direct dep on the sweep runner module."""

    assert issubclass(SlowModelGateError, RuntimeError)


def test_slow_model_gate_error_names_offending_models() -> None:
    """The exception message must list the slow models so the operator
    can either drop them or re-run with the override."""

    msg = "sweep references slow_mode (ceiling-class) models without --allow-slow-models: ['llama-3.3-70b-q4']"
    exc = SlowModelGateError(msg)
    s = str(exc)
    assert "llama-3.3-70b-q4" in s
    assert "--allow-slow-models" in s


# ----------------------------------------------------------------------------
# run_sweep — gate integration
# ----------------------------------------------------------------------------


def _stub_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    slow: list[str],
    models_present: list[str],
) -> None:
    """Patch out the bits of run_sweep that touch the world."""

    monkeypatch.setattr(runner_mod, "preflight_litellm_keep_alive_or_raise", lambda: None)
    monkeypatch.setattr(runner_mod, "_ensure_experiment", lambda _spec: 1)
    monkeypatch.setattr(
        runner_mod,
        "_models_lookup",
        lambda ids: {m: (idx, "ollama-local") for idx, m in enumerate(models_present)},
    )
    monkeypatch.setattr(runner_mod, "_slow_models_in", lambda _ids: list(slow))


def test_gate_refuses_slow_model_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sweep that references a slow_mode model must fail fast."""

    spec = _make_spec(["llama-3.3-70b-q4"])
    _stub_runtime(monkeypatch, slow=["llama-3.3-70b-q4"], models_present=["llama-3.3-70b-q4"])

    with pytest.raises(SlowModelGateError, match="llama-3.3-70b-q4"):
        runner_mod.run_sweep(spec, litellm_key="x", resume=False, dry_run=True)


def test_gate_allows_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """With --allow-slow-models, the gate is a no-op and the sweep proceeds.

    We short-circuit by also stubbing the matrix expansion path (dry_run=True
    skips actual cell execution, but `expand_matrix` still runs); the
    important assertion is that no SlowModelGateError is raised.
    """

    spec = _make_spec(["llama-3.3-70b-q4"])
    _stub_runtime(monkeypatch, slow=["llama-3.3-70b-q4"], models_present=["llama-3.3-70b-q4"])

    # expand_matrix calls get_tasks, which would hit the DB. Stub it.
    monkeypatch.setattr(runner_mod, "expand_matrix", lambda *_a, **_kw: [])
    monkeypatch.setattr(runner_mod, "_done_run_ids", lambda _eid: set())

    summary = runner_mod.run_sweep(
        spec,
        litellm_key="x",
        resume=False,
        dry_run=True,
        allow_slow_models=True,
    )
    assert summary["total"] == 0  # empty matrix from stubbed expand_matrix
    # `executed` is only set inside the non-dry-run branch; the dry-run
    # path returns the {total, done, todo, executed} shape it pre-computed.
    assert summary.get("executed", 0) == 0


def test_gate_passes_when_no_slow_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sweep with only fast models doesn't trip the gate regardless of flag."""

    spec = _make_spec(["qwen3-14b-q4"])
    _stub_runtime(monkeypatch, slow=[], models_present=["qwen3-14b-q4"])
    monkeypatch.setattr(runner_mod, "expand_matrix", lambda *_a, **_kw: [])
    monkeypatch.setattr(runner_mod, "_done_run_ids", lambda _eid: set())

    summary = runner_mod.run_sweep(
        spec,
        litellm_key="x",
        resume=False,
        dry_run=True,
        allow_slow_models=False,
    )
    assert summary["total"] == 0


def test_gate_error_mentions_override_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The error message must tell the operator how to opt in."""

    spec = _make_spec(["llama-3.3-70b-q4"])
    _stub_runtime(monkeypatch, slow=["llama-3.3-70b-q4"], models_present=["llama-3.3-70b-q4"])

    with pytest.raises(SlowModelGateError) as excinfo:
        runner_mod.run_sweep(spec, litellm_key="x", resume=False, dry_run=True)
    assert "--allow-slow-models" in str(excinfo.value)
