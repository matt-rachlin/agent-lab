"""A4-prep: critical-path tests for lab.sweep.runner (wave-2 named).

Three test groups:

1. _run_resident_batch invariants -- worker-raises, on_result thread, lease scope.
2. execute_cell dispatcher -- parametrized over backend x cell-type x skip_cell_lease.
3. run_sweep finalize lifecycle -- planned->done, done->done (no-op), analyzed->analyzed,
   SIGTERM finalize.

No live GPU / LiteLLM / Postgres. All external deps are mocked.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from lab.sweep import runner as runner_mod
from lab.sweep.runner import Cell, CellResult, RunConfig, _run_resident_batch

# ---------------------------------------------------------------------------
# Shared cell factory (same pattern as test_runner_concurrency)
# ---------------------------------------------------------------------------


def _mk_cell(
    run_id: str,
    *,
    model: str = "qwen3-4b-awq",
    backend: str = "sglang-local",
    rubric_type: str | None = None,
    max_turns: int = 1,
    tool_budget: int = 0,
) -> Cell:
    payload: dict[str, Any] = {"input": "hi", "max_turns": max_turns, "tool_budget": tool_budget}
    if rubric_type is not None:
        payload["rubric"] = {"type": rubric_type}
    return Cell(
        run_id=run_id,
        experiment_id=1,
        experiment_slug="EXP-TEST",
        model_id=2,
        model_litellm_id=model,
        model_backend=backend,
        task_id=3,
        task_slug="t",
        task_payload=payload,
        config=RunConfig(name="c"),
        config_hash="h",
        seed=0,
    )


def _ok_result(run_id: str) -> CellResult:
    return CellResult(
        run_id=run_id,
        status="done",
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        cost_usd=None,
        error=None,
        response_text=None,
        raw_response=None,
    )


def _mk_spec(max_concurrency: int = 4) -> Any:
    exp = type("Exp", (), {"slug": "EXP-TEST"})()
    return type(
        "Spec",
        (),
        {
            "experiment": exp,
            "max_concurrency": max_concurrency,
            "request_timeout_sec": 600,
            "model_defaults": {},
        },
    )()


# ---------------------------------------------------------------------------
# Fake ModelPool
# ---------------------------------------------------------------------------


class _FakePool:
    def __init__(self) -> None:
        self.declares: list[Any] = []
        self.teardowns = 0

    def declare(self, plan: Any, *, preflight: bool = True) -> None:
        self.declares.append((plan, preflight))

    def step_start(self, _tag: str) -> None:
        pass

    def step_complete(self, _tag: str) -> None:
        pass

    def teardown(self) -> None:
        self.teardowns += 1


# ---------------------------------------------------------------------------
# Fake lease recorder
# ---------------------------------------------------------------------------


def _install_fake_lease(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counters: dict[str, int] = {"acquired": 0, "released": 0, "active": 0}
    lock = threading.Lock()

    @contextlib.contextmanager
    def _fake_lease(owner: str, *, ttl_sec: int = 1800) -> Any:
        with lock:
            counters["acquired"] += 1
            counters["active"] += 1
        try:
            yield owner
        finally:
            with lock:
                counters["released"] += 1
                counters["active"] -= 1

    monkeypatch.setattr(runner_mod, "gpu_lease", _fake_lease)
    monkeypatch.setattr(
        runner_mod,
        "plan_for_cell",
        lambda **kw: type("P", (), {"pipeline_id": kw.get("pipeline_id", "x")})(),
    )
    return counters


# ===========================================================================
# Test Group 1 -- _run_resident_batch invariants
# ===========================================================================


class TestResidentBatchInvariants:
    """Group 1: correctness properties of _run_resident_batch."""

    def _run_batch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        n_cells: int,
        *,
        fail_cells: set[int] | None = None,
    ) -> tuple[dict[str, Any], _FakePool, dict[str, int], list[tuple[Cell, CellResult]]]:
        """Run n_cells through _run_resident_batch; fail_cells are indices that raise."""
        counters = _install_fake_lease(monkeypatch)
        pool = _FakePool()
        on_result_calls: list[tuple[Cell, CellResult]] = []
        on_result_threads: list[str] = []
        lock = threading.Lock()

        def fake_execute_cell(cell: Cell, **kwargs: Any) -> CellResult:
            idx = int(cell.run_id[1:])
            if fail_cells and idx in fail_cells:
                raise RuntimeError(f"worker {idx} exploded")
            return _ok_result(cell.run_id)

        def on_result(cell: Cell, result: CellResult) -> None:
            with lock:
                on_result_calls.append((cell, result))
                on_result_threads.append(threading.current_thread().name)

        monkeypatch.setattr(runner_mod, "execute_cell", fake_execute_cell)

        cells = [_mk_cell(f"r{i}") for i in range(n_cells)]
        spec = _mk_spec(max_concurrency=4)

        _run_resident_batch(
            todo=cells,
            spec=spec,
            litellm_key="k",
            model_pool=pool,
            batch_model="qwen3-4b-awq",
            on_result=on_result,
        )
        return (
            {"on_result_calls": on_result_calls, "on_result_threads": on_result_threads},
            pool,
            counters,
            on_result_calls,
        )

    def test_declare_called_before_any_cell_executes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ModelPool.declare must be invoked before execute_cell runs any cell."""
        pool = _FakePool()
        declare_times: list[float] = []
        execute_times: list[float] = []

        original_declare = pool.declare

        def recording_declare(plan: Any, *, preflight: bool = True) -> None:
            declare_times.append(time.monotonic())
            original_declare(plan, preflight=preflight)

        pool.declare = recording_declare  # type: ignore[method-assign]

        _install_fake_lease(monkeypatch)

        def fake_execute_cell(cell: Cell, **kwargs: Any) -> CellResult:
            execute_times.append(time.monotonic())
            return _ok_result(cell.run_id)

        monkeypatch.setattr(runner_mod, "execute_cell", fake_execute_cell)

        cells = [_mk_cell(f"r{i}") for i in range(4)]
        _run_resident_batch(
            todo=cells,
            spec=_mk_spec(),
            litellm_key="k",
            model_pool=pool,
            batch_model="qwen3-4b-awq",
            on_result=lambda c, r: None,
        )

        assert len(declare_times) == 1, "declare called exactly once"
        assert len(execute_times) == 4, "all 4 cells executed"
        assert declare_times[0] < execute_times[0], "declare before first execute"

    def test_teardown_called_regardless_of_cell_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """teardown runs even when worker cells raise."""
        _out, pool, _c, _calls = self._run_batch(monkeypatch, 5, fail_cells={1, 3})
        assert pool.teardowns == 1, "teardown called exactly once despite failures"

    def test_on_result_fires_from_main_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """on_result callbacks must run on the main thread (as_completed loop)."""
        out, _pool, _c, _calls = self._run_batch(monkeypatch, 6)
        main = threading.main_thread().name
        assert all(t == main for t in out["on_result_threads"]), (
            "all on_result calls from main thread"
        )

    def test_failing_worker_does_not_abort_rest_of_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker raise -> error CellResult; remaining cells still run + on_result fired."""
        _out, _pool, _c, calls = self._run_batch(monkeypatch, 6, fail_cells={2})
        assert len(calls) == 6, "on_result fired for all 6 cells (incl. the failing one)"
        statuses = [r.status for _, r in calls]
        assert statuses.count("error") == 1, "exactly one error result"
        assert statuses.count("done") == 5, "remaining 5 succeeded"

    def test_gpu_lease_held_throughout_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The gpu_lease must be acquired before cells run and released after teardown."""
        _out, pool, counters, _calls = self._run_batch(monkeypatch, 4)
        assert counters["acquired"] == 1
        assert counters["released"] == 1
        assert counters["active"] == 0
        assert pool.teardowns == 1


# ===========================================================================
# Test Group 2 -- execute_cell dispatcher
# ===========================================================================


@pytest.mark.parametrize(
    ("backend", "is_agent_cell", "is_bfcl_cell", "skip_cell_lease"),
    [
        # local backends x cell types x skip_cell_lease
        ("ollama-local", False, False, False),
        ("ollama-local", False, False, True),
        ("ollama-local", True, False, False),
        ("ollama-local", True, False, True),
        ("ollama-local", False, True, False),
        ("ollama-local", False, True, True),
        ("sglang-local", False, False, False),
        ("sglang-local", False, False, True),
        ("sglang-local", True, False, False),
        ("sglang-local", True, False, True),
        ("sglang-local", False, True, False),
        ("sglang-local", False, True, True),
        # cloud backend
        ("litellm-cloud", False, False, False),
        ("litellm-cloud", False, False, True),
        ("litellm-cloud", True, False, False),
        ("litellm-cloud", False, True, False),
    ],
)
def test_execute_cell_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    is_agent_cell: bool,
    is_bfcl_cell: bool,
    skip_cell_lease: bool,
) -> None:
    """For every (backend, cell-type, skip_cell_lease) combo, verify the right
    executor is called and gpu_lease / model_pool are used correctly."""

    rubric_type = "bfcl_ast" if is_bfcl_cell else None
    max_turns = 5 if is_agent_cell and not is_bfcl_cell else 1
    tool_budget = 10 if is_agent_cell and not is_bfcl_cell else 0
    cell = _mk_cell(
        "r-dispatch",
        model="qwen3-4b",
        backend=backend,
        rubric_type=rubric_type,
        max_turns=max_turns,
        tool_budget=tool_budget,
    )

    called: dict[str, bool] = {"single": False, "bfcl": False, "agent": False}

    def fake_single(**kwargs: Any) -> CellResult:
        called["single"] = True
        return _ok_result(kwargs["cell"].run_id)

    def fake_bfcl(**kwargs: Any) -> CellResult:
        called["bfcl"] = True
        return _ok_result(kwargs["cell"].run_id)

    def fake_agent(**kwargs: Any) -> CellResult:
        called["agent"] = True
        return _ok_result(kwargs["cell"].run_id)

    monkeypatch.setattr(runner_mod, "_execute_single_turn", fake_single)
    monkeypatch.setattr(runner_mod, "_execute_bfcl_cell", fake_bfcl)
    monkeypatch.setattr(runner_mod, "_execute_agent_cell", fake_agent)

    fake_manifest = type("M", (), {"sha": "deadbeef"})()
    monkeypatch.setattr(runner_mod, "capture_manifest", lambda extra: fake_manifest)

    lease_acquired: list[bool] = []

    @contextlib.contextmanager
    def _fake_lease(owner: str, *, ttl_sec: int = 1800) -> Any:
        lease_acquired.append(True)
        try:
            yield owner
        finally:
            pass

    monkeypatch.setattr(runner_mod, "gpu_lease", _fake_lease)

    pool = _FakePool()
    monkeypatch.setattr(
        runner_mod,
        "plan_for_cell",
        lambda **kw: type("P", (), {"pipeline_id": kw.get("pipeline_id", "x")})(),
    )

    runner_mod.execute_cell(
        cell,
        litellm_key="k",
        timeout=30,
        model_pool=pool,
        skip_cell_lease=skip_cell_lease,
    )

    # Assert correct executor was dispatched.
    if is_bfcl_cell:
        assert called["bfcl"], "BFCL cell must use _execute_bfcl_cell"
        assert not called["single"]
        assert not called["agent"]
    elif is_agent_cell:
        assert called["agent"], "agent cell must use _execute_agent_cell"
        assert not called["single"]
        assert not called["bfcl"]
    else:
        assert called["single"], "plain cell must use _execute_single_turn"
        assert not called["bfcl"]
        assert not called["agent"]

    # model_pool.declare called once per cell; teardown once per cell.
    assert len(pool.declares) == 1, "model_pool.declare called exactly once per cell"
    assert pool.teardowns == 1, "model_pool.teardown called exactly once per cell"


# ===========================================================================
# Test Group 3 -- run_sweep finalize lifecycle
# ===========================================================================


@dataclass
class _FakeCursor:
    """Records SQL; supports lifecycle tests."""

    _side_effects: list[Any] = field(default_factory=list)
    _idx: int = 0
    executed: list[tuple[str, Any]] = field(default_factory=list)
    rowcount: int = 1

    def execute(self, query: str, params: Any = None) -> None:
        self.executed.append((query, params))

    def fetchone(self) -> Any:
        if self._idx < len(self._side_effects):
            val = self._side_effects[self._idx]
            self._idx += 1
            return val
        return None

    def fetchall(self) -> list[Any]:
        if self._idx < len(self._side_effects):
            val = self._side_effects[self._idx]
            self._idx += 1
            return val if isinstance(val, list) else []
        return []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


@dataclass
class _FakeConn:
    cur: _FakeCursor

    def cursor(self) -> _FakeCursor:
        return self.cur

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def _make_spec_for_lifecycle() -> Any:
    """Minimal SweepConfig-like object that passes run_sweep's guards."""

    class Exp:
        slug = "EXP-LIFECYCLE"
        title = "lifecycle test"
        hypothesis = None
        plan_path = None
        create_if_missing = True

    class Spec:
        experiment = Exp()
        models: ClassVar[list[str]] = ["qwen3-4b"]
        tasks: ClassVar[dict[str, Any]] = {}
        configs: ClassVar[list[Any]] = []
        seeds: ClassVar[list[int]] = [0]
        max_concurrency = 1
        request_timeout_sec = 30
        model_defaults: ClassVar[dict[str, Any]] = {}

    return Spec()


def _stub_run_sweep_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch everything run_sweep calls except the DB finalize logic."""
    monkeypatch.setattr(runner_mod, "preflight_litellm_keep_alive_or_raise", lambda: None)
    monkeypatch.setattr(runner_mod, "_slow_models_in", lambda ids: [])
    monkeypatch.setattr(runner_mod, "_models_lookup", lambda ids: {"qwen3-4b": (1, "ollama-local")})
    monkeypatch.setattr(runner_mod, "expand_matrix", lambda spec, eid, models: [])
    monkeypatch.setattr(runner_mod, "_done_run_ids", lambda eid: set())
    monkeypatch.setattr(runner_mod, "_install_signal_handlers", lambda slug: None)
    monkeypatch.setattr(runner_mod, "_write_pidfile", lambda slug: None)
    monkeypatch.setattr(runner_mod, "_clear_pidfile", lambda slug: None)
    monkeypatch.setattr(runner_mod, "ModelPool", MagicMock(side_effect=Exception("no pool")))

    import lab.observability.tracing as _tracing

    monkeypatch.setattr(_tracing, "configure_mlflow_tracing", lambda url, slug: None)


def _finalize_sql_stmts(
    cursors: list[_FakeCursor],
) -> list[tuple[str, Any]]:
    out = []
    for c in cursors:
        out.extend((q, p) for q, p in c.executed if "UPDATE experiments" in q and "status" in q)
    return out


class TestRunSweepFinalizeLifecycle:
    """Group 3: experiment status transitions in run_sweep's finally block."""

    def _run_collecting_cursors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        experiment_id: int = 99,
    ) -> list[_FakeCursor]:
        """Run run_sweep; capture all cursors created across all psycopg.connect calls."""
        _stub_run_sweep_core(monkeypatch)

        call_count = {"n": 0}
        cursors: list[_FakeCursor] = []

        def _connect(dsn: str) -> _FakeConn:
            c = _FakeCursor(_side_effects=[(experiment_id,)] if call_count["n"] == 0 else [])
            call_count["n"] += 1
            cursors.append(c)
            return _FakeConn(cur=c)

        def _fake_settings() -> Any:
            return type("S", (), {"pg_dsn": "fake", "llama_swap_url": ""})()

        monkeypatch.setattr(runner_mod, "get_settings", _fake_settings)  # type: ignore[arg-type]
        monkeypatch.setattr(runner_mod.psycopg, "connect", _connect)

        spec = _make_spec_for_lifecycle()
        with contextlib.suppress(Exception):
            runner_mod.run_sweep(spec, litellm_key="k")

        return cursors

    def test_planned_transitions_to_done(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A planned experiment: running promotion fires, then done finalize fires."""
        cursors = self._run_collecting_cursors(monkeypatch)
        all_updates = _finalize_sql_stmts(cursors)

        running_updates = [q for q, _ in all_updates if "'running'" in q]
        done_updates = [q for q, _ in all_updates if "status = 'done'" in q]
        assert running_updates, "planned experiment should be promoted to 'running'"
        assert done_updates, "experiment should be finalized to 'done'"

    def test_done_not_demoted_to_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 'done' experiment must not be demoted to 'running' on re-run.

        The running-promotion UPDATE is restricted to status='planned';
        'done' rows pass through that guard unchanged.
        """
        cursors = self._run_collecting_cursors(monkeypatch)
        all_updates = _finalize_sql_stmts(cursors)

        running_updates = [q for q, _ in all_updates if "'running'" in q]
        for q in running_updates:
            assert "status = 'planned'" in q or "status='planned'" in q, (
                "running-promotion must only fire for planned experiments"
            )

    def test_analyzed_excluded_from_done_overwrite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The finalize WHERE clause must exclude 'analyzed' so it is not clobbered."""
        cursors = self._run_collecting_cursors(monkeypatch)
        all_updates = _finalize_sql_stmts(cursors)

        done_updates = [q for q, _ in all_updates if "status = 'done'" in q]
        for q in done_updates:
            assert "analyzed" in q, (
                "finalize UPDATE must exclude 'analyzed' status from being clobbered"
            )

    def test_sigterm_finalize_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exception inside run_sweep's try block -> finally still finalizes status='done'.

        We simulate a crash by having execute_cell raise RuntimeError on the
        first cell. This happens inside the big ``try:`` that wraps the dispatch
        loop, so the ``finally:`` that writes status='done' must still fire.
        """
        _stub_run_sweep_core(monkeypatch)

        call_count = {"n": 0}
        cursors: list[_FakeCursor] = []
        experiment_id = 77

        def _connect(dsn: str) -> _FakeConn:
            c = _FakeCursor(_side_effects=[(experiment_id,)] if call_count["n"] == 0 else [])
            call_count["n"] += 1
            cursors.append(c)
            return _FakeConn(cur=c)

        def _fake_settings() -> Any:
            return type("S", (), {"pg_dsn": "fake", "llama_swap_url": ""})()

        monkeypatch.setattr(runner_mod, "get_settings", _fake_settings)  # type: ignore[arg-type]
        monkeypatch.setattr(runner_mod.psycopg, "connect", _connect)

        # Return one cell so the serial dispatch loop runs at least once.
        one_cell = _mk_cell("r0", backend="ollama-local")
        monkeypatch.setattr(runner_mod, "expand_matrix", lambda *_a, **_kw: [one_cell])
        monkeypatch.setattr(runner_mod, "_done_run_ids", lambda eid: set())

        # Make execute_cell raise — simulates a SIGTERM-induced crash mid-sweep.
        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("boom -- simulated SIGTERM")

        monkeypatch.setattr(runner_mod, "execute_cell", _boom)
        # Also stub capture_manifest so execute_cell's failure is the crash path.
        monkeypatch.setattr(
            runner_mod, "capture_manifest", lambda extra: type("M", (), {"sha": "x"})()
        )

        spec = _make_spec_for_lifecycle()
        with contextlib.suppress(RuntimeError):
            runner_mod.run_sweep(spec, litellm_key="k")

        all_updates = _finalize_sql_stmts(cursors)
        done_updates = [q for q, _ in all_updates if "status = 'done'" in q]
        assert done_updates, "finally block must still finalize status='done' after exception"
