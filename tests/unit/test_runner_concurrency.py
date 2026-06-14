"""Unit tests for SGLang Phase 1 Stage B runner concurrency (G1/M1/M2).

These cover the resident-batch throughput path in `lab.sweep.runner` WITHOUT
touching the GPU, LiteLLM, or llama-swap:

* `_resident_batch_model` eligibility (only sglang-local, single-model,
  max_concurrency>1 qualifies; ollama / cloud / mixed / c1 fall back to serial).
* `skip_cell_lease` threads through `execute_cell` to the path executors so the
  per-cell lease is suppressed when the batch holds one (M1).
* `_run_resident_batch` dispatches every cell exactly once, acquires exactly ONE
  lease for the whole batch (M1), declares(preflight=False)+teardown once (M2),
  and passes `model_pool=None` + `skip_cell_lease=True` to each cell.
* Parity: the concurrent path produces the same per-cell results as a serial
  walk over the same cells.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any

from lab.sweep import runner as runner_mod
from lab.sweep.runner import Cell, CellResult, _resident_batch_model


def _mk_cell(
    run_id: str,
    *,
    model: str = "qwen3-4b-awq",
    backend: str = "sglang-local",
) -> Cell:
    return Cell(
        run_id=run_id,
        experiment_id=1,
        experiment_slug="EXP",
        model_id=2,
        model_litellm_id=model,
        model_backend=backend,
        task_id=3,
        task_slug="t",
        task_payload={"input": "hi"},
        config=runner_mod.RunConfig(name="c"),
        config_hash="h",
        seed=0,
    )


# --------------------------------------------------------------------------
# _resident_batch_model eligibility
# --------------------------------------------------------------------------


def test_sglang_multi_concurrency_single_model_is_eligible() -> None:
    cells = [_mk_cell(f"r{i}") for i in range(4)]
    assert _resident_batch_model(cells, 16) == "qwen3-4b-awq"


def test_concurrency_one_falls_back_to_serial() -> None:
    cells = [_mk_cell(f"r{i}") for i in range(4)]
    assert _resident_batch_model(cells, 1) is None


def test_ollama_backend_never_resident_batch() -> None:
    cells = [_mk_cell(f"r{i}", model="qwen3-4b", backend="ollama-local") for i in range(4)]
    assert _resident_batch_model(cells, 16) is None


def test_cloud_backend_never_resident_batch() -> None:
    cells = [_mk_cell(f"r{i}", model="claude", backend="anthropic") for i in range(4)]
    assert _resident_batch_model(cells, 16) is None


def test_mixed_models_same_backend_falls_back() -> None:
    cells = [_mk_cell("a", model="qwen3-4b-awq"), _mk_cell("b", model="other-awq")]
    assert _resident_batch_model(cells, 16) is None


def test_empty_todo_is_none() -> None:
    assert _resident_batch_model([], 16) is None


# --------------------------------------------------------------------------
# skip_cell_lease plumbing through execute_cell
# --------------------------------------------------------------------------


def test_skip_cell_lease_threads_to_bfcl_executor(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def fake_bfcl(**kwargs: Any) -> CellResult:
        seen.update(kwargs)
        return CellResult(
            run_id=kwargs["cell"].run_id,
            status="done",
            tokens_in=None,
            tokens_out=None,
            latency_ms=0,
            cost_usd=None,
            error=None,
            response_text=None,
            raw_response=None,
        )

    monkeypatch.setattr(runner_mod, "_execute_bfcl_cell", fake_bfcl)
    monkeypatch.setattr(runner_mod, "capture_manifest", lambda extra: type("M", (), {"sha": "d"})())
    cell = _mk_cell("r1")
    cell.task_payload["rubric"] = {"type": "bfcl_ast"}  # force the bfcl path
    runner_mod.execute_cell(cell, litellm_key="k", timeout=10, skip_cell_lease=True)
    assert seen["skip_cell_lease"] is True


def test_default_execute_cell_does_not_skip_lease(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def fake_single(**kwargs: Any) -> CellResult:
        seen.update(kwargs)
        return CellResult(
            run_id=kwargs["cell"].run_id,
            status="done",
            tokens_in=None,
            tokens_out=None,
            latency_ms=0,
            cost_usd=None,
            error=None,
            response_text=None,
            raw_response=None,
        )

    monkeypatch.setattr(runner_mod, "_execute_single_turn", fake_single)
    monkeypatch.setattr(runner_mod, "capture_manifest", lambda extra: type("M", (), {"sha": "d"})())
    runner_mod.execute_cell(_mk_cell("r1"), litellm_key="k", timeout=10)
    assert seen["skip_cell_lease"] is False


# --------------------------------------------------------------------------
# _run_resident_batch dispatch (G1/M1/M2)
# --------------------------------------------------------------------------


class _FakePool:
    def __init__(self) -> None:
        self.declares: list[tuple[str, bool]] = []
        self.teardowns = 0

    def declare(self, plan: Any, *, preflight: bool = True) -> None:
        self.declares.append((plan.pipeline_id, preflight))

    def teardown(self) -> None:
        self.teardowns += 1


def _install_fake_lease(monkeypatch: Any) -> dict[str, int]:
    """Replace runner.gpu_lease with a recording no-op context manager."""

    counters = {"acquired": 0, "active": 0, "max_active": 0}
    lock = threading.Lock()

    @contextlib.contextmanager
    def fake_lease(owner: str, *, ttl_sec: int = 1800) -> Any:
        with lock:
            counters["acquired"] += 1
            counters["active"] += 1
            counters["max_active"] = max(counters["max_active"], counters["active"])
        try:
            yield owner
        finally:
            with lock:
                counters["active"] -= 1

    monkeypatch.setattr(runner_mod, "gpu_lease", fake_lease)
    monkeypatch.setattr(
        runner_mod,
        "plan_for_cell",
        lambda **kw: type("P", (), {"pipeline_id": kw["pipeline_id"]})(),
    )
    return counters


def _run_batch_with_fakes(
    monkeypatch: Any, n_cells: int, max_concurrency: int
) -> tuple[dict[str, Any], _FakePool, dict[str, int]]:
    counters = _install_fake_lease(monkeypatch)
    pool = _FakePool()
    call_kwargs: list[dict[str, Any]] = []
    klock = threading.Lock()

    def fake_execute_cell(cell: Cell, **kwargs: Any) -> CellResult:
        with klock:
            call_kwargs.append({"run_id": cell.run_id, **kwargs})
        return CellResult(
            run_id=cell.run_id,
            status="done",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            cost_usd=None,
            error=None,
            response_text=None,
            raw_response=None,
        )

    monkeypatch.setattr(runner_mod, "execute_cell", fake_execute_cell)

    cells = [_mk_cell(f"r{i}") for i in range(n_cells)]
    spec = _mk_spec(max_concurrency)
    results: dict[str, str] = {}

    def on_result(cell: Cell, result: CellResult) -> None:
        results[cell.run_id] = result.status

    runner_mod._run_resident_batch(
        todo=cells,
        spec=spec,
        litellm_key="k",
        model_pool=pool,
        batch_model="qwen3-4b-awq",
        on_result=on_result,
    )
    return (
        {"results": results, "call_kwargs": call_kwargs},
        pool,
        counters,
    )


def _mk_spec(max_concurrency: int) -> Any:
    """Minimal stand-in for SweepConfig with the fields the dispatcher reads."""

    exp = type("Exp", (), {"slug": "EXP"})()
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


def test_resident_batch_dispatches_every_cell_once(monkeypatch: Any) -> None:
    out, _pool, _counters = _run_batch_with_fakes(monkeypatch, n_cells=20, max_concurrency=8)
    assert set(out["results"]) == {f"r{i}" for i in range(20)}
    assert all(v == "done" for v in out["results"].values())
    assert len(out["call_kwargs"]) == 20


def test_resident_batch_holds_exactly_one_lease(monkeypatch: Any) -> None:
    _out, _pool, counters = _run_batch_with_fakes(monkeypatch, n_cells=20, max_concurrency=8)
    # M1: one lease for the whole batch, never per-cell.
    assert counters["acquired"] == 1
    assert counters["max_active"] == 1


def test_resident_batch_passes_skip_lease_and_no_pool(monkeypatch: Any) -> None:
    out, _pool, _counters = _run_batch_with_fakes(monkeypatch, n_cells=5, max_concurrency=4)
    for kw in out["call_kwargs"]:
        assert kw["skip_cell_lease"] is True  # M1
        assert kw["model_pool"] is None  # M2


def test_resident_batch_lifecycle_declare_no_preflight_teardown_once(
    monkeypatch: Any,
) -> None:
    _out, pool, _counters = _run_batch_with_fakes(monkeypatch, n_cells=5, max_concurrency=4)
    # M2: declared exactly once with preflight=False, torn down exactly once.
    assert pool.declares == [("sweep:EXP", False)]
    assert pool.teardowns == 1


def test_resident_batch_parity_with_serial(monkeypatch: Any) -> None:
    """Concurrent dispatch yields the same per-cell result set as a serial walk."""

    out, _pool, _counters = _run_batch_with_fakes(monkeypatch, n_cells=32, max_concurrency=16)
    concurrent = out["results"]
    serial = {f"r{i}": "done" for i in range(32)}
    assert concurrent == serial
