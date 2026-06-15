"""Unit tests for :mod:`lab.platform.model_pool`.

The model_pool is a degrade-gracefully optimization layer. Tests focus on:

  * Plan roundtrip (pydantic schema)
  * declare() fires pre-flight + unload for every unique model
  * step_complete() spawns the predictive load as a daemon thread (no block)
  * teardown() is idempotent and walks every model in the plan
  * Every HTTP call site swallows network errors (caller never sees them)

We use ``httpx.MockTransport`` to mock the network without opening real
sockets, mirroring the pattern in :mod:`tests/unit/test_rerank_client.py`.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
import pytest
from lab.platform import model_pool
from lab.platform.model_pool import (
    ModelPool,
    PipelineModelPlan,
    PipelineStep,
    plan_for_cell,
)

# ---------------------------------------------------------------------------
# httpx transport plumbing
# ---------------------------------------------------------------------------


class _RecordingTransport(httpx.MockTransport):
    """Capture every request so tests can assert sequence + bodies."""

    def __init__(
        self,
        handler: Any | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []

        def _wrapped(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if handler is not None:
                return handler(request)
            return httpx.Response(200, json={"msg": "ok"})

        super().__init__(_wrapped)


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: httpx.MockTransport,
) -> None:
    """Patch httpx.Client so every constructed client uses our mock."""

    real_init = httpx.Client.__init__

    def _wrapped_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", _wrapped_init)


# ---------------------------------------------------------------------------
# Plan roundtrip
# ---------------------------------------------------------------------------


def test_pipeline_step_defaults() -> None:
    step = PipelineStep(name="cell")
    assert step.models == []
    assert step.duration_estimate_s is None


def test_pipeline_plan_unique_models_dedupes_and_preserves_order() -> None:
    plan = PipelineModelPlan(
        pipeline_id="t",
        steps=[
            PipelineStep(name="s1", models=["a", "b"]),
            PipelineStep(name="s2", models=["b", "c"]),
            PipelineStep(name="s3", models=["a"]),
        ],
    )
    assert plan.unique_models() == ["a", "b", "c"]


def test_pipeline_plan_unique_models_empty() -> None:
    plan = PipelineModelPlan(pipeline_id="t", steps=[])
    assert plan.unique_models() == []


def test_pipeline_plan_roundtrip() -> None:
    """JSON roundtrip — plans get logged and stored, schema must be stable."""
    plan = PipelineModelPlan(
        pipeline_id="rid-123",
        steps=[
            PipelineStep(name="cell", models=["qwen3-14b-q4"], duration_estimate_s=12.5),
        ],
    )
    raw = plan.model_dump_json()
    rt = PipelineModelPlan.model_validate_json(raw)
    assert rt == plan


# ---------------------------------------------------------------------------
# plan_for_cell helper
# ---------------------------------------------------------------------------


def test_plan_for_cell_without_tools() -> None:
    plan = plan_for_cell(pipeline_id="r1", model_id="qwen3-14b-q4", tools=None)
    assert len(plan.steps) == 1
    assert plan.steps[0].models == ["qwen3-14b-q4"]


def test_plan_for_cell_with_kb_query_adds_side_models() -> None:
    plan = plan_for_cell(
        pipeline_id="r2",
        model_id="qwen3-14b-q4",
        tools=[{"name": "fs_read"}, {"name": "kb_query"}],
    )
    assert plan.steps[0].models == [
        "qwen3-14b-q4",
        "qwen3-embedding",
        "qwen3-reranker-0.6b",
    ]


def test_plan_for_cell_without_kb_query_skips_side_models() -> None:
    plan = plan_for_cell(
        pipeline_id="r3",
        model_id="qwen3-14b-q4",
        tools=[{"name": "fs_read"}, {"name": "shell_exec"}],
    )
    assert plan.steps[0].models == ["qwen3-14b-q4"]


def test_plan_for_cell_side_model_overrides() -> None:
    plan = plan_for_cell(
        pipeline_id="r4",
        model_id="qwen3-14b-q4",
        tools=[{"name": "kb_query"}],
        embedder_model_id=None,
        reranker_model_id="my-reranker",
    )
    assert plan.steps[0].models == ["qwen3-14b-q4", "my-reranker"]


# ---------------------------------------------------------------------------
# declare() — pre-flight pass
# ---------------------------------------------------------------------------


def test_declare_preflights_each_unique_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """declare() should hit /v1/chat/completions + /api/models/unload/<id>
    for each unique model in the plan, deduped."""
    transport = _RecordingTransport()
    _install_transport(monkeypatch, transport)

    plan = PipelineModelPlan(
        pipeline_id="p",
        steps=[
            PipelineStep(name="s1", models=["m1", "m2"]),
            PipelineStep(name="s2", models=["m2", "m3"]),
        ],
    )
    pool = ModelPool(llama_swap_url="http://swap")
    pool.declare(plan)

    # 3 unique models x (completion + unload) = 6 requests
    paths = [(r.method, r.url.path) for r in transport.requests]
    assert paths == [
        ("POST", "/v1/chat/completions"),
        ("POST", "/api/models/unload/m1"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/api/models/unload/m2"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/api/models/unload/m3"),
    ]


def test_declare_empty_plan_makes_no_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _RecordingTransport()
    _install_transport(monkeypatch, transport)

    pool = ModelPool(llama_swap_url="http://swap")
    pool.declare(PipelineModelPlan(pipeline_id="p", steps=[]))

    assert transport.requests == []


def test_declare_preflight_sends_keep_alive_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-flight bodies must include keep_alive=0 so we own eviction."""
    transport = _RecordingTransport()
    _install_transport(monkeypatch, transport)

    plan = PipelineModelPlan(
        pipeline_id="p",
        steps=[PipelineStep(name="s", models=["m1"])],
    )
    ModelPool().declare(plan)

    # The first request is the completion; check its body.
    completion = transport.requests[0]
    body = completion.read()
    import json as _json

    parsed = _json.loads(body)
    assert parsed["model"] == "m1"
    assert parsed["max_tokens"] == 1
    assert parsed["keep_alive"] == 0


def test_declare_swallows_http_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A 5xx from llama-swap must NOT propagate — it's a soft optimization."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = _RecordingTransport(handler=handler)
    _install_transport(monkeypatch, transport)

    plan = PipelineModelPlan(
        pipeline_id="p",
        steps=[PipelineStep(name="s", models=["m1"])],
    )
    # Should not raise.
    ModelPool().declare(plan)
    # Pre-flight failed but the unload should still NOT be attempted —
    # the contract is: if pre-flight blew up we skip the unload (the
    # model likely isn't loaded anyway). Verify by inspecting request log.
    paths = [(r.method, r.url.path) for r in transport.requests]
    assert paths == [("POST", "/v1/chat/completions")]


def test_declare_swallows_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection error must NOT propagate."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = _RecordingTransport(handler=handler)
    _install_transport(monkeypatch, transport)

    plan = PipelineModelPlan(
        pipeline_id="p",
        steps=[PipelineStep(name="s", models=["m1"])],
    )
    ModelPool().declare(plan)  # must not raise


# ---------------------------------------------------------------------------
# step_start / step_complete
# ---------------------------------------------------------------------------


def test_step_start_unknown_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _RecordingTransport()
    _install_transport(monkeypatch, transport)
    pool = ModelPool()
    pool.declare(PipelineModelPlan(pipeline_id="p", steps=[]))
    # No raise, no extra request.
    pool.step_start("unknown")
    assert transport.requests == []


def test_step_complete_unknown_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _RecordingTransport()
    _install_transport(monkeypatch, transport)
    pool = ModelPool()
    pool.declare(
        PipelineModelPlan(
            pipeline_id="p",
            steps=[PipelineStep(name="s1", models=["m1"])],
        )
    )
    # Reset captures after declare's pre-flight requests.
    transport.requests.clear()
    pool.step_complete("never-declared")
    assert transport.requests == []


def test_step_complete_on_last_step_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _RecordingTransport()
    _install_transport(monkeypatch, transport)
    pool = ModelPool()
    pool.declare(
        PipelineModelPlan(
            pipeline_id="p",
            steps=[PipelineStep(name="s1", models=["m1"])],
        )
    )
    transport.requests.clear()
    pool.step_complete("s1")
    # Daemon thread might still spawn, but it has no next step → no work.
    # Give it a beat to confirm nothing happens.
    time.sleep(0.05)
    assert transport.requests == []


def test_step_complete_fires_predictive_load_in_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predictive load runs in a daemon thread — caller must not block."""

    barrier = threading.Event()
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/v1/chat/completions":
            # Block the predictive load briefly to prove non-blocking semantics.
            barrier.wait(timeout=1.0)
        return httpx.Response(200, json={"msg": "ok"})

    transport = _RecordingTransport(handler=handler)
    _install_transport(monkeypatch, transport)

    pool = ModelPool()
    pool.declare(
        PipelineModelPlan(
            pipeline_id="p",
            steps=[
                PipelineStep(name="s1", models=["m1"]),
                PipelineStep(name="s2", models=["m2"]),
            ],
        )
    )
    seen_paths.clear()

    t0 = time.monotonic()
    pool.step_complete("s1")
    elapsed = time.monotonic() - t0

    # The call must return immediately; the handler blocks for up to 1.0s
    # but step_complete must not wait for it.
    assert elapsed < 0.2, f"step_complete blocked for {elapsed:.3f}s"

    # Release the predictive load so daemon thread can finish.
    barrier.set()
    # Give the daemon thread time to issue the request.
    for _ in range(40):
        if seen_paths:
            break
        time.sleep(0.02)

    assert "/v1/chat/completions" in seen_paths


def test_step_complete_predictive_load_targets_next_steps_first_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            import json as _json

            captured_bodies.append(_json.loads(request.read()))
        return httpx.Response(200, json={"msg": "ok"})

    transport = _RecordingTransport(handler=handler)
    _install_transport(monkeypatch, transport)

    pool = ModelPool()
    pool.declare(
        PipelineModelPlan(
            pipeline_id="p",
            steps=[
                PipelineStep(name="s1", models=["m1"]),
                PipelineStep(name="s2", models=["m_target", "m_other"]),
            ],
        )
    )
    captured_bodies.clear()
    pool.step_complete("s1")

    # Wait for the daemon thread.
    for _ in range(40):
        if captured_bodies:
            break
        time.sleep(0.02)
    assert captured_bodies, "predictive load never fired"
    assert captured_bodies[0]["model"] == "m_target"


def test_step_complete_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Predictive load failures must never escape — daemon thread eats them."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = _RecordingTransport(handler=handler)
    _install_transport(monkeypatch, transport)

    pool = ModelPool()
    pool.declare(
        PipelineModelPlan(
            pipeline_id="p",
            steps=[
                PipelineStep(name="s1", models=["m1"]),
                PipelineStep(name="s2", models=["m2"]),
            ],
        )
    )
    pool.step_complete("s1")  # must not raise


# ---------------------------------------------------------------------------
# teardown()
# ---------------------------------------------------------------------------


def test_teardown_unloads_every_unique_model(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _RecordingTransport()
    _install_transport(monkeypatch, transport)

    pool = ModelPool(llama_swap_url="http://swap")
    pool.declare(
        PipelineModelPlan(
            pipeline_id="p",
            steps=[
                PipelineStep(name="s1", models=["m1", "m2"]),
                PipelineStep(name="s2", models=["m2", "m3"]),
            ],
        )
    )
    transport.requests.clear()
    pool.teardown()

    paths = [(r.method, r.url.path) for r in transport.requests]
    assert paths == [
        ("POST", "/api/models/unload/m1"),
        ("POST", "/api/models/unload/m2"),
        ("POST", "/api/models/unload/m3"),
    ]


def test_teardown_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _RecordingTransport()
    _install_transport(monkeypatch, transport)

    pool = ModelPool()
    pool.declare(
        PipelineModelPlan(
            pipeline_id="p",
            steps=[PipelineStep(name="s", models=["m1"])],
        )
    )
    transport.requests.clear()
    pool.teardown()
    pool.teardown()  # second call must not raise

    # Both teardown calls should send the unload — second pass is the
    # idempotency guarantee, not the "do nothing twice" guarantee.
    paths = [(r.method, r.url.path) for r in transport.requests]
    assert paths == [
        ("POST", "/api/models/unload/m1"),
        ("POST", "/api/models/unload/m1"),
    ]


def test_teardown_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    transport = _RecordingTransport(handler=handler)
    _install_transport(monkeypatch, transport)

    pool = ModelPool()
    pool.declare(
        PipelineModelPlan(
            pipeline_id="p",
            steps=[PipelineStep(name="s", models=["m1", "m2"])],
        )
    )
    pool.teardown()  # must not raise


def test_teardown_404_is_treated_as_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """llama-swap returns 404 when the model isn't currently loaded; that's fine."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = _RecordingTransport(handler=handler)
    _install_transport(monkeypatch, transport)

    pool = ModelPool()
    pool.declare(
        PipelineModelPlan(
            pipeline_id="p",
            steps=[PipelineStep(name="s", models=["m1"])],
        )
    )
    pool.teardown()  # must not raise


def test_teardown_before_declare_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _RecordingTransport()
    _install_transport(monkeypatch, transport)

    ModelPool().teardown()
    assert transport.requests == []


# ---------------------------------------------------------------------------
# URL stripping
# ---------------------------------------------------------------------------


def test_trailing_slash_in_url_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _RecordingTransport()
    _install_transport(monkeypatch, transport)

    pool = ModelPool(llama_swap_url="http://swap/")
    pool.declare(
        PipelineModelPlan(
            pipeline_id="p",
            steps=[PipelineStep(name="s", models=["m1"])],
        )
    )
    # No `//` should appear in the path.
    for r in transport.requests:
        assert "//" not in str(r.url).replace("http://", "")


# ---------------------------------------------------------------------------
# Module-level constants sanity
# ---------------------------------------------------------------------------


def test_module_exports_classes() -> None:
    assert hasattr(model_pool, "ModelPool")
    assert hasattr(model_pool, "PipelineModelPlan")
    assert hasattr(model_pool, "PipelineStep")
    assert hasattr(model_pool, "plan_for_cell")
