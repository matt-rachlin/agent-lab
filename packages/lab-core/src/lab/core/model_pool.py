"""Pipeline-aware model lifecycle for the lab.

Sits between sweep-cell / agent-turn orchestration and the llama-swap
proxy on port 8080. Owns three responsibilities:

  1. **Pre-flight pass**: when a pipeline plan is declared, fire a tiny
     completion at each model so its GGUF lands in the OS page cache,
     then explicitly evict it from VRAM. The next real load hits DDR5
     (~0.5-2s) instead of NVMe (~8-15s). Per research: "first cell of a
     sweep is 10-30x slower than subsequent" is fixable.

  2. **Predictive load**: on `step_complete(step_n)`, fire-and-forget a
     warm request for the first model of `step_{n+1}`. By the time the
     caller is ready for that step the model is already in VRAM.

  3. **Explicit teardown**: walks the plan and asks llama-swap to evict
     every model the pipeline declared. Without this we wait for the
     per-model TTL (default 600s) before the slot frees up.

llama-swap API (verified 2026-05-27 against v217):

  * ``GET  /v1/models``               — model registry
  * ``GET  /running``                 — list of currently-loaded models
  * ``POST /api/models/unload``       — unload all models
  * ``POST /api/models/unload/:id``   — unload a single model
  * ``POST /v1/chat/completions``     — OpenAI-compatible inference,
                                        triggers an auto-load if the model
                                        isn't already up

Network failures must never bubble into the caller. The pool is an
optimization layer, not a critical path — a swap-side hiccup must
degrade gracefully (the actual inference call later in the pipeline
will trigger the load anyway, just paying the cold-cache penalty).
"""

from __future__ import annotations

import threading
from typing import Any

import httpx
from pydantic import BaseModel, Field

from lab.observability.log import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Plan data classes
# --------------------------------------------------------------------------


class PipelineStep(BaseModel):
    """One step in a pipeline plan.

    A step is the unit of "we will need these model_ids loaded around the
    same time". The sweep maps a step to a cell; the agent maps a step to
    a turn. Multiple models in one step is the embedder+reranker+LLM
    pattern that fires when a RAG-enabled tool runs.
    """

    name: str
    models: list[str] = Field(
        default_factory=list,
        description="litellm_ids needed in this step (registered in llama-swap).",
    )
    duration_estimate_s: float | None = Field(
        default=None,
        description=(
            "Optional hint for the predictive loader: if we're confident "
            "this step will run >= N seconds, the predictive load of "
            "step+1 has time to finish before the caller needs it."
        ),
    )


class PipelineModelPlan(BaseModel):
    """A declarative description of which models a pipeline touches.

    Declared once by the sweep runner per cell (or the agent solver per
    turn-batch), consumed by :class:`ModelPool` to pre-flight + predict-
    load + tear down. The plan is immutable for the lifetime of the
    pipeline — re-declare for a new cell.
    """

    pipeline_id: str
    steps: list[PipelineStep] = Field(default_factory=list)

    def unique_models(self) -> list[str]:
        """Stable, de-duplicated list of every model_id mentioned in the plan."""

        seen: set[str] = set()
        out: list[str] = []
        for step in self.steps:
            for m in step.models:
                if m not in seen:
                    seen.add(m)
                    out.append(m)
        return out


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------


_PREFLIGHT_TIMEOUT_S = 30.0
"""Wall-time cap on a single pre-flight completion call.

Pre-flight is bounded by NVMe read speed; a 22 GB GGUF on a fast PCIe Gen4
NVMe finishes in ~3-6s. 30s gives plenty of slack for slow disks without
letting a stuck model hold up sweep start indefinitely.
"""


_UNLOAD_TIMEOUT_S = 10.0
"""Wall-time cap on llama-swap unload calls. Should be near-instant."""


_PREDICTIVE_TIMEOUT_S = 30.0
"""Wall-time cap on a predictive (fire-and-forget) load. Same as pre-flight."""


def _preflight_body(model_id: str) -> dict[str, Any]:
    """Minimal /v1/chat/completions body — pure load trigger, no real generation.

    ``max_tokens=1`` keeps the actual decode cheap; ``keep_alive=0`` ensures
    llama-swap (and any underlying Ollama) doesn't pin the model in VRAM
    against our explicit-eviction discipline.
    """

    return {
        "model": model_id,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
        # The Ollama backend behind llama-swap reads keep_alive — set it
        # to 0 so we own the eviction. llama-server ignores it (it has
        # its own TTL machinery, and llama-swap's group rules take over).
        "keep_alive": 0,
    }


# --------------------------------------------------------------------------
# ModelPool
# --------------------------------------------------------------------------


class ModelPool:
    """Client for llama-swap, scoped to one pipeline plan.

    Re-use a single :class:`ModelPool` per pipeline. Re-`declare()` for a
    new plan; the old plan's models are not auto-torn-down on
    re-declaration so callers can decide whether to keep them warm (e.g.
    same-model back-to-back cells should not evict between cells).
    """

    def __init__(self, llama_swap_url: str = "http://localhost:8080") -> None:
        self._url = llama_swap_url.rstrip("/")
        self._plan: PipelineModelPlan | None = None
        self._step_index: dict[str, int] = {}
        self._lock = threading.Lock()
        self._predict_threads: list[threading.Thread] = []

    # ----- public API ------------------------------------------------------

    def declare(self, plan: PipelineModelPlan) -> None:
        """Record the plan and run the pre-flight pass.

        Pre-flight pass: for each unique model in the plan, fire one
        ``max_tokens=1`` completion against llama-swap to trigger a load,
        which:

          1. memory-maps the GGUF (kernel reads the bytes into the page
             cache as it builds the mapping; subsequent loads avoid the
             NVMe seek storm), and
          2. loads the model into VRAM.

        Then immediately ``POST /api/models/unload/:id`` so the VRAM is
        free for the cell's real work. Page cache stays warm.

        Network failures are logged and swallowed — caller is unaffected.
        """

        with self._lock:
            self._plan = plan
            self._step_index = {step.name: idx for idx, step in enumerate(plan.steps)}

        models = plan.unique_models()
        if not models:
            log.info(
                "model_pool_declare_empty",
                pipeline_id=plan.pipeline_id,
            )
            return

        log.info(
            "model_pool_declare",
            pipeline_id=plan.pipeline_id,
            steps=len(plan.steps),
            models=models,
        )

        for model_id in models:
            self._preflight_one(model_id)

    def step_start(self, step_name: str) -> None:
        """Mark the start of a step.

        Placeholder for future per-step timing / instrumentation. Today
        this is a no-op (the cell will trigger its own load on first
        inference call), but recording the boundary lets us add a "if
        the predictive load for THIS step hasn't completed yet, wait
        for it" race-resolution hook later without changing callers.
        """

        if step_name not in self._step_index:
            # Unknown step — log but don't fault. Useful while wiring.
            log.debug(
                "model_pool_step_start_unknown",
                step=step_name,
            )
            return
        log.debug("model_pool_step_start", step=step_name)

    def step_complete(self, step_name: str) -> None:
        """Mark the end of a step and fire the predictive load of step+1.

        Predictive load is fire-and-forget: a daemon thread sends one
        ``max_tokens=1`` warm request for the first model of the next
        step. If the caller proceeds before the load finishes, the
        caller's own inference will just block on the same load — no
        deadlock, no wasted work, just lost predictive value.
        """

        plan = self._plan
        if plan is None or not self._step_index:
            return

        idx = self._step_index.get(step_name)
        if idx is None:
            log.debug("model_pool_step_complete_unknown", step=step_name)
            return

        next_idx = idx + 1
        if next_idx >= len(plan.steps):
            log.debug("model_pool_step_complete_last", step=step_name)
            return

        next_step = plan.steps[next_idx]
        if not next_step.models:
            return

        # Predictive load targets the FIRST model of the next step. If a
        # step needs multiple models, the rest follow on demand.
        target = next_step.models[0]
        log.info(
            "model_pool_predictive_load",
            after_step=step_name,
            next_step=next_step.name,
            model=target,
        )

        t = threading.Thread(
            target=self._warm_one,
            args=(target,),
            name=f"model_pool_predict_{target}",
            daemon=True,
        )
        t.start()
        with self._lock:
            self._predict_threads.append(t)

    def teardown(self) -> None:
        """Evict every model the pipeline declared.

        Idempotent: a second call sees no plan models to evict and the
        llama-swap endpoint is a no-op on a model that isn't loaded.
        Network failures are logged and swallowed.

        Does NOT join in-flight predictive threads — they're daemons and
        the unload call below races them harmlessly (worst case: a
        predictive load completes after the unload and the model is up
        for a few more milliseconds; the next teardown / TTL catches it).
        """

        plan = self._plan
        if plan is None:
            return

        models = plan.unique_models()
        if not models:
            return

        log.info(
            "model_pool_teardown",
            pipeline_id=plan.pipeline_id,
            models=models,
        )
        for model_id in models:
            self._unload_one(model_id)

    # ----- internals -------------------------------------------------------

    def _preflight_one(self, model_id: str) -> None:
        """Fire one max_tokens=1 completion then evict. Failures are swallowed."""

        try:
            with httpx.Client(timeout=_PREFLIGHT_TIMEOUT_S) as client:
                resp = client.post(
                    f"{self._url}/v1/chat/completions",
                    json=_preflight_body(model_id),
                )
                resp.raise_for_status()
            log.debug("model_pool_preflight_ok", model=model_id)
        except Exception as exc:
            log.warning(
                "model_pool_preflight_failed",
                model=model_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return
        # Pre-flight succeeded — evict so VRAM is available for the real load.
        self._unload_one(model_id)

    def _unload_one(self, model_id: str) -> None:
        """POST /api/models/unload/<model_id>. Failures are swallowed."""

        try:
            with httpx.Client(timeout=_UNLOAD_TIMEOUT_S) as client:
                resp = client.post(f"{self._url}/api/models/unload/{model_id}")
                # 200 = ok; 404 means swap doesn't know the model, also fine
                # for our purposes (caller probably misnamed, but we don't
                # gate inference on this).
                if resp.status_code not in (200, 404):
                    resp.raise_for_status()
            log.debug("model_pool_unload_ok", model=model_id, status=resp.status_code)
        except Exception as exc:
            log.warning(
                "model_pool_unload_failed",
                model=model_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    def _warm_one(self, model_id: str) -> None:
        """Fire a max_tokens=1 request to coax the model into VRAM. Failures swallowed.

        Distinct from pre-flight in that we do NOT unload after — the
        whole point is to leave the model loaded so the next step's
        first call doesn't pay the cold-load tax.
        """

        try:
            with httpx.Client(timeout=_PREDICTIVE_TIMEOUT_S) as client:
                resp = client.post(
                    f"{self._url}/v1/chat/completions",
                    json=_preflight_body(model_id),
                )
                resp.raise_for_status()
            log.debug("model_pool_warm_ok", model=model_id)
        except Exception as exc:
            log.warning(
                "model_pool_warm_failed",
                model=model_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )


# --------------------------------------------------------------------------
# Convenience helpers for callers (sweep runner, agent solver)
# --------------------------------------------------------------------------


def plan_for_cell(
    *,
    pipeline_id: str,
    model_id: str,
    tools: list[dict[str, Any]] | None = None,
    embedder_model_id: str | None = "qwen3-embedding",
    reranker_model_id: str | None = "qwen3-reranker-0.6b",
) -> PipelineModelPlan:
    """Build a one-step plan from a sweep cell's (model, tools) tuple.

    Today every cell is one step; the embedder + reranker join the step
    iff the cell can fire ``kb_query``. The default model_ids here match
    the lab.models registry; pass ``None`` to omit a side model.
    """

    models: list[str] = [model_id]
    uses_kb = bool(tools) and any(
        isinstance(t, dict) and t.get("name") == "kb_query" for t in (tools or [])
    )
    if uses_kb:
        if embedder_model_id:
            models.append(embedder_model_id)
        if reranker_model_id:
            models.append(reranker_model_id)

    return PipelineModelPlan(
        pipeline_id=pipeline_id,
        steps=[PipelineStep(name="cell", models=models)],
    )
