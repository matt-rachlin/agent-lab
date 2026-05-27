"""SweepRunner: execute a comparison sweep over the (model, config, task, seed) matrix."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

import psycopg
from psycopg.types.json import Json
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from lab.core.gpu_lease import force_release, gpu_lease
from lab.core.gpu_lease import status as gpu_lease_status
from lab.core.manifest import capture as capture_manifest
from lab.core.settings import get_settings
from lab.observability.log import bind_run_context, clear_run_context, get_logger
from lab.observability.tracing import current_span_attrs, span
from lab.sweep.config import RunConfig, SweepConfig, config_hash, run_id
from lab.tasks.registry import get_tasks

console = Console()
log = get_logger(__name__)


# ----------------------------------------------------------------------------
# PID-file convention for inter-process status/cancel
# ----------------------------------------------------------------------------

PIDFILE_DIR = Path("/data/lab/services/sweep-pids")

# Sandbox image hash file. Recorded by `lab agent sandbox build`; read by
# the in-sweep drift guard (see `ImageHashDriftError`).
_SANDBOX_IMAGE_HASH_PATH = Path("conf/sandbox-image.sha")


class ImageHashDriftError(RuntimeError):
    """Raised when the sandbox image hash changes mid-sweep.

    F-005 EXP-002 follow-up: during EXP-002 the sweep saw three distinct
    `sandbox_image_hash` values for cells nominally descended from the same
    Containerfile commit. Root cause: a background `podman image prune`
    reaped layers between cells, triggering Containerfile rebuilds the next
    time the sweep pulled the image. The launch-time preflight catches
    mismatches against the registered experiment but NOT within-sweep
    drift — this guard closes the gap.

    The error message intentionally names both hashes so the operator can
    diff Containerfile and figure out which rebuild produced the second
    image. The sweep aborts cleanly (no further cells run) but already-
    executed cells stay in the DB tagged with whichever hash they ran
    against — F-005 analysis can still attribute results.
    """


def _read_sandbox_image_hash() -> str | None:
    """Return the recorded sandbox image hash, or None if absent.

    Written by `lab agent sandbox build`. Returns None if the file is
    missing or empty — the drift guard treats `None` as "no hash known"
    and silently skips the check (so non-agent sweeps don't blow up).
    """

    if not _SANDBOX_IMAGE_HASH_PATH.exists():
        return None
    try:
        text = _SANDBOX_IMAGE_HASH_PATH.read_text().strip()
    except OSError:
        return None
    return text or None


def _pidfile_for(slug: str) -> Path:
    return PIDFILE_DIR / f"{slug}.pid"


def _write_pidfile(slug: str) -> Path | None:
    """Write our PID into the slug's pidfile. Returns the path, or None on failure."""
    try:
        PIDFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = _pidfile_for(slug)
        path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        return path
    except OSError as exc:
        log.warning("sweep_pidfile_write_failed", slug=slug, error=str(exc))
        return None


def _clear_pidfile(slug: str) -> None:
    with contextlib.suppress(OSError):
        _pidfile_for(slug).unlink(missing_ok=True)


def read_pidfile(slug: str) -> int | None:
    """Return the PID recorded for an active sweep, or None if none/stale."""
    path = _pidfile_for(slug)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    # Stale-check: signal 0 raises if the process doesn't exist
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid  # exists, just not ours
    return pid


# ----------------------------------------------------------------------------
# Signal handling: SIGTERM/SIGINT → release lease, exit cleanly. Idempotent.
# ----------------------------------------------------------------------------


_shutdown_requested = False


def _install_signal_handlers(slug: str) -> None:
    """Trap SIGTERM and SIGINT: release the GPU lease and exit cleanly.

    Idempotent — running twice is a no-op. Safe to call from non-main thread
    contexts (we only call signal.signal here, which Python guards).
    """

    def _handler(signum: int, _frame: FrameType | None) -> None:
        global _shutdown_requested  # noqa: PLW0603 — module-level flag, by design
        if _shutdown_requested:
            return
        _shutdown_requested = True
        sig_name = signal.Signals(signum).name
        log.warning("sweep_shutdown_signal", signal=sig_name, slug=slug)
        try:
            holder, _ttl = gpu_lease_status()
            if holder:
                force_release()
        except Exception as exc:
            log.error("gpu_lease_release_failed", slug=slug, error=str(exc))
        _clear_pidfile(slug)
        sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        # Skip silently if we're not on the main thread
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handler)


@dataclass(frozen=True)
class Cell:
    """One (model, config, task, seed) cell in the sweep matrix."""

    run_id: str
    experiment_id: int | None
    experiment_slug: str
    model_id: int
    model_litellm_id: str
    model_backend: str
    task_id: int
    task_slug: str
    task_payload: dict[str, Any]
    config: RunConfig
    config_hash: str
    seed: int


@dataclass
class CellResult:
    """Outcome of running one cell."""

    run_id: str
    status: str
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: int
    cost_usd: float | None
    error: str | None
    response_text: str | None
    raw_response: dict[str, Any] | None


# ----------------------------------------------------------------------------
# DB helpers
# ----------------------------------------------------------------------------


def _ensure_experiment(spec: SweepConfig) -> int | None:
    """Find or create the experiment row referenced by the sweep."""
    if spec.experiment.create_if_missing is False:
        return None
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT experiment_id FROM experiments WHERE slug = %s",
            (spec.experiment.slug,),
        )
        row = cur.fetchone()
        if row:
            return int(row[0])
        cur.execute(
            """
            INSERT INTO experiments (slug, title, hypothesis, status, plan_path, created_at)
            VALUES (%s, %s, %s, 'planned', %s, NOW())
            RETURNING experiment_id
            """,
            (
                spec.experiment.slug,
                spec.experiment.title or spec.experiment.slug,
                spec.experiment.hypothesis,
                spec.experiment.plan_path or f"docs/exp/{spec.experiment.slug}.md",
            ),
        )
        new_row = cur.fetchone()
        return int(new_row[0]) if new_row else None


def _models_lookup(litellm_ids: list[str]) -> dict[str, tuple[int, str]]:
    """{litellm_id: (model_id, backend)} for the requested models."""
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT model_id, litellm_id, backend FROM models WHERE litellm_id = ANY(%s)",
            (litellm_ids,),
        )
        return {row[1]: (int(row[0]), row[2]) for row in cur.fetchall()}


def _done_run_ids(experiment_id: int | None) -> set[str]:
    """Run IDs already in the runs table for this experiment (any non-error status)."""
    if experiment_id is None:
        return set()
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT run_id FROM experiment_runs WHERE experiment_id = %s AND status = 'done'",
            (experiment_id,),
        )
        return {row[0] for row in cur.fetchall()}


# ----------------------------------------------------------------------------
# Matrix expansion
# ----------------------------------------------------------------------------


def expand_matrix(
    spec: SweepConfig,
    experiment_id: int | None,
    models: dict[str, tuple[int, str]],
) -> list[Cell]:
    """Cartesian product of the sweep matrix."""
    rows = get_tasks(spec.tasks.suite, spec.tasks.slugs)
    if not rows:
        raise ValueError(f"no tasks found in suite={spec.tasks.suite!r}")
    cells: list[Cell] = []
    for cfg in spec.configs:
        chash = config_hash(cfg)
        for model_name in spec.models:
            if model_name not in models:
                raise ValueError(f"model {model_name!r} not registered in lab.models")
            mid, backend = models[model_name]
            for task in rows:
                for seed in spec.seeds:
                    rid = run_id(
                        spec.experiment.slug,
                        model_name,
                        task["slug"],
                        chash,
                        seed,
                    )
                    cells.append(
                        Cell(
                            run_id=rid,
                            experiment_id=experiment_id,
                            experiment_slug=spec.experiment.slug,
                            model_id=mid,
                            model_litellm_id=model_name,
                            model_backend=backend,
                            task_id=int(task["task_id"]),
                            task_slug=task["slug"],
                            task_payload=task["payload"],
                            config=cfg,
                            config_hash=chash,
                            seed=seed,
                        )
                    )
    return cells


# ----------------------------------------------------------------------------
# Single-cell execution
# ----------------------------------------------------------------------------


def _build_messages(
    task_payload: dict[str, Any],
    config_system: str | None = None,
    model_default_system: str | None = None,
) -> list[dict[str, str]]:
    """Build the chat-completion message list.

    Precedence for the system message (highest → lowest):
        1. task-level `system` field (from the Task schema)
        2. task-level `system_prompt_id` resolved via :class:`PromptRegistry`
           (Phase 16.4 — only one of {system, system_prompt_id} is allowed)
        3. sweep `model_defaults[<litellm_id>].system_prompt`
        4. RunConfig.extra `system_prompt`
    """
    messages: list[dict[str, str]] = []
    system: str | None = task_payload.get("system")
    if system is None:
        prompt_id = task_payload.get("system_prompt_id")
        if prompt_id:
            # Local import: keep sweep package light when system_prompt_id
            # is unused. Failures here are loud (PromptNotFoundError) on
            # purpose — a typo'd id should not silently fall through.
            from lab.eval.prompts import PromptRegistry

            system = PromptRegistry().get(str(prompt_id))
    if system is None:
        system = model_default_system or config_system
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(task_payload["input"])})
    return messages


def _is_local_backend(backend: str) -> bool:
    return backend == "ollama-local"


def _call_litellm(
    *,
    settings: Any,
    litellm_key: str,
    model: str,
    messages: list[dict[str, str]],
    config: RunConfig,
    timeout: int,
) -> tuple[dict[str, Any], int]:
    """Hit the LiteLLM proxy; returns (response_json, latency_ms).

    Single-turn fast path. Thin wrapper around `lab.llm.call_litellm_chat`
    so the multi-turn agent solver and this path share the same request
    shape.
    """
    from lab.core.llm import call_litellm_chat

    return call_litellm_chat(
        settings=settings,
        litellm_key=litellm_key,
        model=model,
        messages=messages,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        extra=config.extra or None,
        timeout=timeout,
    )


def _persist_trace(*, run_id_: str, payload: dict[str, Any]) -> str:
    """Upload trace JSONL to MinIO. Returns the s3:// path."""
    from lab.core.minio_io import run_key, upload_bytes

    data = (json.dumps(payload) + "\n").encode()
    return upload_bytes(
        key=run_key(run_id_, "trace.jsonl"),
        data=data,
        content_type="application/x-ndjson",
    )


def _insert_run(
    *,
    cell: Cell,
    result: CellResult,
    manifest_sha: str,
    trace_path: str | None,
) -> None:
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO experiment_runs
                (run_id, experiment_id, model_id, task_id, config_hash, config, seed,
                 status, manifest_sha, trace_path, tokens_in, tokens_out, latency_ms,
                 cost_usd, error, started_at, completed_at)
            VALUES
                (%(run_id)s, %(experiment_id)s, %(model_id)s, %(task_id)s, %(config_hash)s,
                 %(config)s, %(seed)s, %(status)s, %(manifest_sha)s, %(trace_path)s,
                 %(tokens_in)s, %(tokens_out)s, %(latency_ms)s, %(cost_usd)s, %(error)s,
                 NOW(), NOW())
            ON CONFLICT (run_id) DO UPDATE SET
                status = EXCLUDED.status,
                manifest_sha = EXCLUDED.manifest_sha,
                trace_path = EXCLUDED.trace_path,
                tokens_in = EXCLUDED.tokens_in,
                tokens_out = EXCLUDED.tokens_out,
                latency_ms = EXCLUDED.latency_ms,
                cost_usd = EXCLUDED.cost_usd,
                error = EXCLUDED.error,
                completed_at = NOW();
            """,
            {
                "run_id": cell.run_id,
                "experiment_id": cell.experiment_id,
                "model_id": cell.model_id,
                "task_id": cell.task_id,
                "config_hash": cell.config_hash,
                "config": Json(cell.config.model_dump()),
                "seed": cell.seed,
                "status": result.status,
                "manifest_sha": manifest_sha,
                "trace_path": trace_path,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "latency_ms": result.latency_ms,
                "cost_usd": result.cost_usd,
                "error": result.error,
            },
        )
    # Phase 15.2: additive mirror into MLflow. Best-effort, never blocks.
    _mirror_cell_to_mlflow(
        cell=cell,
        result=result,
        trace_path=trace_path,
        sandbox_image_hash=None,
        actual_turns=None,
        tool_call_count=None,
    )


def _mirror_cell_to_mlflow(
    *,
    cell: Cell,
    result: CellResult,
    trace_path: str | None,
    sandbox_image_hash: str | None,
    actual_turns: int | None,
    tool_call_count: int | None,
) -> None:
    """Mirror one cell into MLflow + write back the assigned mlflow_run_id."""

    try:
        from lab.observability.mlflow_mirror import MlflowMirror

        metrics: dict[str, float] = {}
        if result.latency_ms is not None:
            metrics["latency_ms"] = float(result.latency_ms)
        if result.tokens_in is not None:
            metrics["tokens_in"] = float(result.tokens_in)
        if result.tokens_out is not None:
            metrics["tokens_out"] = float(result.tokens_out)
        if result.cost_usd is not None:
            metrics["cost_usd"] = float(result.cost_usd)
        if actual_turns is not None:
            metrics["actual_turns"] = float(actual_turns)
        if tool_call_count is not None:
            metrics["tool_call_count"] = float(tool_call_count)

        tags: dict[str, str] = {
            "model_backend": cell.model_backend,
            "config_hash": cell.config_hash,
            "config_name": cell.config.name,
        }
        if sandbox_image_hash:
            tags["sandbox_image_hash"] = sandbox_image_hash

        mlflow_run_id = MlflowMirror().log_run(
            cell.experiment_slug,
            cell.run_id,
            model=cell.model_litellm_id,
            task=cell.task_slug,
            seed=cell.seed,
            config=cell.config.model_dump(),
            metrics=metrics,
            tags=tags,
            artifact_uri=trace_path,
            status="FAILED" if result.status == "error" else "FINISHED",
        )
        if mlflow_run_id:
            with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE experiment_runs SET mlflow_run_id = %s WHERE run_id = %s",
                    (mlflow_run_id, cell.run_id),
                )
    except Exception:  # noqa: S110 — belt-and-suspenders; mirror already logs
        # The mirror already swallows everything; this is belt-and-suspenders.
        pass


def _is_agent_cell(task_payload: dict[str, Any]) -> bool:
    """Return True if this cell needs the multi-turn agent path.

    Tasks shipped before Phase 6 default to `max_turns=1` and `tool_budget=0`
    so they continue to go through the single-turn fast path untouched.
    """

    max_turns = int(task_payload.get("max_turns") or 1)
    tool_budget = int(task_payload.get("tool_budget") or 0)
    return max_turns > 1 or tool_budget > 0


def execute_cell(
    cell: Cell,
    *,
    litellm_key: str,
    timeout: int,
    model_default_system: str | None = None,
    model_default_extra: dict[str, Any] | None = None,
) -> CellResult:
    """Execute one matrix cell.

    Dispatches between the single-turn fast path (Phase 1-5 behaviour,
    unchanged) and the multi-turn agent path (Phase 6+). The fast path is
    preserved bit-for-bit - only cells whose task declares `max_turns > 1`
    or `tool_budget > 0` go through Inspect.
    """

    is_agent = _is_agent_cell(cell.task_payload)
    bind_run_context(
        run_id=cell.run_id,
        experiment_slug=cell.experiment_slug,
        model=cell.model_litellm_id,
        task=cell.task_slug,
        seed=cell.seed,
        config_hash=cell.config_hash,
    )
    try:
        with span(
            "sweep_cell",
            **{
                "lab.run_id": cell.run_id,
                "lab.experiment_slug": cell.experiment_slug,
                "lab.model": cell.model_litellm_id,
                "lab.task": cell.task_slug,
                "lab.seed": cell.seed,
                "lab.config_hash": cell.config_hash,
                "lab.path": "agent" if is_agent else "single_turn",
            },
        ):
            log.info("sweep_cell_started", path="agent" if is_agent else "single_turn")
            with span("manifest_capture"):
                manifest = capture_manifest(
                    extra={
                        "kind": "sweep_run",
                        "experiment_slug": cell.experiment_slug,
                        "model": cell.model_litellm_id,
                        "task_slug": cell.task_slug,
                        "config_hash": cell.config_hash,
                        "seed": cell.seed,
                        "run_id": cell.run_id,
                    }
                )

            if is_agent:
                result = _execute_agent_cell(
                    cell=cell,
                    manifest_sha=manifest.sha,
                    timeout=timeout,
                    model_default_extra=model_default_extra,
                )
            else:
                result = _execute_single_turn(
                    cell=cell,
                    litellm_key=litellm_key,
                    timeout=timeout,
                    manifest_sha=manifest.sha,
                    model_default_system=model_default_system,
                    model_default_extra=model_default_extra,
                )

            current_span_attrs(
                **{
                    "lab.status": result.status,
                    "lab.latency_ms": result.latency_ms,
                    "lab.tokens_in": result.tokens_in,
                    "lab.tokens_out": result.tokens_out,
                }
            )
            log.info(
                "sweep_cell_finished",
                status=result.status,
                latency_ms=result.latency_ms,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                error=result.error,
            )
            return result
    finally:
        clear_run_context()


def _execute_single_turn(
    *,
    cell: Cell,
    litellm_key: str,
    timeout: int,
    manifest_sha: str,
    model_default_system: str | None,
    model_default_extra: dict[str, Any] | None = None,
) -> CellResult:
    """Phase 1-5 fast path: one LiteLLM call, one trace row, no Inspect."""

    settings = get_settings()
    config_system = cell.config.extra.get("system_prompt") if cell.config.extra else None
    if config_system is None:
        config_system = getattr(cell.config, "system_prompt", None)
    messages = _build_messages(
        cell.task_payload,
        config_system=config_system,
        model_default_system=model_default_system,
    )
    # Merge per-model extra over config.extra. Per-model wins on key clash;
    # system_prompt is consumed locally and not forwarded.
    merged_extra: dict[str, Any] = {}
    if cell.config.extra:
        merged_extra.update(cell.config.extra)
    if model_default_extra:
        merged_extra.update(model_default_extra)
    # Materialise a config copy with merged extra so _call_litellm forwards it.
    cell_config_for_call = cell.config.model_copy(update={"extra": merged_extra})
    result: CellResult

    try:
        if _is_local_backend(cell.model_backend):
            with (
                span("gpu_lease_acquire", **{"lab.model": cell.model_litellm_id}),
                gpu_lease(
                    f"sweep:{cell.experiment_slug}:{cell.model_litellm_id}",
                    ttl_sec=timeout + 60,
                ),
                span("litellm_call", **{"lab.model": cell.model_litellm_id}),
            ):
                resp_json, latency_ms = _call_litellm(
                    settings=settings,
                    litellm_key=litellm_key,
                    model=cell.model_litellm_id,
                    messages=messages,
                    config=cell_config_for_call,
                    timeout=timeout,
                )
                current_span_attrs(**{"lab.latency_ms": latency_ms})
        else:
            with span("litellm_call", **{"lab.model": cell.model_litellm_id}):
                resp_json, latency_ms = _call_litellm(
                    settings=settings,
                    litellm_key=litellm_key,
                    model=cell.model_litellm_id,
                    messages=messages,
                    config=cell.config,
                    timeout=timeout,
                )
                current_span_attrs(**{"lab.latency_ms": latency_ms})

        usage = resp_json.get("usage") or {}
        message = ((resp_json.get("choices") or [{}])[0]).get("message") or {}
        text = message.get("content") or ""

        result = CellResult(
            run_id=cell.run_id,
            status="done",
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            cost_usd=None,  # filled in Phase 4 from LiteLLM spend ledger
            error=None,
            response_text=text,
            raw_response=resp_json,
        )
    except Exception as exc:  # any failure → record error, don't crash sweep
        log.error(
            "single_turn_cell_error",
            error=f"{type(exc).__name__}: {exc}",
        )
        result = CellResult(
            run_id=cell.run_id,
            status="error",
            tokens_in=None,
            tokens_out=None,
            latency_ms=0,
            cost_usd=None,
            error=f"{type(exc).__name__}: {exc}",
            response_text=None,
            raw_response=None,
        )

    trace_path: str | None = None
    if result.raw_response is not None:
        with span("persist"):
            try:
                trace_path = _persist_trace(
                    run_id_=cell.run_id,
                    payload={
                        "run_id": cell.run_id,
                        "experiment_slug": cell.experiment_slug,
                        "manifest_sha": manifest_sha,
                        "model": cell.model_litellm_id,
                        "task_slug": cell.task_slug,
                        "config": cell.config.model_dump(),
                        "seed": cell.seed,
                        "input_messages": messages,
                        "response_text": result.response_text,
                        "raw_response": result.raw_response,
                        "latency_ms": result.latency_ms,
                    },
                )
            except Exception as exc:
                log.warning(
                    "trace_upload_failed",
                    run_id=cell.run_id,
                    error=str(exc),
                )

    _insert_run(cell=cell, result=result, manifest_sha=manifest_sha, trace_path=trace_path)
    return result


def _execute_agent_cell(
    *,
    cell: Cell,
    manifest_sha: str,
    timeout: int,
    model_default_extra: dict[str, Any] | None = None,
) -> CellResult:
    """Phase 6 path: build an Inspect task, run it, mirror the log into Postgres + MinIO."""

    from lab.agent.sandbox import Sandbox
    from lab.inspect_bridge.adapter import lab_task_to_inspect
    from lab.inspect_bridge.logwriter import SweepContext, write_run_from_inspect_log
    from lab.tasks.registry import Task as LabTask

    payload = cell.task_payload

    # EXP-003b: allow the sweep config to ablate kb_query out of the tools
    # list. The config carries `extra.tool_filter: exclude_kb_query`; we
    # apply it BEFORE LabTask construction so the rest of the harness
    # (kb_mount detection, adapter scorer selection) sees the filtered tools.
    raw_tools = payload.get("tools")
    cfg_extra = cell.config.extra or {}
    tool_filter = cfg_extra.get("tool_filter")
    if tool_filter == "exclude_kb_query" and raw_tools:
        raw_tools = [
            t for t in raw_tools if not (isinstance(t, dict) and t.get("name") == "kb_query")
        ]

    lab_task = LabTask.model_validate(
        {
            "suite": payload.get("suite", "agent"),
            "slug": cell.task_slug,
            "category": payload.get("category"),
            "input": payload["input"],
            "system": payload.get("system"),
            # Phase 16.4: tasks can reference a prompt by id; the adapter
            # resolves it via PromptRegistry at build time.
            "system_prompt_id": payload.get("system_prompt_id"),
            "tools": raw_tools,
            "max_turns": payload.get("max_turns", 1),
            "tool_budget": payload.get("tool_budget", 0),
            "success_predicate": payload.get("success_predicate"),
            "sandbox": payload.get("sandbox"),
            "gold_answer": payload.get("gold_answer"),
            "rubric": payload.get("rubric"),
            "description": payload.get("description"),
        }
    )

    sandbox_cfg = lab_task.sandbox or {}
    network: str | list[str] = sandbox_cfg.get("network", "none")
    env: dict[str, str] = dict(sandbox_cfg.get("env", {}))
    workspace_files_raw: dict[str, str] | None = sandbox_cfg.get("workspace_files")
    workspace_files: dict[str, bytes] = {
        k: v.encode("utf-8") if isinstance(v, str) else v
        for k, v in (workspace_files_raw or {}).items()
    }

    from lab.agent.tools import (
        task_needs_hf_cache_mount as _task_needs_hf_cache_mount,
    )
    from lab.agent.tools import task_needs_kb_mount as _task_needs_kb_mount

    kb_root_mount: Path | None = None
    if _task_needs_kb_mount(lab_task.tools):
        from lab.core.settings import get_settings as _get_settings_kb

        kb_root_mount = _get_settings_kb().kb_root
        env.setdefault("LAB_KB_ROOT", "/kb")
        # See cli.py twin: the kb_query tool embeds queries via Ollama;
        # `localhost` inside the sandbox is the container, not the host,
        # so route through podman's host-internal alias.
        env.setdefault("OLLAMA_HOST", "http://host.containers.internal:11434")
        # Force bridge networking with host.containers.internal allowed so
        # the sandbox can actually reach Ollama. sandbox.py has a magic-name
        # carve-out so this doesn't try a host-side `getaddrinfo` lookup.
        if network == "none":
            network = ["host.containers.internal"]
        elif isinstance(network, list) and "host.containers.internal" not in network:
            network = [*network, "host.containers.internal"]

    # Phase 7 reranker: persistent HF cache across cells. Parallels the
    # cli.py twin — mount the host hf_cache_root rw at /hf-cache and point
    # HF_HOME / TRANSFORMERS_CACHE at it. Force offline mode because the
    # sandbox network only allows ``host.containers.internal``: a cache
    # miss would silently fall through to stage-1, hiding misconfigs.
    # Skipped when the task can't trigger the reranker (no kb_query) or
    # LAB_RAG_RERANKER=none in the sweep env.
    # Propagate the host's LAB_RAG_RERANKER (if any) so the sandbox tool
    # surface honours the same disable/select as the host. Done BEFORE the
    # mount-needs check so the heuristic sees the propagated value.
    import os as _os

    _host_reranker = _os.environ.get("LAB_RAG_RERANKER")
    if _host_reranker is not None:
        env.setdefault("LAB_RAG_RERANKER", _host_reranker)

    hf_cache_mount: Path | None = None
    if _task_needs_hf_cache_mount(lab_task.tools, reranker_env=env.get("LAB_RAG_RERANKER")):
        from lab.core.settings import get_settings as _get_settings_hf

        hf_cache_root = _get_settings_hf().hf_cache_root
        hf_cache_root.mkdir(parents=True, exist_ok=True)
        hf_cache_mount = hf_cache_root
        env.setdefault("HF_HOME", "/hf-cache")
        env.setdefault("TRANSFORMERS_CACHE", "/hf-cache/transformers")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Phase 7.1: route in-sandbox reranks to the host-side service.
        # Without LAB_RAG_RERANKER_URL the slim sandbox image (no
        # sentence-transformers/torch) would fail back to pass-through.
        _rerank_port = _os.environ.get("LAB_RAG_RERANKER_PORT", "8401")
        env.setdefault(
            "LAB_RAG_RERANKER_URL",
            f"http://host.containers.internal:{_rerank_port}",
        )

    result: CellResult
    sweep_ctx = SweepContext(
        run_id=cell.run_id,
        experiment_id=cell.experiment_id,
        experiment_slug=cell.experiment_slug,
        model_id=cell.model_id,
        model_litellm_id=cell.model_litellm_id,
        task_id=cell.task_id,
        task_slug=cell.task_slug,
        config_hash=cell.config_hash,
        config=cell.config.model_dump(),
        seed=cell.seed,
        manifest_sha=manifest_sha,
    )

    try:
        from inspect_ai import eval as inspect_eval

        with Sandbox(
            network=network,
            env=env,
            workspace_files=workspace_files,
            time_limit_sec=timeout,
            kb_root_mount=kb_root_mount,
            hf_cache_mount=hf_cache_mount,
            hf_cache_target="/hf-cache",
        ) as sandbox:
            # Merge per-model `extra` over config.extra (per-model wins).
            merged_extra: dict[str, Any] = {}
            if cell.config.extra:
                merged_extra.update(cell.config.extra)
            if model_default_extra:
                merged_extra.update(model_default_extra)
            # system_prompt is consumed locally and not forwarded as a
            # backend knob; drop it before plumbing to the solver.
            merged_extra.pop("system_prompt", None)
            # tool_filter is a sweep-level ablation knob (EXP-003b), already
            # applied above to lab_task.tools; not a backend knob.
            merged_extra.pop("tool_filter", None)

            inspect_task = lab_task_to_inspect(
                lab_task,
                model=cell.model_litellm_id,
                sandbox=sandbox,
                temperature=cell.config.temperature,
                max_tokens=cell.config.max_tokens,
                extra=merged_extra or None,
            )
            import tempfile

            # The Inspect EvalLog is lazy-loaded from the .eval file on disk;
            # we must keep the log_dir alive until we have finished reading
            # samples / metadata out of it. Persistence and metric-extraction
            # therefore happen INSIDE the TemporaryDirectory `with` block.
            with tempfile.TemporaryDirectory(prefix="lab-inspect-") as log_dir:
                if _is_local_backend(cell.model_backend):
                    with (
                        span(
                            "gpu_lease_acquire",
                            **{"lab.model": cell.model_litellm_id},
                        ),
                        gpu_lease(
                            f"sweep:{cell.experiment_slug}:{cell.model_litellm_id}",
                            ttl_sec=timeout + 60,
                        ),
                        span("inspect_eval", **{"lab.model": cell.model_litellm_id}),
                    ):
                        logs = inspect_eval(
                            inspect_task,
                            display="none",
                            log_samples=True,
                            log_dir=log_dir,
                            log_format="json",
                            log_realtime=False,
                        )
                else:
                    with span("inspect_eval", **{"lab.model": cell.model_litellm_id}):
                        logs = inspect_eval(
                            inspect_task,
                            display="none",
                            log_samples=True,
                            log_dir=log_dir,
                            log_format="json",
                            log_realtime=False,
                        )
                eval_log = logs[0] if logs else None
                if eval_log is None:
                    raise RuntimeError("inspect_ai.eval returned no logs")
                with span("persist"):
                    trace_uri = write_run_from_inspect_log(eval_log, sweep_ctx)
                # Read back the aggregated metrics we just upserted so the
                # in-memory CellResult matches what's in the DB.
                samples = getattr(eval_log, "samples", None) or []
                sample = samples[0] if samples else None
                lab_agent: dict[str, Any] = {}
                if sample is not None:
                    metadata = sample.metadata or {}
                    lab_agent = metadata.get("lab_agent") or {}
                # Extract usage/latency while the .eval file is still on disk.
                latency_ms = int(lab_agent.get("total_latency_ms") or 0)
                usage = getattr(sample, "model_usage", None) if sample is not None else None
                if usage is not None and hasattr(usage, "model_dump"):
                    usage = usage.model_dump()
                tokens_in: int | None = None
                tokens_out: int | None = None
                for v in (usage or {}).values():
                    if isinstance(v, dict):
                        if v.get("input_tokens") is not None:
                            tokens_in = (tokens_in or 0) + int(v["input_tokens"])
                        if v.get("output_tokens") is not None:
                            tokens_out = (tokens_out or 0) + int(v["output_tokens"])
        result = CellResult(
            run_id=cell.run_id,
            status="error" if lab_agent.get("error") else "done",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=None,
            error=lab_agent.get("error"),
            response_text=None,
            raw_response={"trajectory_key": trace_uri, "lab_agent": lab_agent},
        )
    except Exception as exc:
        log.error(
            "agent_cell_error",
            run_id=cell.run_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        result = CellResult(
            run_id=cell.run_id,
            status="error",
            tokens_in=None,
            tokens_out=None,
            latency_ms=0,
            cost_usd=None,
            error=f"{type(exc).__name__}: {exc}",
            response_text=None,
            raw_response=None,
        )
        # Still record the error row in experiment_runs so the sweep
        # bookkeeping survives.
        _insert_run(cell=cell, result=result, manifest_sha=manifest_sha, trace_path=None)
    return result


# ----------------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------------


def run_sweep(
    spec: SweepConfig,
    *,
    litellm_key: str,
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the full sweep. Returns a summary dict."""
    # Preflight: refuse to start if proxy config has Ollama models without keep_alive
    if not dry_run:
        preflight_litellm_keep_alive_or_raise()
    experiment_id = _ensure_experiment(spec)
    models = _models_lookup(spec.models)
    missing = sorted(set(spec.models) - set(models))
    if missing:
        raise ValueError(f"models not registered in lab.models: {missing}")

    cells = expand_matrix(spec, experiment_id, models)
    done = _done_run_ids(experiment_id) if resume else set()
    todo = [c for c in cells if c.run_id not in done]

    console.print(
        f"[bold]sweep[/]: experiment={spec.experiment.slug} "
        f"cells={len(cells)} done={len(done)} todo={len(todo)}"
    )
    if dry_run:
        console.print("[yellow]dry-run: not executing")
        return {"total": len(cells), "done": len(done), "todo": len(todo), "executed": 0}

    _install_signal_handlers(spec.experiment.slug)
    _write_pidfile(spec.experiment.slug)

    # Group by model to minimize swap cost (outer = model)
    todo_sorted = sorted(
        todo, key=lambda c: (c.model_litellm_id, c.config.name, c.task_slug, c.seed)
    )

    # Cache the sandbox image hash at sweep start. The drift guard re-reads
    # `conf/sandbox-image.sha` before each cell and aborts if the value
    # changes. `None` at start (no agent cells / non-agent sweep) disables
    # the check entirely; `None` mid-sweep means the file vanished, which
    # is also drift.
    starting_image_hash: str | None = _read_sandbox_image_hash()
    summary: dict[str, Any] = {
        "total": len(cells),
        "done_before": len(done),
        "executed": 0,
        "errors": 0,
        "starting_image_hash": starting_image_hash,
        "image_hash_drift": None,
    }
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            bar = progress.add_task("sweeping", total=len(todo_sorted))
            for cell in todo_sorted:
                # Image-hash drift guard. If we know a starting hash, refuse
                # to dispatch a cell whose image hash disagrees. We re-read
                # the file each iteration so an out-of-band rebuild (e.g.
                # `podman image prune` reaping layers + a rebuild on the
                # next pull) trips the guard the same way a manual change
                # would. Sweep aborts cleanly; the already-executed cells
                # remain in the DB tagged with their actual hash, so F-005-
                # style attribution still works.
                if starting_image_hash is not None:
                    current_image_hash = _read_sandbox_image_hash()
                    if current_image_hash != starting_image_hash:
                        summary["image_hash_drift"] = {
                            "starting": starting_image_hash,
                            "current": current_image_hash,
                            "at_cell": cell.run_id,
                            "executed": summary["executed"],
                        }
                        progress.console.log(
                            f"[red]image_hash_drift[/]: "
                            f"starting={starting_image_hash[:12]}, "
                            f"current={(current_image_hash or 'MISSING')[:12]}, "
                            f"at_cell={cell.run_id}; "
                            f"aborting sweep after {summary['executed']} cells"
                        )
                        log.error(
                            "sweep_image_hash_drift",
                            starting=starting_image_hash,
                            current=current_image_hash,
                            at_cell=cell.run_id,
                            executed=summary["executed"],
                        )
                        raise ImageHashDriftError(
                            "sandbox image hash drifted mid-sweep: "
                            f"starting={starting_image_hash}, "
                            f"current={current_image_hash}, "
                            f"at_cell={cell.run_id}, "
                            f"executed={summary['executed']}/{len(todo_sorted)}"
                        )

                model_default_system = None
                model_default_extra: dict[str, Any] | None = None
                md = spec.model_defaults.get(cell.model_litellm_id)
                if md is not None:
                    model_default_system = md.system_prompt
                    model_default_extra = dict(md.extra) if md.extra else None
                result = execute_cell(
                    cell,
                    litellm_key=litellm_key,
                    timeout=spec.request_timeout_sec,
                    model_default_system=model_default_system,
                    model_default_extra=model_default_extra,
                )
                summary["executed"] += 1
                if result.status == "error":
                    summary["errors"] += 1
                    progress.console.log(
                        f"[red]ERROR[/] {cell.model_litellm_id} {cell.task_slug} seed={cell.seed}: {result.error}"
                    )
                    log.error(
                        "sweep_cell_error",
                        model=cell.model_litellm_id,
                        task=cell.task_slug,
                        seed=cell.seed,
                        run_id=cell.run_id,
                        error=result.error,
                    )
                progress.update(bar, advance=1)
    finally:
        _clear_pidfile(spec.experiment.slug)

    # Mark experiment running/completed
    if experiment_id is not None:
        with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE experiments
                   SET status = 'running',
                       started_at = COALESCE(started_at, NOW())
                 WHERE experiment_id = %s
                """,
                (experiment_id,),
            )

    # Best-effort notification on sweep completion
    try:
        from lab.core.notify import notify as _notify

        n_exec = summary.get("executed", 0)
        n_err = summary.get("errors", 0)
        priority = "high" if n_err else "default"
        tag = "x" if n_err else "white_check_mark"
        _notify(
            f"{spec.experiment.slug}: {n_exec} runs ({n_err} errors)",
            title="lab sweep done",
            priority=priority,  # type: ignore[arg-type]
            tags=[tag],
        )
    except Exception as exc:
        log.warning("sweep_notify_failed", error=str(exc))

    return summary


# ----------------------------------------------------------------------------
# status / cancel (second-terminal monitoring)
# ----------------------------------------------------------------------------


@dataclass
class SweepStatus:
    """Snapshot of in-flight sweep state for status reporting."""

    in_progress: list[dict[str, Any]]  # one row per active experiment_run
    gpu_lease_holder: str | None
    gpu_lease_ttl: int
    sweep_pids: list[tuple[str, int]]  # (slug, pid)


def get_sweep_status() -> SweepStatus:
    """Snapshot in-flight sweeps from the DB + GPU lease + pidfile dir."""
    rows: list[dict[str, Any]] = []
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.run_id, e.slug, m.litellm_id, r.task_id, r.seed, r.started_at
            FROM experiment_runs r
            JOIN experiments e ON e.experiment_id = r.experiment_id
            JOIN models m ON m.model_id = r.model_id
            WHERE r.status IN ('running', 'in_progress')
            ORDER BY r.started_at DESC
            LIMIT 50
            """
        )
        for row in cur.fetchall():
            rows.append(
                {
                    "run_id": row[0],
                    "experiment_slug": row[1],
                    "model": row[2],
                    "task_id": int(row[3]),
                    "seed": int(row[4]),
                    "started_at": row[5],
                }
            )
    holder, ttl = gpu_lease_status()
    pids: list[tuple[str, int]] = []
    if PIDFILE_DIR.exists():
        for f in sorted(PIDFILE_DIR.glob("*.pid")):
            slug = f.stem
            pid = read_pidfile(slug)
            if pid is not None:
                pids.append((slug, pid))
    return SweepStatus(
        in_progress=rows,
        gpu_lease_holder=holder,
        gpu_lease_ttl=ttl,
        sweep_pids=pids,
    )


def cancel_sweep(slug: str, *, release_lease: bool = True) -> dict[str, Any]:
    """Cancel an in-flight sweep by slug. Signals the runner PID with SIGTERM.

    Returns {"signaled": pid|None, "released_lease": bool}.
    """
    pid = read_pidfile(slug)
    signaled: int | None = None
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
            signaled = pid
        except (ProcessLookupError, PermissionError) as exc:
            log.warning("sweep_cancel_signal_failed", pid=pid, slug=slug, error=str(exc))
    released = False
    if release_lease:
        # Brief grace period for the signal handler; then force-release as a backstop.
        time.sleep(0.5)
        try:
            holder, _ttl = gpu_lease_status()
            if holder:
                released = force_release()
        except Exception as exc:
            log.error("gpu_lease_release_failed", slug=slug, error=str(exc))
    _clear_pidfile(slug)
    return {"signaled": signaled, "released_lease": released}


# ----------------------------------------------------------------------------
# Preflight: refuse to start if LiteLLM proxy is missing keep_alive on locals
# ----------------------------------------------------------------------------


def preflight_litellm_keep_alive_or_raise(
    config_path: Path | None = None,
) -> None:
    """Read the LiteLLM proxy config; raise if any local Ollama model lacks `keep_alive`.

    Surfaced by EXP-001 postmortem: a missing `keep_alive` caused VRAM thrash
    when the proxy unloaded a model between cells of the same sweep.
    """
    from lab.sweep.preflight import check_litellm_keep_alive

    check_litellm_keep_alive(config_path)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a sweep")
    parser.add_argument("config", type=str, help="Path to sweep YAML/JSON")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    from pathlib import Path

    from lab.sweep.config import load_sweep

    spec = load_sweep(Path(args.config))
    from pathlib import Path as _P

    litellm_key_path = _P("/data/lab/services/litellm-master-key")
    litellm_key = litellm_key_path.read_text().strip()

    summary = run_sweep(
        spec, litellm_key=litellm_key, resume=not args.no_resume, dry_run=args.dry_run
    )
    console.print(f"[bold green]summary[/]: {summary}")
    return 0 if summary.get("errors", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
