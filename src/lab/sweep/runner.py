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

from lab.gpu_lease import force_release, gpu_lease
from lab.gpu_lease import status as gpu_lease_status
from lab.manifest import capture as capture_manifest
from lab.settings import get_settings
from lab.sweep.config import RunConfig, SweepConfig, config_hash, run_id
from lab.tasks.registry import get_tasks

console = Console()


# ----------------------------------------------------------------------------
# PID-file convention for inter-process status/cancel
# ----------------------------------------------------------------------------

PIDFILE_DIR = Path("/data/lab/services/sweep-pids")


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
        console.log(f"[yellow]could not write sweep pidfile: {exc}")
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
        console.log(f"[yellow]received {sig_name}; releasing GPU lease and exiting")
        try:
            holder, _ttl = gpu_lease_status()
            if holder:
                force_release()
        except Exception as exc:
            console.log(f"[red]lease release failed: {exc}")
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
        2. sweep `model_defaults[<litellm_id>].system_prompt`
        3. RunConfig.extra `system_prompt`
    """
    messages: list[dict[str, str]] = []
    system = task_payload.get("system") or model_default_system or config_system
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
    from lab.llm import call_litellm_chat

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
    from lab.minio_io import run_key, upload_bytes

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

    if _is_agent_cell(cell.task_payload):
        return _execute_agent_cell(
            cell=cell,
            manifest_sha=manifest.sha,
            timeout=timeout,
            model_default_extra=model_default_extra,
        )

    return _execute_single_turn(
        cell=cell,
        litellm_key=litellm_key,
        timeout=timeout,
        manifest_sha=manifest.sha,
        model_default_system=model_default_system,
        model_default_extra=model_default_extra,
    )


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
            with gpu_lease(
                f"sweep:{cell.experiment_slug}:{cell.model_litellm_id}", ttl_sec=timeout + 60
            ):
                resp_json, latency_ms = _call_litellm(
                    settings=settings,
                    litellm_key=litellm_key,
                    model=cell.model_litellm_id,
                    messages=messages,
                    config=cell_config_for_call,
                    timeout=timeout,
                )
        else:
            resp_json, latency_ms = _call_litellm(
                settings=settings,
                litellm_key=litellm_key,
                model=cell.model_litellm_id,
                messages=messages,
                config=cell.config,
                timeout=timeout,
            )

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
            console.print(f"[yellow]trace upload failed for {cell.run_id}: {exc}")

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
    lab_task = LabTask.model_validate(
        {
            "suite": payload.get("suite", "agent"),
            "slug": cell.task_slug,
            "category": payload.get("category"),
            "input": payload["input"],
            "system": payload.get("system"),
            "tools": payload.get("tools"),
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

    from lab.agent.tools import task_needs_kb_mount as _task_needs_kb_mount

    kb_root_mount: Path | None = None
    if _task_needs_kb_mount(lab_task.tools):
        from lab.settings import get_settings as _get_settings_kb

        kb_root_mount = _get_settings_kb().kb_root
        env.setdefault("LAB_KB_ROOT", "/kb")

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
                    with gpu_lease(
                        f"sweep:{cell.experiment_slug}:{cell.model_litellm_id}",
                        ttl_sec=timeout + 60,
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
                    logs = inspect_eval(
                        inspect_task,
                        display="none",
                        log_samples=True,
                        log_dir=log_dir,
                        log_format="json",
                        log_realtime=False,
                    )
                log = logs[0] if logs else None
                if log is None:
                    raise RuntimeError("inspect_ai.eval returned no logs")
                trace_uri = write_run_from_inspect_log(log, sweep_ctx)
                # Read back the aggregated metrics we just upserted so the
                # in-memory CellResult matches what's in the DB.
                samples = getattr(log, "samples", None) or []
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

    summary = {"total": len(cells), "done_before": len(done), "executed": 0, "errors": 0}
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
        from lab.notify import notify as _notify

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
        console.log(f"[yellow]notify failed: {exc}")

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
            console.log(f"[yellow]could not signal pid {pid}: {exc}")
    released = False
    if release_lease:
        # Brief grace period for the signal handler; then force-release as a backstop.
        time.sleep(0.5)
        try:
            holder, _ttl = gpu_lease_status()
            if holder:
                released = force_release()
        except Exception as exc:
            console.log(f"[yellow]lease release failed: {exc}")
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
