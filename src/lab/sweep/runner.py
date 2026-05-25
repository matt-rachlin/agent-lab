"""SweepRunner: execute a comparison sweep over the (model, config, task, seed) matrix."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import httpx
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

from lab.gpu_lease import gpu_lease
from lab.manifest import capture as capture_manifest
from lab.settings import get_settings
from lab.sweep.config import RunConfig, SweepConfig, config_hash, run_id
from lab.tasks.registry import get_tasks

console = Console()


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


def _build_messages(task_payload: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = task_payload.get("system")
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
    """Hit the LiteLLM proxy; returns (response_json, latency_ms)."""
    url = settings.litellm_url.rstrip("/") + "/v1/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if config.max_tokens is not None:
        body["max_tokens"] = config.max_tokens
    headers = {"Authorization": f"Bearer {litellm_key}", "Content-Type": "application/json"}
    t0 = time.monotonic()
    resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
    latency_ms = int((time.monotonic() - t0) * 1000)
    resp.raise_for_status()
    return resp.json(), latency_ms


def _persist_trace(*, run_id_: str, payload: dict[str, Any]) -> str:
    """Upload trace JSONL to MinIO. Returns the s3:// path."""
    settings = get_settings()
    from minio import Minio

    client = Minio(
        settings.s3_endpoint.removeprefix("http://").removeprefix("https://"),
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        secure=settings.s3_endpoint.startswith("https://"),
    )
    ts = datetime.now(UTC)
    key = f"runs/{ts:%Y-%m/%d}/{run_id_}/trace.jsonl"
    data = (json.dumps(payload) + "\n").encode()
    client.put_object(
        settings.s3_bucket,
        key,
        BytesIO(data),
        length=len(data),
        content_type="application/x-ndjson",
    )
    return f"s3://{settings.s3_bucket}/{key}"


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


def execute_cell(cell: Cell, *, litellm_key: str, timeout: int) -> CellResult:
    """Execute one matrix cell: capture manifest, call model, persist trace + row."""
    settings = get_settings()
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

    messages = _build_messages(cell.task_payload)
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
                    config=cell.config,
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
                    "manifest_sha": manifest.sha,
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

    _insert_run(cell=cell, result=result, manifest_sha=manifest.sha, trace_path=trace_path)
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

    # Group by model to minimize swap cost (outer = model)
    todo_sorted = sorted(
        todo, key=lambda c: (c.model_litellm_id, c.config.name, c.task_slug, c.seed)
    )

    summary = {"total": len(cells), "done_before": len(done), "executed": 0, "errors": 0}
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
            result = execute_cell(cell, litellm_key=litellm_key, timeout=spec.request_timeout_sec)
            summary["executed"] += 1
            if result.status == "error":
                summary["errors"] += 1
                progress.console.log(
                    f"[red]ERROR[/] {cell.model_litellm_id} {cell.task_slug} seed={cell.seed}: {result.error}"
                )
            progress.update(bar, advance=1)

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
    except Exception as exc:  # noqa: BLE001
        console.log(f"[yellow]notify failed: {exc}")

    return summary


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
