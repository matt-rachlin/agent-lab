"""One-time MLflow backfill: walk existing Postgres rows and mirror them.

Idempotent — rows that already carry a ``mlflow_*_id`` are skipped (unless
``force=True``). Errors on individual rows are logged and the walk
continues; the goal is "best effort, never lose what we can mirror" not
"all-or-nothing transaction".

Usage::

    from lab.observability.mlflow_backfill import backfill_all

    summary = backfill_all(dry_run=False)
    print(summary)

Or via the thin CLI wrapper at ``tools/backfill_mlflow.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import psycopg

from lab.core.settings import get_settings
from lab.observability.mlflow_mirror import MlflowMirror

_CONFIDENCE_FLOAT = {"low": 0.3, "medium": 0.6, "high": 0.9}


@dataclass
class BackfillSummary:
    experiments: int = 0
    experiments_skipped: int = 0
    runs: int = 0
    runs_skipped: int = 0
    findings: int = 0
    findings_skipped: int = 0
    models: int = 0
    models_skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiments": self.experiments,
            "experiments_skipped": self.experiments_skipped,
            "runs": self.runs,
            "runs_skipped": self.runs_skipped,
            "findings": self.findings,
            "findings_skipped": self.findings_skipped,
            "models": self.models,
            "models_skipped": self.models_skipped,
            "errors": self.errors[:20],  # cap for readability
        }


def _log(msg: str) -> None:
    print(f"[mlflow_backfill] {msg}", file=sys.stderr)


def backfill_experiments(
    mirror: MlflowMirror,
    *,
    dry_run: bool,
    force: bool,
    summary: BackfillSummary,
) -> dict[str, str]:
    """Mirror every experiment; return slug → mlflow_experiment_id."""

    out: dict[str, str] = {}
    pg_dsn = get_settings().pg_dsn
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT slug, title, plan_path, hypothesis, mlflow_experiment_id
              FROM experiments
             ORDER BY slug
            """
        )
        rows = cur.fetchall()
    for slug, title, plan_path, hypothesis, mlflow_exp_id in rows:
        if mlflow_exp_id and not force:
            summary.experiments_skipped += 1
            out[slug] = str(mlflow_exp_id)
            continue
        if dry_run:
            summary.experiments += 1
            _log(f"DRY: would upsert_experiment({slug})")
            continue
        try:
            exp_id = mirror.upsert_experiment(
                slug,
                title=title or slug,
                plan_path=plan_path or "",
                hypothesis=hypothesis,
            )
            if exp_id:
                out[slug] = exp_id
                with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE experiments SET mlflow_experiment_id = %s WHERE slug = %s",
                        (exp_id, slug),
                    )
                summary.experiments += 1
            else:
                summary.errors.append(f"experiment {slug}: mirror returned None")
        except Exception as exc:
            summary.errors.append(f"experiment {slug}: {type(exc).__name__}: {exc}")
    return out


def backfill_runs(
    mirror: MlflowMirror,
    *,
    dry_run: bool,
    force: bool,
    summary: BackfillSummary,
    limit: int | None = None,
) -> None:
    """Mirror experiment_runs in batches, oldest-first."""

    pg_dsn = get_settings().pg_dsn
    query = """
        SELECT er.run_id, er.experiment_id, e.slug, m.litellm_id, m.backend,
               t.slug, er.config_hash, er.config, er.seed, er.status,
               er.trace_path, er.tokens_in, er.tokens_out, er.latency_ms,
               er.cost_usd, er.actual_turns, er.tool_call_count,
               er.sandbox_image_hash, er.mlflow_run_id
          FROM experiment_runs er
          JOIN experiments e ON e.experiment_id = er.experiment_id
          JOIN models      m ON m.model_id      = er.model_id
          JOIN tasks       t ON t.task_id       = er.task_id
         ORDER BY er.started_at
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        # `query` was concatenated with an f-string LIMIT clause above so
        # it's no longer `LiteralString` (psycopg's Query type uses that to
        # prevent SQL injection at the type level). `int(limit)` is the
        # only interpolation and is integer-coerced. Pass bytes (also a
        # valid Query) to satisfy pyright.
        cur.execute(query.encode("utf-8"))
        rows = cur.fetchall()

    for row in rows:
        (
            run_id,
            _experiment_id,
            exp_slug,
            litellm_id,
            model_backend,
            task_slug,
            config_hash,
            config,
            seed,
            status,
            trace_path,
            tokens_in,
            tokens_out,
            latency_ms,
            cost_usd,
            actual_turns,
            tool_call_count,
            sandbox_hash,
            mlflow_run_id,
        ) = row
        if mlflow_run_id and not force:
            summary.runs_skipped += 1
            continue
        if dry_run:
            summary.runs += 1
            continue
        try:
            metrics: dict[str, float] = {}
            for name, val in (
                ("latency_ms", latency_ms),
                ("tokens_in", tokens_in),
                ("tokens_out", tokens_out),
                ("cost_usd", cost_usd),
                ("actual_turns", actual_turns),
                ("tool_call_count", tool_call_count),
            ):
                if val is None:
                    continue
                try:
                    metrics[name] = float(val)
                except (TypeError, ValueError):
                    continue

            tags: dict[str, str] = {
                "model_backend": model_backend or "unknown",
                "config_hash": config_hash or "",
            }
            if sandbox_hash:
                tags["sandbox_image_hash"] = sandbox_hash

            mlflow_status = "FAILED" if status == "error" else "FINISHED"
            mlflow_uuid = mirror.log_run(
                exp_slug,
                run_id,
                model=litellm_id or "unknown",
                task=task_slug or "unknown",
                seed=int(seed) if seed is not None else 0,
                config=dict(config) if isinstance(config, dict) else {},
                metrics=metrics,
                tags=tags,
                artifact_uri=trace_path,
                status="FAILED" if mlflow_status == "FAILED" else "FINISHED",
            )
            if mlflow_uuid:
                with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE experiment_runs SET mlflow_run_id = %s WHERE run_id = %s",
                        (mlflow_uuid, run_id),
                    )
                summary.runs += 1
            else:
                summary.errors.append(f"run {run_id}: mirror returned None")
        except Exception as exc:
            summary.errors.append(f"run {run_id}: {type(exc).__name__}: {exc}")


def backfill_findings(
    mirror: MlflowMirror,
    *,
    dry_run: bool,
    force: bool,
    summary: BackfillSummary,
) -> None:
    """Mirror every finding."""

    pg_dsn = get_settings().pg_dsn
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.slug, f.claim, f.confidence, e.slug, f.mlflow_run_id
              FROM findings f
              LEFT JOIN experiments e ON e.experiment_id = f.source_exp
             ORDER BY f.slug
            """
        )
        rows = cur.fetchall()
    for slug, claim, confidence, source_exp_slug, mlflow_run_id in rows:
        if mlflow_run_id and not force:
            summary.findings_skipped += 1
            continue
        if dry_run:
            summary.findings += 1
            continue
        try:
            conf_val = _CONFIDENCE_FLOAT.get((confidence or "").lower(), 0.3)
            evidence = [source_exp_slug] if source_exp_slug else None
            mlflow_uuid = mirror.log_finding(
                slug,
                claim=claim or slug,
                importance=3,
                confidence=conf_val,
                evidence=evidence,
            )
            if mlflow_uuid:
                with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE findings SET mlflow_run_id = %s WHERE slug = %s",
                        (mlflow_uuid, slug),
                    )
                summary.findings += 1
        except Exception as exc:
            summary.errors.append(f"finding {slug}: {type(exc).__name__}: {exc}")


def backfill_models(
    mirror: MlflowMirror,
    *,
    dry_run: bool,
    force: bool,
    summary: BackfillSummary,
) -> None:
    """Mirror every model registry row."""

    pg_dsn = get_settings().pg_dsn
    with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT litellm_id, publisher, variant, capabilities, mlflow_model_uri
              FROM models
             ORDER BY model_id
            """
        )
        rows = cur.fetchall()
    for litellm_id, publisher, variant, capabilities, mlflow_uri in rows:
        if mlflow_uri and not force:
            summary.models_skipped += 1
            continue
        if dry_run:
            summary.models += 1
            continue
        try:
            caps = list(capabilities) if capabilities else []
            uri = mirror.log_model_card(
                litellm_id,
                publisher=publisher or "",
                variant=variant,
                capabilities=caps,
                known_issues=None,
            )
            if uri:
                with psycopg.connect(pg_dsn) as conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE models SET mlflow_model_uri = %s WHERE litellm_id = %s",
                        (uri, litellm_id),
                    )
                summary.models += 1
        except Exception as exc:
            summary.errors.append(f"model {litellm_id}: {type(exc).__name__}: {exc}")


def backfill_all(
    *,
    dry_run: bool = False,
    force: bool = False,
    mirror: MlflowMirror | None = None,
    runs_limit: int | None = None,
) -> BackfillSummary:
    """Walk all four tables and mirror anything not already mirrored."""

    summary = BackfillSummary()
    if mirror is None:
        mirror = MlflowMirror()
    if not mirror.enabled and not dry_run:
        _log("MLflow mirror disabled; nothing to do (set LAB_MLFLOW_URL and re-run)")
        return summary
    backfill_experiments(mirror, dry_run=dry_run, force=force, summary=summary)
    backfill_runs(mirror, dry_run=dry_run, force=force, summary=summary, limit=runs_limit)
    backfill_findings(mirror, dry_run=dry_run, force=force, summary=summary)
    backfill_models(mirror, dry_run=dry_run, force=force, summary=summary)
    return summary


__all__ = [
    "BackfillSummary",
    "backfill_all",
    "backfill_experiments",
    "backfill_findings",
    "backfill_models",
    "backfill_runs",
]
