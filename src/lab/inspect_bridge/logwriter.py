"""Inspect EvalLog → MinIO trajectory JSONL + Postgres upsert.

The Inspect harness records everything it sees in its own log format
(`EvalLog`); we mirror the essential parts into the lab's own Postgres
tables so the existing analysis / sweep tooling keeps working.

  * MinIO: `runs/YYYY-MM/DD/<run_id>/trajectory.jsonl` — the full per-turn
    trajectory written one line at a time. The same key layout the
    single-turn fast path uses for `trace.jsonl`.
  * Postgres `experiment_runs`: upsert the standard run columns plus the
    three Phase 6 additions (`actual_turns`, `tool_call_count`,
    `sandbox_image_hash`).
  * Postgres `agent_logs`: pointer to the MinIO key plus a compact per-turn
    array (latencies, tool names, budget state — everything you'd want to
    answer "what did the agent do?" without re-fetching MinIO).

Idempotent on `run_id`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Json

from lab.minio_io import run_key, upload_bytes
from lab.settings import get_settings


@dataclass
class SweepContext:
    """The lab-side identity bits the Inspect log doesn't carry.

    `experiment_id` and friends already exist on the lab side; Inspect
    doesn't know about them, so the runner passes them in alongside the
    `EvalLog` when calling `write_run_from_inspect_log`.
    """

    run_id: str
    experiment_id: int | None
    experiment_slug: str
    model_id: int
    model_litellm_id: str
    task_id: int
    task_slug: str
    config_hash: str
    config: dict[str, Any]
    seed: int
    manifest_sha: str


_SANDBOX_HASH_PATH = Path("conf/sandbox-image.sha")


def _read_sandbox_hash() -> str | None:
    """Return the recorded sandbox image hash, or None if absent.

    Written by `lab agent sandbox build`. We don't fail if it's missing —
    some unit tests run without ever building the image.
    """

    if not _SANDBOX_HASH_PATH.exists():
        return None
    try:
        return _SANDBOX_HASH_PATH.read_text().strip() or None
    except OSError:
        return None


def _extract_sample_metadata(log: Any) -> dict[str, Any]:
    """Pull the `lab_agent` trajectory and other useful bits off the EvalLog.

    Inspect stores samples on `log.samples`. We expect exactly one sample
    per cell — if there are zero (e.g. catastrophic failure), we return a
    minimal record so the upsert still happens with `status='error'`.
    """

    samples = getattr(log, "samples", None) or []
    if not samples:
        return {
            "lab_agent": {
                "error": str(getattr(log, "error", "")) or "no samples",
                "turns": [],
                "actual_turns": 0,
                "tool_call_count": 0,
                "terminated_reason": "no_samples",
            },
            "score": None,
            "messages": [],
            "model_usage": {},
            "total_time": None,
        }
    sample = samples[0]
    metadata = dict(sample.metadata or {})
    lab_agent = metadata.get("lab_agent") or {}
    # Score is in `sample.scores`; we surface the noop value for the row.
    scores = sample.scores or {}
    primary = next(iter(scores.values()), None)
    score_value: float | None
    if primary is not None:
        try:
            score_value = float(primary.value)
        except (TypeError, ValueError):
            score_value = None
    else:
        score_value = None
    messages = [_message_to_jsonable(m) for m in (sample.messages or [])]
    model_usage = getattr(sample, "model_usage", None) or {}
    if hasattr(model_usage, "model_dump"):
        model_usage = model_usage.model_dump()
    total_time = getattr(sample, "total_time", None)
    return {
        "lab_agent": lab_agent,
        "score": score_value,
        "messages": messages,
        "model_usage": model_usage,
        "total_time": total_time,
        "error": (str(sample.error) if getattr(sample, "error", None) is not None else None),
    }


def _message_to_jsonable(msg: Any) -> dict[str, Any]:
    """Best-effort message → JSON. Used for the MinIO trajectory record."""

    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump()  # type: ignore[no-any-return]
    return {
        "role": getattr(msg, "role", "user"),
        "content": getattr(msg, "content", ""),
    }


def _trajectory_bytes(*, ctx: SweepContext, extracted: dict[str, Any]) -> bytes:
    """Serialise the trajectory to one JSON object per line.

    Format:
      Line 1: header — run identity + sweep context
      Line 2: messages — the final conversation
      Line 3..N: per-turn entries from `lab_agent.turns`
      Final line: footer — aggregate stats
    """

    lines: list[str] = []
    lab_agent = extracted.get("lab_agent") or {}
    lines.append(
        json.dumps(
            {
                "type": "header",
                "run_id": ctx.run_id,
                "experiment_slug": ctx.experiment_slug,
                "experiment_id": ctx.experiment_id,
                "model": ctx.model_litellm_id,
                "task_slug": ctx.task_slug,
                "config_hash": ctx.config_hash,
                "seed": ctx.seed,
                "manifest_sha": ctx.manifest_sha,
            },
            default=str,
        )
    )
    lines.append(
        json.dumps(
            {"type": "messages", "messages": extracted.get("messages", [])},
            default=str,
        )
    )
    for turn in lab_agent.get("turns") or []:
        lines.append(json.dumps({"type": "turn", **turn}, default=str))
    lines.append(
        json.dumps(
            {
                "type": "footer",
                "actual_turns": lab_agent.get("actual_turns"),
                "tool_call_count": lab_agent.get("tool_call_count"),
                "terminated_reason": lab_agent.get("terminated_reason"),
                "total_latency_ms": lab_agent.get("total_latency_ms"),
                "error": lab_agent.get("error") or extracted.get("error"),
                "model_usage": extracted.get("model_usage", {}),
                "score": extracted.get("score"),
            },
            default=str,
        )
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _compact_turns(lab_agent: dict[str, Any]) -> list[dict[str, Any]]:
    """Strip per-turn entries down to the columns `agent_logs.turns` cares about.

    We omit the verbose previews / args / results — the full record lives
    in MinIO under `trajectory.jsonl`. The compact form here exists to make
    quick analytics queries cheap (no JSONB-deep-walk + de-serialise).
    """

    out: list[dict[str, Any]] = []
    for entry in lab_agent.get("turns") or []:
        compact: dict[str, Any] = {
            "turn": entry.get("turn"),
            "latency_ms": entry.get("latency_ms"),
            "tokens_in": entry.get("tokens_in"),
            "tokens_out": entry.get("tokens_out"),
            "tool_calls_requested": entry.get("tool_calls_requested"),
            "budget_exhausted": entry.get("budget_exhausted", False),
        }
        if entry.get("error"):
            compact["error"] = entry["error"]
        tool_calls = entry.get("tool_calls")
        if tool_calls:
            compact["tools"] = [
                {
                    "tool": tc.get("tool"),
                    "latency_ms": tc.get("latency_ms"),
                    "error": tc.get("error"),
                }
                for tc in tool_calls
            ]
        out.append(compact)
    return out


def _upsert_experiment_run(
    *,
    ctx: SweepContext,
    extracted: dict[str, Any],
    trajectory_key: str,
) -> None:
    """UPSERT into `experiment_runs` with the agent-loop metrics."""

    lab_agent = extracted.get("lab_agent") or {}
    error = lab_agent.get("error") or extracted.get("error")
    status = "error" if error else "done"

    usage = extracted.get("model_usage") or {}
    tokens_in: int | None = None
    tokens_out: int | None = None
    # `model_usage` is keyed by model name; aggregate across models for the
    # cell's totals.
    for v in usage.values():
        if isinstance(v, dict):
            ti = v.get("input_tokens")
            to = v.get("output_tokens")
            if ti is not None:
                tokens_in = (tokens_in or 0) + int(ti)
            if to is not None:
                tokens_out = (tokens_out or 0) + int(to)
    latency_ms = lab_agent.get("total_latency_ms") or 0

    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO experiment_runs
                (run_id, experiment_id, model_id, task_id, config_hash, config, seed,
                 status, manifest_sha, trace_path, tokens_in, tokens_out, latency_ms,
                 cost_usd, error, started_at, completed_at,
                 actual_turns, tool_call_count, sandbox_image_hash)
            VALUES
                (%(run_id)s, %(experiment_id)s, %(model_id)s, %(task_id)s, %(config_hash)s,
                 %(config)s, %(seed)s, %(status)s, %(manifest_sha)s, %(trace_path)s,
                 %(tokens_in)s, %(tokens_out)s, %(latency_ms)s, %(cost_usd)s, %(error)s,
                 NOW(), NOW(),
                 %(actual_turns)s, %(tool_call_count)s, %(sandbox_image_hash)s)
            ON CONFLICT (run_id) DO UPDATE SET
                status = EXCLUDED.status,
                manifest_sha = EXCLUDED.manifest_sha,
                trace_path = EXCLUDED.trace_path,
                tokens_in = EXCLUDED.tokens_in,
                tokens_out = EXCLUDED.tokens_out,
                latency_ms = EXCLUDED.latency_ms,
                cost_usd = EXCLUDED.cost_usd,
                error = EXCLUDED.error,
                completed_at = NOW(),
                actual_turns = EXCLUDED.actual_turns,
                tool_call_count = EXCLUDED.tool_call_count,
                sandbox_image_hash = EXCLUDED.sandbox_image_hash;
            """,
            {
                "run_id": ctx.run_id,
                "experiment_id": ctx.experiment_id,
                "model_id": ctx.model_id,
                "task_id": ctx.task_id,
                "config_hash": ctx.config_hash,
                "config": Json(ctx.config),
                "seed": ctx.seed,
                "status": status,
                "manifest_sha": ctx.manifest_sha,
                "trace_path": trajectory_key,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
                "cost_usd": None,
                "error": error,
                "actual_turns": lab_agent.get("actual_turns"),
                "tool_call_count": lab_agent.get("tool_call_count"),
                "sandbox_image_hash": _read_sandbox_hash(),
            },
        )


def _upsert_agent_log(
    *,
    run_id_: str,
    trajectory_key: str,
    compact_turns: list[dict[str, Any]],
) -> None:
    """UPSERT into `agent_logs`. Idempotent on `run_id`."""

    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO agent_logs (run_id, inspect_log_path, turns, inserted_at)
            VALUES (%(run_id)s, %(path)s, %(turns)s, NOW())
            ON CONFLICT (run_id) DO UPDATE SET
                inspect_log_path = EXCLUDED.inspect_log_path,
                turns = EXCLUDED.turns,
                inserted_at = NOW();
            """,
            {
                "run_id": run_id_,
                "path": trajectory_key,
                "turns": Json(compact_turns),
            },
        )


def write_run_from_inspect_log(log: Any, sweep_context: SweepContext) -> str:
    """Persist one cell's run to MinIO + Postgres.

    Returns the MinIO key the trajectory was uploaded to.
    """

    extracted = _extract_sample_metadata(log)
    data = _trajectory_bytes(ctx=sweep_context, extracted=extracted)
    key = run_key(sweep_context.run_id, "trajectory.jsonl")
    trace_uri = upload_bytes(key=key, data=data, content_type="application/x-ndjson")
    _upsert_experiment_run(ctx=sweep_context, extracted=extracted, trajectory_key=trace_uri)
    _upsert_agent_log(
        run_id_=sweep_context.run_id,
        trajectory_key=trace_uri,
        compact_turns=_compact_turns(extracted.get("lab_agent") or {}),
    )
    return trace_uri


__all__ = ["SweepContext", "write_run_from_inspect_log"]
