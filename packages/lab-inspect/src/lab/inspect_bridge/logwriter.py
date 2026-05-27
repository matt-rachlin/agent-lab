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

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Json

from lab.core.minio_io import run_key, upload_bytes
from lab.core.settings import get_settings


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
    # Score is in `sample.scores`; pick the "primary" via 6e preference
    # ordering. The per-scorer breakdown is stored on agent_logs so the
    # granularity isn't lost; this is just the single value the runs
    # table carries for fast filtering.
    scores = sample.scores or {}
    primary = _select_primary_score(scores)
    score_value: float | None
    score_breakdown: dict[str, Any] = _build_score_breakdown(scores)
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
        "score_breakdown": score_breakdown,
        "messages": messages,
        "model_usage": model_usage,
        "total_time": total_time,
        "error": (str(sample.error) if getattr(sample, "error", None) is not None else None),
    }


# Preferred order for picking the cell's "primary" score. The Inspect
# scorer registration names (see `lab.inspect_bridge.scorer` and
# `lab.inspect_bridge.scorers.rag`) live here. On RAG tasks `recall_at_k`
# is the headline metric, so it outranks `tool_correctness`; `mrr` /
# `ndcg` / `attribution` ride along but the primary score that drives
# pass/fail dashboards is recall@k.
_PRIMARY_PREFERENCE = (
    "end_state",
    "recall_at_k",
    "tool_correctness",
    "faithfulness",
    "mrr",
    "ndcg",
    "attribution",
    "trajectory_judge",
    "budget_respected",
)


def _select_primary_score(scores: dict[str, Any]) -> Any | None:
    """Return the preferred scorer's `Score` object (or None).

    `scores` is `dict[scorer_name, Score]` as recorded by Inspect on the
    `EvalSample`. We walk the preference order and return the first one
    that exists and isn't NOANSWER (so a tool_correctness=NOANSWER on a
    non-tool-call task doesn't outrank a real budget_respected score).
    Falls back to the first value if none of the preferred ones are
    meaningful.
    """

    # Inspect scorer values use the literal "N" string for NOANSWER.
    def _is_meaningful(score: Any) -> bool:
        if score is None:
            return False
        val = getattr(score, "value", None)
        if val is None:
            return False
        return not (isinstance(val, str) and val == "N")

    for name in _PRIMARY_PREFERENCE:
        for key, score in scores.items():
            if key == name and _is_meaningful(score):
                return score
    # Fall back to the first non-NOANSWER, else the first value.
    for score in scores.values():
        if _is_meaningful(score):
            return score
    return next(iter(scores.values()), None)


def _build_score_breakdown(scores: dict[str, Any]) -> dict[str, Any]:
    """Compact per-scorer breakdown for the agent_logs row.

    Each entry is `{value, explanation}` — enough for the analysis tier
    to surface "task X passed end_state but failed budget_respected"
    without parsing the full Inspect log.
    """

    breakdown: dict[str, Any] = {}
    for key, score in scores.items():
        if score is None:
            continue
        try:
            value: Any = score.value
        except AttributeError:
            value = None
        explanation = getattr(score, "explanation", None)
        # Normalise NOANSWER sentinel for JSON readability.
        if isinstance(value, str) and value == "N":
            normalised: Any = None
        else:
            try:
                normalised = float(value) if value is not None else None
            except (TypeError, ValueError):
                normalised = value
        breakdown[key] = {"value": normalised, "explanation": explanation}
    return breakdown


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
    score_breakdown: dict[str, Any] | None = None,
) -> None:
    """UPSERT into `agent_logs`. Idempotent on `run_id`.

    `score_breakdown` is folded into the `turns` JSONB under a sentinel
    key (`_score_breakdown`) — there's no separate column yet (6e
    deliberately doesn't run a migration for this; 6f can promote it to
    a column if EXP-002 wants to filter on it). The sentinel key cannot
    collide with a turn since real turns are appended as `{turn, ...}`
    objects, not strings.
    """

    payload: list[dict[str, Any]] | dict[str, Any] = compact_turns
    if score_breakdown:
        # Wrap as `{turns, score_breakdown}` so analytics can pick out
        # either piece without parsing both.
        payload = {
            "turns": compact_turns,
            "score_breakdown": score_breakdown,
        }
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
                "turns": Json(payload),
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
        score_breakdown=extracted.get("score_breakdown"),
    )
    # Phase 15.2: additive MLflow mirror. Best-effort, never blocks the
    # canonical Postgres / MinIO writes above.
    _mirror_agent_run_to_mlflow(ctx=sweep_context, extracted=extracted, trace_uri=trace_uri)
    return trace_uri


def _mirror_agent_run_to_mlflow(
    *, ctx: SweepContext, extracted: dict[str, Any], trace_uri: str
) -> None:
    """Mirror an agent-path cell into MLflow + write back mlflow_run_id."""

    try:
        from lab.observability.mlflow_mirror import MlflowMirror

        lab_agent = extracted.get("lab_agent") or {}
        error = lab_agent.get("error") or extracted.get("error")
        status = "FAILED" if error else "FINISHED"

        usage = extracted.get("model_usage") or {}
        tokens_in: int | None = None
        tokens_out: int | None = None
        for v in usage.values():
            if isinstance(v, dict):
                ti = v.get("input_tokens")
                to = v.get("output_tokens")
                if ti is not None:
                    tokens_in = (tokens_in or 0) + int(ti)
                if to is not None:
                    tokens_out = (tokens_out or 0) + int(to)
        metrics: dict[str, float] = {}
        latency = lab_agent.get("total_latency_ms")
        if latency is not None:
            metrics["latency_ms"] = float(latency)
        if tokens_in is not None:
            metrics["tokens_in"] = float(tokens_in)
        if tokens_out is not None:
            metrics["tokens_out"] = float(tokens_out)
        if lab_agent.get("actual_turns") is not None:
            metrics["actual_turns"] = float(lab_agent["actual_turns"])
        if lab_agent.get("tool_call_count") is not None:
            metrics["tool_call_count"] = float(lab_agent["tool_call_count"])
        if extracted.get("score") is not None:
            with contextlib.suppress(TypeError, ValueError):
                metrics["score"] = float(extracted["score"])
        for scorer, info in (extracted.get("score_breakdown") or {}).items():
            val = info.get("value") if isinstance(info, dict) else None
            if val is None:
                continue
            with contextlib.suppress(TypeError, ValueError):
                metrics[f"score.{scorer}"] = float(val)

        sandbox_hash = _read_sandbox_hash()
        tags: dict[str, str] = {
            "model_backend": "agent",
            "config_hash": ctx.config_hash,
        }
        if sandbox_hash:
            tags["sandbox_image_hash"] = sandbox_hash
        if lab_agent.get("terminated_reason"):
            tags["terminated_reason"] = str(lab_agent["terminated_reason"])

        mlflow_run_id = MlflowMirror().log_run(
            ctx.experiment_slug,
            ctx.run_id,
            model=ctx.model_litellm_id,
            task=ctx.task_slug,
            seed=ctx.seed,
            config=ctx.config,
            metrics=metrics,
            tags=tags,
            artifact_uri=trace_uri,
            status="FAILED" if status == "FAILED" else "FINISHED",
        )
        if mlflow_run_id:
            with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE experiment_runs SET mlflow_run_id = %s WHERE run_id = %s",
                    (mlflow_run_id, ctx.run_id),
                )
    except Exception:  # noqa: S110 — belt-and-suspenders; mirror already logs
        pass


__all__ = ["SweepContext", "write_run_from_inspect_log"]
