"""NS-1 Analyst v0 (charter NS-1, ADR-012) — read tools + LAR wiring.

A thin caller of the Lab Agent Runtime (ADR-012, lab.core.agent_runtime),
mirroring the scout (lab.scout_scan). The analyst is strictly READ-ONLY: it
pulls experiment results out of Postgres through three SELECT-only tools, computes
per-(model, evaluator) pass rates, flags the F-017 non-emission artefact (a cell
whose mean score collapses to ~0 — typically a reasoning model that narrates
instead of emitting a tool call), and drafts a writeup.

The read tools accept an injectable ``cursor_factory`` (default = a real psycopg
cursor over the configured ``pg_dsn``) so tests drive them against synthetic rows
without ever touching the live DB. Every SQL statement is SELECT-only and binds
its slug as a parameter (no string interpolation).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from decimal import Decimal
from typing import Any

from lab.core.agent_runtime import Tool, run_agent
from lab.core.settings import get_settings

#: A cell at or below this mean score (with n>=1) is treated as an F-017 smell.
F017_SCORE_FLOOR: float = 0.02

#: Stable label for the F-017 (non-emission / zero-score collapse) artefact.
F017_SMELL: str = "f017"

CursorFactory = Callable[[], "contextlib.AbstractContextManager[Any]"]


@contextlib.contextmanager
def _default_cursor_factory() -> Iterator[Any]:
    """A read cursor over the configured Postgres DSN (real-DB default)."""
    import psycopg

    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        yield cur


def _rows(cur: Any) -> list[dict[str, Any]]:
    """Materialize the last result set as a list of column->value dicts."""
    cols = [d.name for d in cur.description] if cur.description else []
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _as_float(value: Any) -> Any:
    """Coerce Decimal (psycopg NUMERIC) to float; leave None/others alone."""
    return float(value) if isinstance(value, Decimal) else value


# --------------------------------------------------------------------------- #
# SQL (all SELECT-only; slug is always a bound parameter).                     #
# --------------------------------------------------------------------------- #

_RUNS_SQL = """
    SELECT er.run_id          AS run_id,
           er.status          AS status,
           er.trust_level     AS trust_level,
           m.litellm_id       AS model,
           m.backend          AS backend,
           ev.name            AS evaluator,
           res.score          AS score,
           res.passed         AS passed
    FROM experiments e
    JOIN experiment_runs er ON er.experiment_id = e.experiment_id
    JOIN models m           ON m.model_id = er.model_id
    LEFT JOIN eval_results res ON res.run_id = er.run_id
    LEFT JOIN evaluators ev    ON ev.evaluator_id = res.evaluator_id
    WHERE e.slug = %s
    ORDER BY er.run_id, ev.name
"""

# Named _RATES_SQL (not _PASS_RATE_SQL) to dodge the S105 ruff false-positive on
# the substring "PASS" looking like a hardcoded password.
_RATES_SQL = """
    SELECT m.litellm_id                              AS model,
           ev.name                                   AS evaluator,
           COUNT(res.score)                          AS n,
           AVG(res.score)                             AS mean_score,
           AVG(CASE WHEN res.passed THEN 1.0 ELSE 0.0 END) AS pass_rate
    FROM experiments e
    JOIN experiment_runs er ON er.experiment_id = e.experiment_id
    JOIN models m           ON m.model_id = er.model_id
    JOIN eval_results res   ON res.run_id = er.run_id
    JOIN evaluators ev      ON ev.evaluator_id = res.evaluator_id
    WHERE e.slug = %s
    GROUP BY m.litellm_id, ev.name
    ORDER BY m.litellm_id, ev.name
"""

_METADATA_SQL = """
    SELECT er.run_id          AS run_id,
           er.status          AS status,
           er.error           AS error,
           er.trace_path      AS trace_path,
           er.tool_call_count AS tool_call_count,
           er.actual_turns    AS actual_turns,
           er.tokens_out      AS tokens_out
    FROM experiments e
    JOIN experiment_runs er ON er.experiment_id = e.experiment_id
    WHERE e.slug = %s
    ORDER BY er.run_id
"""


# --------------------------------------------------------------------------- #
# Read-only tools (Tool ABI, side_effect="read").                             #
# --------------------------------------------------------------------------- #


def analyst_query_experiment(
    slug: str, *, cursor_factory: CursorFactory = _default_cursor_factory
) -> dict[str, Any]:
    """Per-run rows for an experiment, joined to per-evaluator scores."""
    with cursor_factory() as cur:
        cur.execute(_RUNS_SQL, (slug,))
        rows = _rows(cur)
    for r in rows:
        if "score" in r:
            r["score"] = _as_float(r["score"])
    return {"slug": slug, "found": bool(rows), "rows": rows}


def analyst_pass_rates(
    slug: str, *, cursor_factory: CursorFactory = _default_cursor_factory
) -> dict[str, Any]:
    """Per-(model, evaluator) mean score + pass rate, plus F-017 smell flags.

    A cell with n>=1 whose mean_score collapses to <= F017_SCORE_FLOOR is tagged
    as an F-017 non-emission artefact (smell="f017")."""
    with cursor_factory() as cur:
        cur.execute(_RATES_SQL, (slug,))
        cells = _rows(cur)
    for c in cells:
        c["mean_score"] = _as_float(c.get("mean_score"))
        c["pass_rate"] = _as_float(c.get("pass_rate"))
    smells: list[dict[str, Any]] = []
    for c in cells:
        n = c.get("n")
        mean = c.get("mean_score")
        if n is not None and n >= 1 and mean is not None and mean <= F017_SCORE_FLOOR:
            smells.append(
                {
                    "smell": F017_SMELL,
                    "model": c.get("model"),
                    "evaluator": c.get("evaluator"),
                    "n": n,
                    "mean_score": mean,
                    "pass_rate": c.get("pass_rate"),
                }
            )
    return {"slug": slug, "cells": cells, "smells": smells}


def analyst_run_metadata(
    slug: str, *, cursor_factory: CursorFactory = _default_cursor_factory
) -> dict[str, Any]:
    """Per-run status/error/trace + tool_call_count/actual_turns/tokens_out.

    Corroborates F-017: high tokens_out with zero tool calls means the model
    produced prose instead of emitting the tool call the task required."""
    with cursor_factory() as cur:
        cur.execute(_METADATA_SQL, (slug,))
        runs = _rows(cur)
    return {"slug": slug, "runs": runs}


_SLUG_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {"slug": {"type": "string"}},
    "required": ["slug"],
}


def build_tools() -> list[Tool]:
    """The analyst's three read-only tools as ADR-012 Tool ABI instances."""
    return [
        Tool(
            name="analyst_query_experiment",
            description=(
                "Read every run of an experiment (by slug) joined to its "
                "per-evaluator scores: run_id, model, backend, evaluator, score, "
                "passed, status, trust_level. Read-only."
            ),
            parameters=_SLUG_PARAMS,
            impl=analyst_query_experiment,
            side_effect="read",
        ),
        Tool(
            name="analyst_pass_rates",
            description=(
                "Aggregate an experiment (by slug) to per-(model, evaluator) mean "
                "score and pass rate, and pre-flag F-017 non-emission artefacts "
                "(cells whose mean score collapses to ~0). Read-only."
            ),
            parameters=_SLUG_PARAMS,
            impl=analyst_pass_rates,
            side_effect="read",
        ),
        Tool(
            name="analyst_run_metadata",
            description=(
                "Read per-run trace metadata for an experiment (by slug): status, "
                "error, trace_path, tool_call_count, actual_turns, tokens_out — to "
                "corroborate F-017 (prose tokens, zero tool calls). Read-only."
            ),
            parameters=_SLUG_PARAMS,
            impl=analyst_run_metadata,
            side_effect="read",
        ),
    ]


# --------------------------------------------------------------------------- #
# LAR caller (mirrors lab.scout_scan.run_scan).                               #
# --------------------------------------------------------------------------- #

_SYSTEM = """You are the lab's results analyst. Investigate ONE experiment.

Workflow: pull the data with the read tools (analyst_query_experiment for the raw
per-run scores, analyst_pass_rates for per-(model, evaluator) aggregates, and
analyst_run_metadata for trace metadata), compute what the numbers say, then FLAG
the F-017 non-emission artefact wherever a cell's mean score collapses to ~0 —
corroborate it with the run metadata (high tokens_out + zero tool_call_count is
the tell of a reasoning model that narrated instead of emitting a tool call).
Finally draft a short writeup of the findings: which models pass, which are
suspect, and an explicit recommendation to quarantine any F-017 cell before it
reaches a leaderboard.

You are read-only; you cannot modify any lab state."""


def _smells_from_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lift F-017 smells out of the analyst_pass_rates tool result(s)."""
    smells: list[dict[str, Any]] = []
    for r in results:
        if r.get("name") != "analyst_pass_rates":
            continue
        res = r.get("result")
        if isinstance(res, dict):
            found = res.get("smells")
            if isinstance(found, list):
                smells.extend(found)
    return smells


def _writeup_from_messages(messages: list[dict[str, Any]]) -> str:
    """The final assistant message is the drafted writeup."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return ""


def analyze_experiment(
    *,
    slug: str,
    model: str = "qwen3-4b-ft-toolcall-q4-latest",
    max_tool_calls: int = 12,
    timeout: int = 90,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Drive the analyst LAR over one experiment and return the surfaced result.

    Leaves ``allow_side_effects`` at the runtime default (read-only); the analyst
    is wired as actor="analyst". Returns the slug, the model, the number of tool
    calls, the stop reason, the lifted F-017 smells, and the drafted writeup."""
    settings = get_settings()
    res = run_agent(
        settings=settings,
        litellm_key=settings.litellm_key,
        model=model,
        system=_SYSTEM,
        user=(
            f"Analyze experiment {slug}. Use the read tools to pull its results, "
            "compute pass rates, flag any F-017 artefact, and draft a writeup."
        ),
        tools=build_tools(),
        actor="analyst",
        max_turns=max_tool_calls,
        max_tool_calls=max_tool_calls,
        timeout=timeout,
        num_ctx=num_ctx,
    )
    return {
        "slug": slug,
        "model": model,
        "tool_calls": res.tool_calls,
        "stop": res.stop_reason,
        "smells": _smells_from_results(res.tool_results),
        "writeup": _writeup_from_messages(res.messages),
    }
