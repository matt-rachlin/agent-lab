"""Cost backfill from the LiteLLM proxy spend ledger.

LiteLLM persists per-request spend to its own Postgres DB (table:
`LiteLLM_SpendLogs`). We don't have time-of-request correlation in our
experiment_runs rows beyond timestamp + model, so we join on (model,
started_at within ±5s).

This is a Phase 4 best-effort: cost numbers improve as LiteLLM's pricing
catalog gets richer. Until then, treat backfilled cost_usd as a lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from lab.core.settings import get_settings


@dataclass(frozen=True)
class BackfillReport:
    runs_examined: int
    spends_found: int
    runs_updated: int
    total_cost_usd: float


_SQL = """
WITH candidates AS (
    SELECT r.run_id,
           m.litellm_id,
           r.started_at,
           r.cost_usd
    FROM experiment_runs r
    JOIN models m ON m.model_id = r.model_id
    WHERE r.status = 'done'
      AND r.cost_usd IS NULL
)
SELECT * FROM candidates;
"""

_SPENDS_SQL = """
SELECT model, spend, "startTime", "endTime"
FROM "LiteLLM_SpendLogs"
WHERE "startTime" BETWEEN %s AND %s
ORDER BY "startTime"
"""


def backfill(limit: int = 1000) -> BackfillReport:
    """Pull spend rows from the litellm DB and match them to experiment_runs."""
    lab_dsn = get_settings().pg_dsn
    # LiteLLM is a separate DB on the same host
    litellm_dsn = lab_dsn.replace("/lab", "/litellm")

    with psycopg.connect(lab_dsn) as lab_conn, lab_conn.cursor() as lab_cur:
        lab_cur.execute(_SQL)
        rows = lab_cur.fetchall()[:limit]

    if not rows:
        return BackfillReport(0, 0, 0, 0.0)

    # Window covering all candidate runs
    from datetime import timedelta

    times = [r[2] for r in rows]
    t_min = min(times) - timedelta(seconds=10)
    t_max = max(times) + timedelta(seconds=600)

    try:
        with psycopg.connect(litellm_dsn) as ll_conn, ll_conn.cursor() as ll_cur:
            ll_cur.execute(_SPENDS_SQL, (t_min, t_max))
            spends = ll_cur.fetchall()
    except psycopg.errors.UndefinedTable:
        # LiteLLM hasn't created the spend table yet (e.g. fresh install)
        return BackfillReport(len(rows), 0, 0, 0.0)

    # Naive index: list of (model, startTime, spend)
    spend_idx: list[tuple[str, object, float]] = [
        (str(s[0]), s[2], float(s[1] or 0.0)) for s in spends
    ]

    updated = 0
    total_cost = 0.0
    with psycopg.connect(lab_dsn) as lab_conn, lab_conn.cursor() as lab_cur:
        for run_id, model, started, _ in rows:
            # Match: same model substring AND startTime within 30s window

            matches = [
                cost
                for sm, st, cost in spend_idx
                if model.split("-")[0] in sm and abs((st - started).total_seconds()) < 30
            ]
            if not matches:
                continue
            cost = max(matches)
            lab_cur.execute(
                "UPDATE experiment_runs SET cost_usd = %s WHERE run_id = %s",
                (cost, run_id),
            )
            updated += 1
            total_cost += cost

    return BackfillReport(
        runs_examined=len(rows),
        spends_found=len(spend_idx),
        runs_updated=updated,
        total_cost_usd=round(total_cost, 6),
    )
