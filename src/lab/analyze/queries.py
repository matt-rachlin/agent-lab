"""DuckDB-backed analyze queries.

DuckDB connects directly to Postgres via the `postgres_scanner` extension, so
we can write SQL that joins the `experiment_runs` table against any local
Parquet dumps we may have without copying data.
"""

from __future__ import annotations

import duckdb


def open_db() -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection attached to the lab Postgres."""
    con = duckdb.connect(":memory:")
    con.execute("INSTALL postgres; LOAD postgres;")
    con.execute("ATTACH 'dbname=lab' AS lab (TYPE postgres, READ_ONLY);")
    return con


def fetchall_as_dicts(rel: duckdb.DuckDBPyRelation) -> list[dict[str, object]]:
    """Helper: relation → list of dicts."""
    columns = [d[0] for d in rel.description]
    rows: list[dict[str, object]] = [dict(zip(columns, row, strict=True)) for row in rel.fetchall()]
    return rows


def summary_by_model_simple(experiment_slug: str) -> list[dict[str, object]]:
    """Per-model aggregate stats — implemented without relying on extension SQL functions
    that differ between DuckDB versions."""
    con = open_db()
    rel = con.sql(
        f"""
        SELECT
            m.litellm_id                                            AS model,
            m.backend                                               AS backend,
            COUNT(*)                                                AS n,
            SUM(CASE WHEN r.status='done'  THEN 1 ELSE 0 END)       AS done_n,
            SUM(CASE WHEN r.status='error' THEN 1 ELSE 0 END)       AS error_n,
            ROUND(AVG(r.latency_ms)::DOUBLE, 1)                     AS latency_ms_mean,
            ROUND(MEDIAN(r.latency_ms)::DOUBLE, 1)                  AS latency_ms_p50,
            ROUND(QUANTILE_CONT(r.latency_ms, 0.95)::DOUBLE, 1)     AS latency_ms_p95,
            ROUND(AVG(r.tokens_in)::DOUBLE, 1)                      AS tokens_in_mean,
            ROUND(AVG(r.tokens_out)::DOUBLE, 1)                     AS tokens_out_mean
        FROM lab.public.experiment_runs r
        JOIN lab.public.models m       ON m.model_id      = r.model_id
        JOIN lab.public.experiments e  ON e.experiment_id = r.experiment_id
        WHERE e.slug = '{experiment_slug}'
        GROUP BY m.litellm_id, m.backend
        ORDER BY m.litellm_id
        """
    )
    return fetchall_as_dicts(rel)


def summary_by_model_config(experiment_slug: str) -> list[dict[str, object]]:
    """Per-(model, config) aggregate stats."""
    con = open_db()
    rel = con.sql(
        f"""
        SELECT
            m.litellm_id                                            AS model,
            r.config_hash                                           AS config_hash,
            COUNT(*)                                                AS n,
            SUM(CASE WHEN r.status='done'  THEN 1 ELSE 0 END)       AS done_n,
            SUM(CASE WHEN r.status='error' THEN 1 ELSE 0 END)       AS error_n,
            ROUND(AVG(r.latency_ms)::DOUBLE, 1)                     AS latency_ms_mean,
            ROUND(AVG(r.tokens_out)::DOUBLE, 1)                     AS tokens_out_mean
        FROM lab.public.experiment_runs r
        JOIN lab.public.models m       ON m.model_id      = r.model_id
        JOIN lab.public.experiments e  ON e.experiment_id = r.experiment_id
        WHERE e.slug = '{experiment_slug}'
        GROUP BY m.litellm_id, r.config_hash
        ORDER BY m.litellm_id, r.config_hash
        """
    )
    return fetchall_as_dicts(rel)
