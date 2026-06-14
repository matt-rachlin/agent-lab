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


def summary_eval_by_model(experiment_slug: str) -> list[dict[str, object]]:
    """Per-(model, evaluator) pass-rate summary."""
    con = open_db()
    rel = con.sql(
        f"""
        SELECT
            m.litellm_id                                            AS model,
            ev.name                                                 AS evaluator,
            ev.version                                              AS eval_version,
            COUNT(*)                                                AS n,
            SUM(CASE WHEN er.passed THEN 1 ELSE 0 END)              AS passed,
            ROUND(100.0 * AVG(CASE WHEN er.passed THEN 1.0 ELSE 0.0 END), 1) AS pass_rate_pct,
            ROUND(AVG(er.score)::DOUBLE, 3)                         AS score_mean,
            ROUND(MEDIAN(er.score)::DOUBLE, 3)                      AS score_p50
        FROM lab.public.eval_results er
        JOIN lab.public.experiment_runs r ON r.run_id = er.run_id
        JOIN lab.public.evaluators ev     ON ev.evaluator_id = er.evaluator_id
        JOIN lab.public.models m          ON m.model_id = r.model_id
        JOIN lab.public.experiments e     ON e.experiment_id = r.experiment_id
        WHERE e.slug = '{experiment_slug}'
        GROUP BY m.litellm_id, ev.name, ev.version
        ORDER BY ev.name, m.litellm_id
        """
    )
    return fetchall_as_dicts(rel)


def per_cell_results(experiment_slug: str, evaluator_name: str) -> list[dict[str, object]]:
    """One row per (model, task, seed) for a single evaluator — feeds pass^k and CIs."""
    con = open_db()
    rel = con.sql(
        f"""
        SELECT
            m.litellm_id   AS model,
            t.slug         AS task,
            r.seed         AS seed,
            er.score       AS score,
            er.passed      AS passed
        FROM lab.public.eval_results er
        JOIN lab.public.experiment_runs r ON r.run_id = er.run_id
        JOIN lab.public.evaluators ev     ON ev.evaluator_id = er.evaluator_id
        JOIN lab.public.models m          ON m.model_id = r.model_id
        JOIN lab.public.tasks t           ON t.task_id  = r.task_id
        JOIN lab.public.experiments e     ON e.experiment_id = r.experiment_id
        WHERE e.slug = '{experiment_slug}'
          AND ev.name = '{evaluator_name}'
        """
    )
    return fetchall_as_dicts(rel)


def summary_bfcl_emission(experiment_slug: str) -> list[dict[str, object]]:
    """BFCL function-calling decomposition: emission rate vs accuracy-given-emission.

    BFCL measures *function-calling*, so a model that answers in prose (emits no
    tool call) is a NON-EMISSION, not a wrong answer. Conflating the two scores a
    capable model 0 for a decode/format failure (acute for reasoning models under
    a permissive tool_choice). We split them:
      * emit_rate_pct      - fraction of runs that emitted any tool call
      * acc_given_emit_pct - pass rate among runs that DID emit a call

    ``model_output:no_tool_call`` is the one error_type reliable in historical
    data (the success-path 'unclear' label was cosmetic). Returns [] when the
    experiment has no bfcl_ast_match results.
    """
    con = open_db()
    rel = con.sql(
        f"""
        SELECT
            m.litellm_id                                            AS model,
            COUNT(*)                                                AS n,
            ROUND(100.0 * SUM(CASE WHEN json_extract_string(CAST(er.raw AS JSON),
                  '$.bfcl.error_type') IS DISTINCT FROM 'model_output:no_tool_call'
                  THEN 1 ELSE 0 END) / COUNT(*), 1)                 AS emit_rate_pct,
            ROUND(100.0 * AVG(CASE WHEN er.passed THEN 1.0 ELSE 0.0 END), 1) AS pass_rate_pct,
            ROUND(100.0 * SUM(CASE WHEN er.passed THEN 1.0 ELSE 0.0 END)
                  / NULLIF(SUM(CASE WHEN json_extract_string(CAST(er.raw AS JSON),
                  '$.bfcl.error_type') IS DISTINCT FROM 'model_output:no_tool_call'
                  THEN 1 ELSE 0 END), 0), 1)                        AS acc_given_emit_pct
        FROM lab.public.eval_results er
        JOIN lab.public.experiment_runs r ON r.run_id = er.run_id
        JOIN lab.public.evaluators ev     ON ev.evaluator_id = er.evaluator_id
        JOIN lab.public.models m          ON m.model_id = r.model_id
        JOIN lab.public.experiments e     ON e.experiment_id = r.experiment_id
        WHERE e.slug = '{experiment_slug}' AND ev.name = 'bfcl_ast_match'
        GROUP BY m.litellm_id
        ORDER BY acc_given_emit_pct DESC NULLS LAST
        """
    )
    return fetchall_as_dicts(rel)
