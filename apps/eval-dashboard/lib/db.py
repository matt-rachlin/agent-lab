"""Postgres + DuckDB access for the eval dashboard.

The dashboard reads the lab's Postgres directly via psycopg, plus uses a
DuckDB cache (postgres_scanner) for snappy reloads on heavier aggregates.

Connection params come from env (LAB_PG_DSN). Never import from `lab.*` —
this app is intentionally decoupled so it doesn't break on sibling
refactors that split src/lab/ into workspace packages.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg

DEFAULT_DSN = "postgresql://m@/lab"
DUCKDB_CACHE = Path.home() / ".cache" / "lab" / "dash.duckdb"


def pg_dsn() -> str:
    """Return the Postgres DSN for the lab DB."""
    return os.environ.get("LAB_PG_DSN", DEFAULT_DSN)


@contextmanager
def pg_conn() -> Iterable[psycopg.Connection]:
    """Yield a psycopg connection. Caller is responsible for the with-block."""
    with psycopg.connect(pg_dsn(), application_name="eval-dashboard") as conn:
        yield conn


def pg_query(sql: str, params: Mapping[str, Any] | tuple | None = None) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame. Empty frame on failure."""
    try:
        with pg_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description is None:
                return pd.DataFrame()
            cols = [c.name for c in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=cols)
    except psycopg.Error as e:
        # Surface the error in the frame so the page can render it.
        return pd.DataFrame({"_error": [str(e)]})


def pg_healthy() -> bool:
    """Cheap health check: SELECT 1."""
    try:
        with pg_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone() == (1,)
    except Exception:
        return False


def ensure_duckdb_cache() -> Path:
    """Create the duckdb cache dir if missing and return its path."""
    DUCKDB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    return DUCKDB_CACHE


def aggregate_stats() -> dict[str, int | float]:
    """Cumulative stats for the Home page."""
    stats: dict[str, int | float] = {
        "experiments": 0,
        "runs": 0,
        "findings": 0,
        "spend_7d_usd": 0.0,
    }
    try:
        with pg_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM experiments")
            stats["experiments"] = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM experiment_runs")
            stats["runs"] = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM findings")
            stats["findings"] = int(cur.fetchone()[0])
            cur.execute(
                "SELECT COALESCE(SUM(cost_usd), 0)::float FROM experiment_runs "
                "WHERE started_at >= NOW() - INTERVAL '7 days'"
            )
            stats["spend_7d_usd"] = float(cur.fetchone()[0])
    except Exception:  # noqa: S110 — render-on-failure is the design
        pass
    return stats
