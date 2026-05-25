"""Integration: DuckDB postgres_scanner reads `lab.experiment_runs`.

This is the regression for the Phase 1 connection-string bug — postgres_scanner
needs a libpq-flavored conninfo, not the `postgresql://m@/lab` URL form.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.integration


def test_duckdb_can_scan_experiment_runs(pg: Any, settings: Any) -> None:
    import duckdb

    con = duckdb.connect()
    try:
        con.execute("INSTALL postgres_scanner;")
    except Exception as exc:
        pytest.skip(f"postgres_scanner extension unavailable: {exc}")
    con.execute("LOAD postgres_scanner;")
    # postgres_scanner accepts the same DSN we use in psycopg.
    con.execute(f"ATTACH '{settings.pg_dsn}' AS pg_lab (TYPE POSTGRES, READ_ONLY);")
    res = con.execute(
        "SELECT COUNT(*) FROM pg_lab.public.experiment_runs WHERE status IS NOT NULL"
    ).fetchone()
    assert res is not None
    assert int(res[0]) >= 0
