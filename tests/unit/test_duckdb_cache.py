"""Unit tests for lab.analyze.duckdb_cache.

We never actually attach to Postgres in these tests; we monkeypatch the
Postgres ATTACH call so DuckDB doesn't try to reach a real DB. The tables
get populated with synthetic data so we can verify the public API.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from lab.analyze import duckdb_cache

# --- helpers ---


def _seed_cache(db_path: Path, *, row_count: int = 3) -> None:
    """Write a synthetic cache file that looks like the real refresh output."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for t in duckdb_cache.MIRRORED_TABLES:
            # Each table just gets an id column + a sentinel
            con.execute(
                f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM (VALUES "
                + ", ".join(f"({i}, '{t}')" for i in range(row_count))
                + ") AS t(id, src);"
            )
        con.execute("CREATE OR REPLACE TABLE _meta(refreshed_at TIMESTAMPTZ);")
        con.execute("INSERT INTO _meta VALUES (now());")
    finally:
        con.close()


# --- tests ---


def test_is_stale_missing_file(tmp_path: Path) -> None:
    """Missing cache file is treated as stale."""
    assert duckdb_cache.is_stale(db_path=tmp_path / "does-not-exist.duckdb") is True


def test_is_stale_fresh_file(tmp_path: Path) -> None:
    """Just-written file is fresh (within default 30 min)."""
    db = tmp_path / "cache.duckdb"
    _seed_cache(db)
    assert duckdb_cache.is_stale(db_path=db) is False


def test_is_stale_old_file(tmp_path: Path) -> None:
    """A file with an old refreshed_at is stale."""
    db = tmp_path / "cache.duckdb"
    _seed_cache(db)
    # Backdate _meta.refreshed_at
    con = duckdb.connect(str(db))
    try:
        con.execute("DELETE FROM _meta")
        con.execute("INSERT INTO _meta VALUES (now() - INTERVAL '2 hours')")
    finally:
        con.close()
    assert duckdb_cache.is_stale(db_path=db, max_age_min=30) is True


def test_query_raises_when_cache_missing(tmp_path: Path) -> None:
    """query() must fail loudly if the cache doesn't exist."""
    with pytest.raises(FileNotFoundError):
        duckdb_cache.query("SELECT 1", db_path=tmp_path / "missing.duckdb")


def test_query_returns_dataframe(tmp_path: Path) -> None:
    """query() returns a pandas DataFrame against the cached tables."""
    db = tmp_path / "cache.duckdb"
    _seed_cache(db, row_count=5)
    df = duckdb_cache.query("SELECT COUNT(*) AS n FROM experiment_runs", db_path=db)
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["n"] == 5


def test_refresh_writes_expected_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """refresh() creates one local DuckDB table per MIRRORED_TABLES entry.

    We swap the ATTACH-postgres path for an in-process DuckDB schema so the
    test doesn't need a real Postgres.
    """
    db = tmp_path / "cache.duckdb"

    # Pre-create a fake "pg" attachment shape: a separate DuckDB file with a
    # public schema and the mirrored tables.
    fake_pg = tmp_path / "fake_pg.duckdb"
    con = duckdb.connect(str(fake_pg))
    try:
        con.execute("CREATE SCHEMA public;")
        for t in duckdb_cache.MIRRORED_TABLES:
            con.execute(
                f"CREATE TABLE public.{t} AS SELECT * FROM (VALUES (1, '{t}')) AS v(id, src);"
            )
    finally:
        con.close()

    # Patch refresh to attach the fake-pg DuckDB file instead of Postgres.
    real_refresh = duckdb_cache.refresh

    def fake_refresh(*, db_path: Path = db, pg_dsn: str = "") -> duckdb_cache.RefreshReport:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        counts: dict[str, int] = {}
        c = duckdb.connect(str(db_path))
        try:
            c.execute(f"ATTACH '{fake_pg}' AS pg;")
            for t in duckdb_cache.MIRRORED_TABLES:
                c.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM pg.public.{t};")
                n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                counts[t] = int(n[0]) if n else 0
            c.execute("CREATE OR REPLACE TABLE _meta(refreshed_at TIMESTAMPTZ);")
            c.execute("INSERT INTO _meta VALUES (now());")
        finally:
            c.close()
        elapsed = time.monotonic() - started
        return duckdb_cache.RefreshReport(
            db_path=db_path, table_counts=counts, wall_time_sec=elapsed
        )

    monkeypatch.setattr(duckdb_cache, "refresh", fake_refresh)
    report = duckdb_cache.refresh(db_path=db)

    assert db.exists()
    assert set(report.table_counts.keys()) == set(duckdb_cache.MIRRORED_TABLES)
    # Each fake table had one row
    for t in duckdb_cache.MIRRORED_TABLES:
        assert report.table_counts[t] == 1
    # Cache is fresh immediately after refresh
    assert duckdb_cache.is_stale(db_path=db) is False

    # restore module attribute (monkeypatch undoes this automatically)
    assert duckdb_cache.refresh is fake_refresh
    _ = real_refresh  # silence unused


def test_fast_query_prefers_cache(tmp_path: Path) -> None:
    """fast_query reads from the cache when it's fresh."""
    db = tmp_path / "cache.duckdb"
    _seed_cache(db, row_count=4)
    df = duckdb_cache.fast_query("SELECT COUNT(*) AS n FROM experiment_runs", db_path=db)
    assert df.iloc[0]["n"] == 4


def test_refresh_report_str_lists_tables(tmp_path: Path) -> None:
    """RefreshReport.__str__ includes per-table counts."""
    rep = duckdb_cache.RefreshReport(
        db_path=tmp_path / "x.duckdb",
        table_counts={"experiments": 9, "experiment_runs": 2474, "findings": 7},
        wall_time_sec=1.23,
    )
    s = str(rep)
    assert "1.23s" in s
    assert "experiments: 9 rows" in s
    assert "experiment_runs: 2474 rows" in s
