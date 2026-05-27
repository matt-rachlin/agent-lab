"""Local DuckDB analytics cache mirrored from Postgres.

Postgres holds the source of truth (`experiments`, `experiment_runs`,
`eval_results`, `agent_logs`, `findings`, `models`, `tasks`). The lab is
sub-10k rows per table, so we mirror those tables into a single DuckDB file
on local disk for sub-second ad-hoc analytics queries — instead of paying
per-query psycopg roundtrip + planner cost.

The cache file lives at `~/.cache/lab/analytics.duckdb` by default.

Public API:
    refresh(*, db_path=...) -> RefreshReport
        Replace all mirrored tables. Full refresh (not incremental).
        Fast for our scale (~12k total rows) — ~1-3 s wall-clock.

    is_stale(*, max_age_min=30, db_path=...) -> bool
        Whether the cache is older than `max_age_min` minutes.

    query(sql, *, db_path=...) -> pd.DataFrame
        Run a SQL query against the cached DuckDB. Fails loudly if the
        cache file is missing (call `refresh()` first).

    fast_query(sql, *, max_age_min=30, db_path=...) -> pd.DataFrame
        Prefer the cache; transparent fallback to live Postgres scan via
        DuckDB's postgres_scanner extension if the cache is missing or
        stale. Logs which path was taken to stderr.

Connection-string pattern matches `lab.analyze.queries.open_db`.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd  # type: ignore[import-untyped]

# Default Postgres connection — matches the convention in the analyzer
# scripts and lab.analyze.queries. `host=/var/run/postgresql` keeps us on
# the local UNIX socket.
DEFAULT_PG_DSN = "dbname=lab host=/var/run/postgresql"

CACHE_DEFAULT = Path("~/.cache/lab/analytics.duckdb").expanduser()

# Tables we mirror. Order matters only for log readability.
MIRRORED_TABLES: tuple[str, ...] = (
    "experiments",
    "experiment_runs",
    "eval_results",
    "agent_logs",
    "findings",
    "models",
    "tasks",
)


@dataclass
class RefreshReport:
    """Result of a `refresh()` call."""

    db_path: Path
    table_counts: dict[str, int] = field(default_factory=dict)
    wall_time_sec: float = 0.0
    refreshed_at: datetime = field(
        default_factory=lambda: datetime.now(tz=UTC)
    )

    def __str__(self) -> str:
        lines = [f"refreshed {self.db_path} in {self.wall_time_sec:.2f}s"]
        for t in MIRRORED_TABLES:
            n = self.table_counts.get(t)
            if n is not None:
                lines.append(f"  {t}: {n} rows")
        return "\n".join(lines)


def _ensure_parent(db_path: Path) -> None:
    """mkdir -p the cache file's parent dir."""
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _open_with_postgres(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB file connection + load the postgres extension + ATTACH lab.

    The attach name is `pg` (vs. `lab` used by lab.analyze.queries) to make it
    unambiguous when reading: `FROM pg.public.<t>` is always Postgres-live.
    """
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL postgres; LOAD postgres;")
    con.execute(f"ATTACH '{DEFAULT_PG_DSN}' AS pg (TYPE postgres, READ_ONLY);")
    return con


def refresh(
    *,
    db_path: Path = CACHE_DEFAULT,
    pg_dsn: str = DEFAULT_PG_DSN,
) -> RefreshReport:
    """Mirror Postgres tables into the DuckDB cache file.

    Uses `CREATE OR REPLACE TABLE ... AS SELECT * FROM pg.public.<t>` for
    each mirrored table. This is a full refresh — we don't track an
    incremental cursor. Fast enough at our scale.
    """
    _ensure_parent(db_path)
    started = time.monotonic()
    counts: dict[str, int] = {}
    con = duckdb.connect(str(db_path))
    try:
        con.execute("INSTALL postgres; LOAD postgres;")
        con.execute(f"ATTACH '{pg_dsn}' AS pg (TYPE postgres, READ_ONLY);")
        for t in MIRRORED_TABLES:
            # CREATE OR REPLACE is the idiomatic full-refresh primitive.
            con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM pg.public.{t};")
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
            counts[t] = int(n[0]) if n else 0
        # Record refresh time in a small meta table so is_stale() can read it.
        con.execute("CREATE OR REPLACE TABLE _meta(refreshed_at TIMESTAMPTZ);")
        con.execute("INSERT INTO _meta VALUES (now());")
    finally:
        con.close()
    elapsed = time.monotonic() - started
    return RefreshReport(
        db_path=db_path,
        table_counts=counts,
        wall_time_sec=elapsed,
    )


def is_stale(*, max_age_min: int = 30, db_path: Path = CACHE_DEFAULT) -> bool:
    """Whether the cache is missing or older than `max_age_min` minutes."""
    if not db_path.exists():
        return True
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            row = con.execute(
                "SELECT (now() - refreshed_at) AS age FROM _meta LIMIT 1"
            ).fetchone()
        finally:
            con.close()
    except duckdb.Error:
        # No _meta table → treat as stale (probably a stale or corrupt cache).
        return True
    if not row or row[0] is None:
        return True
    # `age` arrives as a python timedelta from duckdb.
    age = row[0]
    return bool(age.total_seconds() > max_age_min * 60)


def query(sql: str, *, db_path: Path = CACHE_DEFAULT) -> pd.DataFrame:
    """Run a SQL query against the cached DuckDB. Returns a DataFrame.

    Raises FileNotFoundError if the cache file doesn't exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"analytics cache not found at {db_path} — run `refresh()` first"
        )
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def fast_query(
    sql: str,
    *,
    db_path: Path = CACHE_DEFAULT,
    max_age_min: int = 30,
    pg_dsn: str = DEFAULT_PG_DSN,
    verbose: bool = False,
) -> pd.DataFrame:
    """Prefer the cache; fall back to live Postgres scan on miss or stale.

    On fallback, attaches Postgres via the `pg` schema. Caller SQL must use
    plain table names (e.g. `FROM experiments`) — we transparently make them
    resolve in both modes by attaching `pg` and prepending `USE pg.public`
    when falling back? No — DuckDB ATTACH doesn't allow USE in the way SQL
    Server does. Cleanest is to let callers query unqualified tables, and on
    fallback we create temp views: `CREATE VIEW <t> AS SELECT * FROM
    pg.public.<t>;`
    """
    if not is_stale(max_age_min=max_age_min, db_path=db_path):
        if verbose:
            print(f"fast_query: HIT cache={db_path}", file=sys.stderr)
        return query(sql, db_path=db_path)

    # Cache miss / stale → live Postgres path via an ephemeral in-memory DB.
    if verbose:
        print("fast_query: MISS → live PG (cache stale or missing)", file=sys.stderr)
    con = duckdb.connect(":memory:")
    try:
        con.execute("INSTALL postgres; LOAD postgres;")
        con.execute(f"ATTACH '{pg_dsn}' AS pg (TYPE postgres, READ_ONLY);")
        # Expose the mirrored tables as in-memory views so the same SQL works.
        for t in MIRRORED_TABLES:
            con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM pg.public.{t};")
        return con.execute(sql).df()
    finally:
        con.close()


def _summarize(report: RefreshReport) -> str:
    """One-line summary suitable for log output."""
    total = sum(report.table_counts.values())
    return (
        f"refreshed {len(report.table_counts)} tables, {total} rows total, "
        f"in {report.wall_time_sec:.2f}s → {report.db_path}"
    )
