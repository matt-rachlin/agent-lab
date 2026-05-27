"""SQLite reader for the Phase-14 doc graph.

Reads ~/db/m/docs.db (read-only). Used by pages/5_Docs.py.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

DOCS_DB = Path.home() / "db" / "m" / "docs.db"


def _connect() -> sqlite3.Connection:
    # uri=True + mode=ro enforces read-only and avoids accidental writes.
    return sqlite3.connect(
        f"file:{DOCS_DB}?mode=ro",
        uri=True,
        isolation_level=None,
    )


def db_exists() -> bool:
    return DOCS_DB.is_file()


def query(sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    if not db_exists():
        return pd.DataFrame()
    try:
        with _connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)
    except Exception as e:
        return pd.DataFrame({"_error": [str(e)]})


def stats() -> dict[str, int]:
    """Total docs, orphans, gaps. Orphans = no incoming/outgoing edges."""
    out: dict[str, int] = {"total": 0, "orphans": 0, "gaps": 0}
    if not db_exists():
        return out
    try:
        with _connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM docs")
            out["total"] = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT COUNT(*) FROM docs d
                WHERE NOT EXISTS (SELECT 1 FROM doc_deps WHERE doc_id = d.doc_id)
                  AND NOT EXISTS (SELECT 1 FROM doc_deps WHERE dep_target = d.doc_id)
                """
            )
            out["orphans"] = int(cur.fetchone()[0])
            # Gaps = parse errors recorded by the doc scanner.
            cur.execute("SELECT COUNT(*) FROM parse_errors")
            out["gaps"] = int(cur.fetchone()[0])
    except Exception:  # noqa: S110 — render-on-failure is the design
        pass
    return out


def by_zone() -> pd.DataFrame:
    return query("SELECT zone, COUNT(*) AS n FROM docs GROUP BY zone ORDER BY n DESC")


def search(q: str, limit: int = 50) -> pd.DataFrame:
    like = f"%{q}%"
    return query(
        "SELECT doc_id, zone, kind, status, title, path FROM docs "
        "WHERE title LIKE ? OR doc_id LIKE ? OR path LIKE ? "
        "ORDER BY last_updated DESC LIMIT ?",
        (like, like, like, limit),
    )


def edges_for(doc_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (outgoing, incoming) edges for a doc."""
    out = query(
        "SELECT dep_kind, dep_target FROM doc_deps WHERE doc_id = ?",
        (doc_id,),
    )
    inc = query(
        "SELECT doc_id, dep_kind FROM doc_deps WHERE dep_target = ?",
        (doc_id,),
    )
    return out, inc
