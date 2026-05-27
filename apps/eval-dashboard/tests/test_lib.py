"""Unit tests for apps/eval-dashboard/lib/.

These do not require live Postgres / MinIO / Valkey. Anything that
would touch them is patched. Run with:

    cd apps/eval-dashboard && uv run pytest tests/ -q

or via the repo-root convention once `just dash-test` lands.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Make `lib` importable regardless of where pytest is invoked from.
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from lib import db as db_lib  # noqa: E402
from lib import docs as docs_lib  # noqa: E402
from lib import minio as minio_lib  # noqa: E402
from lib import services as services_lib  # noqa: E402

# ----- db.py ------------------------------------------------------------


def test_pg_dsn_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAB_PG_DSN", raising=False)
    assert db_lib.pg_dsn() == "postgresql://m@/lab"


def test_pg_dsn_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_PG_DSN", "postgresql://otheruser@/other")
    assert db_lib.pg_dsn() == "postgresql://otheruser@/other"


def test_pg_healthy_returns_false_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Point at a port nothing's listening on - must NOT raise.
    monkeypatch.setenv("LAB_PG_DSN", "postgresql://nobody@127.0.0.1:1/nodb")
    assert db_lib.pg_healthy() is False


def test_pg_query_returns_error_df_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_PG_DSN", "postgresql://nobody@127.0.0.1:1/nodb")
    out = db_lib.pg_query("SELECT 1")
    assert isinstance(out, pd.DataFrame)
    assert "_error" in out.columns


def test_aggregate_stats_falls_back_when_db_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_PG_DSN", "postgresql://nobody@127.0.0.1:1/nodb")
    stats = db_lib.aggregate_stats()
    assert stats == {"experiments": 0, "runs": 0, "findings": 0, "spend_7d_usd": 0.0}


# ----- minio.py ---------------------------------------------------------


def test_parse_s3_path_uri() -> None:
    bucket, key = minio_lib.parse_s3_path("s3://mybucket/runs/abc.json")
    assert bucket == "mybucket"
    assert key == "runs/abc.json"


def test_parse_s3_path_bare_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_S3_BUCKET", "lab")
    bucket, key = minio_lib.parse_s3_path("runs/abc.json")
    assert bucket == "lab"
    assert key == "runs/abc.json"


def test_minio_healthy_false_on_bad_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_S3_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("LAB_S3_ACCESS_KEY", "x")
    monkeypatch.setenv("LAB_S3_SECRET_KEY", "y")
    minio_lib.s3_client.cache_clear()
    assert minio_lib.healthy() is False


def test_minio_get_json_returns_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_S3_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("LAB_S3_ACCESS_KEY", "x")
    monkeypatch.setenv("LAB_S3_SECRET_KEY", "y")
    minio_lib.s3_client.cache_clear()
    assert minio_lib.get_json("s3://nope/nope.json") is None


# ----- services.py ------------------------------------------------------


def test_check_postgres_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_PG_DSN", "postgresql://nobody@127.0.0.1:1/nodb")
    s = services_lib.check_postgres()
    assert s.name == "postgres"
    assert s.ok is False


def test_all_services_returns_six_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    # Point env vars at dead ports for the services we control. Others
    # (e.g. ollama at the hard-coded localhost:11434) may legitimately
    # be up on the dev box - we only require that the check runs and
    # returns a valid ServiceStatus, not that it reports red.
    monkeypatch.setenv("LAB_PG_DSN", "postgresql://nobody@127.0.0.1:1/nodb")
    monkeypatch.setenv("LAB_S3_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("LAB_REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setenv("LAB_LITELLM_URL", "http://127.0.0.1:1")
    statuses = services_lib.all_services()
    assert len(statuses) == 6
    names = {s.name for s in statuses}
    assert names == {"postgres", "minio", "ollama", "litellm", "rerank-server", "valkey"}
    # The services we explicitly pointed at dead ports MUST be red. Each
    # status object also exposes a detail string.
    by_name = {s.name: s for s in statuses}
    assert by_name["postgres"].ok is False
    assert by_name["minio"].ok is False
    assert by_name["valkey"].ok is False
    assert by_name["litellm"].ok is False
    for s in statuses:
        assert isinstance(s.detail, str)


# ----- docs.py ----------------------------------------------------------


def _make_test_docsdb(tmp_path: Path) -> Path:
    p = tmp_path / "docs.db"
    conn = sqlite3.connect(p)
    conn.executescript(
        """
        CREATE TABLE docs (
            doc_id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE,
            zone TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
            owner TEXT NOT NULL, title TEXT NOT NULL,
            created DATE NOT NULL, last_updated DATE NOT NULL,
            last_verified DATE, supersedes TEXT, content_hash TEXT NOT NULL,
            parsed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE doc_deps (
            doc_id TEXT NOT NULL, dep_kind TEXT NOT NULL, dep_target TEXT NOT NULL,
            PRIMARY KEY (doc_id, dep_kind, dep_target)
        );
        CREATE TABLE doc_tags (
            doc_id TEXT NOT NULL, tag TEXT NOT NULL,
            PRIMARY KEY (doc_id, tag)
        );
        CREATE TABLE doc_history (
            doc_id TEXT NOT NULL, content_hash TEXT NOT NULL,
            observed_at TIMESTAMP NOT NULL,
            PRIMARY KEY (doc_id, observed_at)
        );
        CREATE TABLE parse_errors (
            path TEXT PRIMARY KEY, error TEXT NOT NULL,
            observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO docs VALUES
            ('f-001', '/a/F-001.md', 'lab',       'finding', 'active', 'm',
             'F-001 plumbing', '2026-05-25', '2026-05-25', NULL, NULL, 'h1', CURRENT_TIMESTAMP),
            ('exp-001', '/a/EXP-001.md', 'lab',   'exp',     'active', 'm',
             'EXP-001 12gb', '2026-05-25', '2026-05-26', NULL, NULL, 'h2', CURRENT_TIMESTAMP),
            ('lonely', '/a/lonely.md',   'workspace', 'note', 'active', 'm',
             'lonely note', '2026-05-25', '2026-05-25', NULL, NULL, 'h3', CURRENT_TIMESTAMP);
        INSERT INTO doc_deps VALUES ('f-001', 'doc', 'exp-001');
        INSERT INTO parse_errors VALUES ('/a/broken.md', 'YAML error', CURRENT_TIMESTAMP);
        """
    )
    conn.commit()
    conn.close()
    return p


def test_docs_stats_against_test_db(tmp_path: Path) -> None:
    test_db = _make_test_docsdb(tmp_path)
    with patch.object(docs_lib, "DOCS_DB", test_db):
        assert docs_lib.db_exists() is True
        s = docs_lib.stats()
        assert s["total"] == 3
        # exp-001 has incoming (f-001 depends on it); f-001 has outgoing;
        # 'lonely' has neither -> orphan.
        assert s["orphans"] == 1
        assert s["gaps"] == 1


def test_docs_by_zone_against_test_db(tmp_path: Path) -> None:
    test_db = _make_test_docsdb(tmp_path)
    with patch.object(docs_lib, "DOCS_DB", test_db):
        df = docs_lib.by_zone()
        assert set(df["zone"].tolist()) == {"lab", "workspace"}
        # lab has 2 entries, workspace has 1
        zone_counts = dict(zip(df["zone"], df["n"], strict=True))
        assert zone_counts["lab"] == 2
        assert zone_counts["workspace"] == 1


def test_docs_search_against_test_db(tmp_path: Path) -> None:
    test_db = _make_test_docsdb(tmp_path)
    with patch.object(docs_lib, "DOCS_DB", test_db):
        hits = docs_lib.search("plumbing")
        assert len(hits) == 1
        assert hits.iloc[0]["doc_id"] == "f-001"


def test_docs_edges_against_test_db(tmp_path: Path) -> None:
    test_db = _make_test_docsdb(tmp_path)
    with patch.object(docs_lib, "DOCS_DB", test_db):
        out, inc = docs_lib.edges_for("f-001")
        assert len(out) == 1
        assert out.iloc[0]["dep_target"] == "exp-001"
        assert len(inc) == 0
        out2, inc2 = docs_lib.edges_for("exp-001")
        assert len(out2) == 0
        assert len(inc2) == 1
        assert inc2.iloc[0]["doc_id"] == "f-001"


def test_docs_db_missing_returns_safe_defaults(tmp_path: Path) -> None:
    with patch.object(docs_lib, "DOCS_DB", tmp_path / "nope.db"):
        assert docs_lib.db_exists() is False
        assert docs_lib.stats() == {"total": 0, "orphans": 0, "gaps": 0}
        assert docs_lib.by_zone().empty


# Sanity touch: keep `os` import linked to a real test so linters don't
# complain about unused imports if these fixtures are pruned.
def test_env_is_present() -> None:
    assert isinstance(os.environ, os._Environ)
