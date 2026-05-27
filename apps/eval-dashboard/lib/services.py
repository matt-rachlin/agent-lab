"""Service health checks for the Home page.

Each check returns a (name, ok, detail) tuple. Failures must NOT raise —
the dashboard is read-only and should render even when half the stack is
down.
"""

from __future__ import annotations

import os
import socket
from typing import NamedTuple
from urllib.request import Request, urlopen

import psycopg


class ServiceStatus(NamedTuple):
    name: str
    ok: bool
    detail: str


def _http_ok(url: str, timeout: float = 1.5, expected: int = 200) -> tuple[bool, str]:
    try:
        req = Request(url, headers={"User-Agent": "eval-dashboard/1"})  # noqa: S310
        with urlopen(req, timeout=timeout) as r:  # noqa: S310
            code = r.getcode()
            return code == expected, f"HTTP {code}"
    except Exception as e:
        return False, type(e).__name__


def _tcp_ok(host: str, port: int, timeout: float = 1.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{host}:{port} reachable"
    except Exception as e:
        return False, type(e).__name__


def check_postgres() -> ServiceStatus:
    dsn = os.environ.get("LAB_PG_DSN", "postgresql://m@/lab")
    try:
        with (
            psycopg.connect(dsn, application_name="eval-dashboard", connect_timeout=2) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT 1")
            cur.fetchone()
        return ServiceStatus("postgres", True, "SELECT 1 ok")
    except Exception as e:
        return ServiceStatus("postgres", False, type(e).__name__)


def check_minio() -> ServiceStatus:
    endpoint = os.environ.get("LAB_S3_ENDPOINT", "http://localhost:9000")
    ok, detail = _http_ok(f"{endpoint}/minio/health/live")
    return ServiceStatus("minio", ok, detail)


def check_ollama() -> ServiceStatus:
    ok, detail = _http_ok("http://localhost:11434/api/version")
    return ServiceStatus("ollama", ok, detail)


def check_litellm() -> ServiceStatus:
    url = os.environ.get("LAB_LITELLM_URL", "http://localhost:4000")
    ok, detail = _http_ok(f"{url}/health/liveliness")
    if not ok:
        # Try a more generic root path as fallback.
        ok, detail = _http_ok(f"{url}/")
    return ServiceStatus("litellm", ok, detail)


def check_rerank_server() -> ServiceStatus:
    ok, detail = _http_ok("http://127.0.0.1:8401/healthz")
    return ServiceStatus("rerank-server", ok, detail)


def check_valkey() -> ServiceStatus:
    url = os.environ.get("LAB_REDIS_URL", "redis://localhost:6379/0")
    # Parse host/port out of the URL string without pulling redis-py.
    host = "localhost"
    port = 6379
    try:
        from urllib.parse import urlparse

        p = urlparse(url)
        if p.hostname:
            host = p.hostname
        if p.port:
            port = p.port
    except Exception:  # noqa: S110 — fall back to defaults
        pass
    ok, detail = _tcp_ok(host, port)
    return ServiceStatus("valkey", ok, detail)


def all_services() -> list[ServiceStatus]:
    """Run every service check. Order is the display order on Home."""
    return [
        check_postgres(),
        check_minio(),
        check_ollama(),
        check_litellm(),
        check_rerank_server(),
        check_valkey(),
    ]
