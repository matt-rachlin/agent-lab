"""Shared fixtures + service-availability helpers for integration tests.

Each helper does a cheap probe and `pytest.skip(...)` with a clear reason if
the dependent service is unreachable, so a CI without services still passes.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import httpx
import psycopg
import pytest
import redis

from lab.core.settings import get_settings


def _tcp_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def settings() -> Any:
    return get_settings()


@pytest.fixture(scope="session")
def pg(settings: Any) -> Iterator[psycopg.Connection[Any]]:
    try:
        conn = psycopg.connect(settings.pg_dsn, connect_timeout=2)
    except psycopg.Error as exc:
        pytest.skip(f"postgres not reachable at {settings.pg_dsn}: {exc}")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM experiments LIMIT 1")
    except psycopg.Error as exc:
        conn.close()
        pytest.skip(f"postgres `lab` schema not initialised: {exc}")
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def valkey(settings: Any) -> Iterator[redis.Redis[str]]:
    try:
        client: redis.Redis[str] = redis.Redis.from_url(
            settings.redis_url, decode_responses=True, socket_timeout=1
        )
        client.ping()
    except (redis.RedisError, ConnectionError) as exc:
        pytest.skip(f"valkey not reachable at {settings.redis_url}: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="session")
def minio_client(settings: Any) -> Any:
    host_port = settings.s3_endpoint.removeprefix("http://").removeprefix("https://")
    host, _, port_s = host_port.partition(":")
    port = int(port_s) if port_s else 80
    if not _tcp_open(host, port):
        pytest.skip(f"minio not reachable at {settings.s3_endpoint}")
    if not settings.s3_secret_key:
        pytest.skip("LAB_S3_SECRET_KEY not set; cannot authenticate to MinIO")
    from minio import Minio

    client = Minio(
        host_port,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        secure=settings.s3_endpoint.startswith("https://"),
    )
    try:
        client.bucket_exists(settings.s3_bucket)
    except Exception as exc:
        pytest.skip(f"minio auth/health failed: {exc}")
    return client


@pytest.fixture(scope="session")
def litellm_url(settings: Any) -> str:
    url = settings.litellm_url.rstrip("/")
    try:
        r = httpx.get(f"{url}/health/liveliness", timeout=2)
        if r.status_code >= 500:
            pytest.skip(f"litellm proxy unhealthy ({r.status_code})")
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"litellm proxy not reachable at {url}: {exc}")
    return url
