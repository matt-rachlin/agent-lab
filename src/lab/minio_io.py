"""Shared MinIO upload/download helpers.

The sweep runner (single-turn) and the agent log writer (multi-turn) both
upload run traces under `runs/YYYY-MM/DD/<run_id>/...`. We keep one
implementation so the key layout and the MinIO client construction stay in
sync.
"""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from typing import Any

from lab.settings import get_settings


def make_minio_client() -> Any:
    """Build a `minio.Minio` configured from settings.

    Pulled into a helper so test code can monkeypatch one entry point.
    """

    from minio import Minio

    settings = get_settings()
    return Minio(
        settings.s3_endpoint.removeprefix("http://").removeprefix("https://"),
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        secure=settings.s3_endpoint.startswith("https://"),
    )


def run_key(run_id_: str, name: str, ts: datetime | None = None) -> str:
    """Return the MinIO object key for a per-run artifact.

    Format: `runs/YYYY-MM/DD/<run_id>/<name>`. `name` is the trailing path
    (e.g. `trace.jsonl`, `trajectory.jsonl`). `ts` defaults to "now" in UTC;
    callers pass it explicitly when they want a deterministic key for an
    earlier run.
    """

    ts = ts or datetime.now(UTC)
    return f"runs/{ts:%Y-%m/%d}/{run_id_}/{name}"


def upload_bytes(
    *,
    key: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload `data` to MinIO under `key`. Returns the `s3://bucket/key` URI."""

    settings = get_settings()
    client = make_minio_client()
    client.put_object(
        settings.s3_bucket,
        key,
        BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return f"s3://{settings.s3_bucket}/{key}"


__all__ = ["make_minio_client", "run_key", "upload_bytes"]
