"""MinIO / S3 access for the eval dashboard.

Loads trajectory JSONs and other run artifacts. Env-driven; no creds in
the repo. Falls back gracefully when MinIO is offline so the dashboard
still renders.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

DEFAULT_ENDPOINT = "http://localhost:9000"
DEFAULT_BUCKET = "lab"


def _endpoint() -> str:
    return os.environ.get("LAB_S3_ENDPOINT", DEFAULT_ENDPOINT)


def _bucket() -> str:
    return os.environ.get("LAB_S3_BUCKET", DEFAULT_BUCKET)


@lru_cache(maxsize=1)
def s3_client():
    """Build a boto3 S3 client pointing at MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=_endpoint(),
        aws_access_key_id=os.environ.get("LAB_S3_ACCESS_KEY", "labadmin"),
        aws_secret_access_key=os.environ.get("LAB_S3_SECRET_KEY", ""),
        config=Config(signature_version="s3v4", retries={"max_attempts": 2}),
        region_name="us-east-1",
    )


def healthy() -> bool:
    """Cheap MinIO check: head_bucket."""
    try:
        s3_client().head_bucket(Bucket=_bucket())
        return True
    except (BotoCoreError, ClientError, Exception):
        return False


def parse_s3_path(path: str) -> tuple[str, str]:
    """Parse 's3://bucket/key' into (bucket, key). Bare keys assume default bucket."""
    if path.startswith("s3://"):
        parsed = urlparse(path)
        return parsed.netloc, parsed.path.lstrip("/")
    return _bucket(), path.lstrip("/")


def get_json(path: str) -> dict[str, Any] | list[Any] | None:
    """Fetch a JSON object from MinIO. None on failure."""
    bucket, key = parse_s3_path(path)
    try:
        obj = s3_client().get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def get_text(path: str, max_bytes: int = 1_000_000) -> str | None:
    """Fetch text content (e.g. JSONL traces). None on failure."""
    bucket, key = parse_s3_path(path)
    try:
        obj = s3_client().get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read(max_bytes)
        return body.decode("utf-8", errors="replace")
    except Exception:
        return None
