"""Integration: upload + download a fake trace JSONL via MinIO."""

from __future__ import annotations

import json
import time
from io import BytesIO
from typing import Any

import pytest

pytestmark = pytest.mark.integration


def test_trace_round_trip(minio_client: Any, settings: Any) -> None:
    bucket = settings.s3_bucket
    if not minio_client.bucket_exists(bucket):
        pytest.skip(f"bucket {bucket!r} missing")
    key = f"runs/_test/{int(time.time() * 1000)}.jsonl"
    payload = {
        "run_id": "test-run-id",
        "response_text": "hello world",
        "latency_ms": 123,
    }
    body = (json.dumps(payload) + "\n").encode()
    minio_client.put_object(
        bucket, key, BytesIO(body), length=len(body), content_type="application/x-ndjson"
    )
    try:
        resp = minio_client.get_object(bucket, key)
        try:
            got = resp.read()
        finally:
            resp.close()
            resp.release_conn()
        line = got.decode("utf-8").splitlines()[0]
        loaded = json.loads(line)
        assert loaded == payload
    finally:
        minio_client.remove_object(bucket, key)
