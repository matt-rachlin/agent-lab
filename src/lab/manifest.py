"""Manifest — the keystone helper.

Every run (sweep iteration, ad-hoc script, eval) captures an immutable manifest
that proves what produced the result. Without a manifest, the result does not
exist.

Usage:

    from lab.manifest import capture
    m = capture(extra={"config_hash": "...", "task_id": 42})
    # m.sha is the canonical id; m.payload is the full dict
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg

from lab.settings import get_settings


@dataclass(frozen=True)
class Manifest:
    sha: str
    payload: dict[str, Any]
    captured_at: datetime


def _git_state(repo: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def _uv_pip_freeze() -> tuple[str, str]:
    try:
        out = subprocess.check_output(["uv", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            out = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            out = ""
    sha = hashlib.sha256(out.encode()).hexdigest()
    return out, sha


def _nvidia_info() -> dict[str, str]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        line = out.strip().splitlines()[0] if out.strip() else ""
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            return {
                "gpu_name": parts[0],
                "driver_version": parts[1],
                "memory_total": parts[2],
                "memory_free": parts[3],
            }
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return {"gpu_name": "unknown", "driver_version": "unknown"}


def _cuda_version() -> str:
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "release" in line.lower():
                return line.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return "unknown"


def capture(
    *,
    repo: Path | None = None,
    extra: dict[str, Any] | None = None,
    persist: bool = True,
) -> Manifest:
    """Capture the full environment manifest for a run.

    Args:
        repo: Path to the git repo; defaults to the lab code repo.
        extra: Caller-supplied fields (config_hash, seeds, model ids, etc.).
        persist: If True, write to MinIO + insert row in `manifests` table.

    Returns:
        Manifest with deterministic sha256 id.
    """
    if repo is None:
        repo = Path(__file__).resolve().parents[2]

    git_sha, git_dirty = _git_state(repo)
    deps_blob, deps_sha = _uv_pip_freeze()
    nv = _nvidia_info()
    captured_at = datetime.now(UTC)

    payload: dict[str, Any] = {
        "captured_at": captured_at.isoformat(),
        "git": {"sha": git_sha, "dirty": git_dirty, "repo": str(repo)},
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "deps_sha256": deps_sha,
        "nvidia": nv,
        "cuda_version": _cuda_version(),
        "env": {k: os.environ.get(k, "") for k in ("USER", "HOSTNAME", "SHELL")},
        "extra": extra or {},
    }

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    sha = hashlib.sha256(canonical).hexdigest()

    if persist:
        _persist(sha, payload, deps_blob)

    return Manifest(sha=sha, payload=payload, captured_at=captured_at)


def _persist(sha: str, payload: dict[str, Any], deps_blob: str) -> None:
    """Write manifest to MinIO + Postgres `manifests` table."""
    settings = get_settings()
    s3_path = f"s3://{settings.s3_bucket}/manifests/{sha}.json"

    # Postgres insert
    with psycopg.connect(settings.pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO manifests
                (manifest_sha, s3_path, captured_at, git_sha, git_dirty,
                 python_version, deps_sha256, nvidia_driver, cuda_version,
                 gpu_name, payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (manifest_sha) DO NOTHING
            """,
            (
                sha,
                s3_path,
                payload["captured_at"],
                payload["git"]["sha"],
                payload["git"]["dirty"],
                payload["python"]["version"],
                payload["deps_sha256"],
                payload["nvidia"].get("driver_version"),
                payload["cuda_version"],
                payload["nvidia"].get("gpu_name"),
                json.dumps(payload),
            ),
        )

    # MinIO upload — manifest json + deps blob for re-creation
    try:
        from minio import Minio  # type: ignore[import-untyped]

        client = Minio(
            settings.s3_endpoint.removeprefix("http://").removeprefix("https://"),
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=settings.s3_endpoint.startswith("https://"),
        )
        from io import BytesIO

        manifest_bytes = json.dumps(payload, indent=2).encode()
        client.put_object(
            settings.s3_bucket,
            f"manifests/{sha}.json",
            BytesIO(manifest_bytes),
            length=len(manifest_bytes),
            content_type="application/json",
        )
        deps_bytes = deps_blob.encode()
        client.put_object(
            settings.s3_bucket,
            f"manifests/{sha}.deps.txt",
            BytesIO(deps_bytes),
            length=len(deps_bytes),
            content_type="text/plain",
        )
    except Exception as exc:  # noqa: BLE001
        # MinIO not yet up is OK during bootstrap — postgres row is the source of truth
        print(f"[manifest] minio upload skipped: {exc}", file=sys.stderr)


def main() -> None:
    """`uv run python -m lab.manifest --test` — capture a manifest and print summary."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()

    m = capture(persist=not args.no_persist)
    print(f"manifest sha: {m.sha}")
    print(f"git: {m.payload['git']}")
    print(f"gpu: {m.payload['nvidia'].get('gpu_name')}")
    print(f"deps_sha256: {m.payload['deps_sha256']}")
    print(f"captured_at: {m.captured_at.isoformat()}")
    if args.test:
        # Verify it's in the DB
        with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM manifests WHERE manifest_sha = %s", (m.sha,))
            count = cur.fetchone()[0]
            assert count == 1, f"expected 1 row, got {count}"
            print("postgres: row found OK")


if __name__ == "__main__":
    main()
