"""Shared utilities for lab.rag: atomic IO, hashing, slugging, paths, time, console.

Vendored from kb-builder (formerly kb_builder.util). Marked private — not part
of lab's stable surface.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 80) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "unnamed"


def kb_name_ok(name: str) -> bool:
    return re.fullmatch(r"[a-z][a-z0-9-]{1,40}", name) is not None


def expanduser(p: str | Path) -> Path:
    return Path(os.path.expandvars(str(p))).expanduser().resolve()


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def append_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(content)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    import json

    body = "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n"
    atomic_write_text(path, body)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
