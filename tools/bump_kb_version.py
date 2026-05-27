"""Stamp a fresh ``kb_version`` token into a KB manifest.yaml.

Designed to run before ``just kb-publish <name>`` so the manifest captures
the build identity of the next DVC revision. We use a short, sortable
timestamp + 4 random hex chars; this stays human-readable and unique
across rebuilds without needing to consult git.

Usage:
    uv run python tools/bump_kb_version.py kbs/<name>/manifest.yaml

The script ONLY mutates the manifest's ``kb_version`` field; it does not
call DVC. ``just kb-publish`` runs ``dvc add`` separately, which captures
the new file content automatically.
"""

from __future__ import annotations

import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True


def make_version_token() -> str:
    """Return a ``YYYYMMDD-HHMMSS-xxxx`` style token (UTC).

    Sortable, ~20 chars, no path-unsafe characters.
    """

    now = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(2)
    return f"{now}-{suffix}"


def bump(manifest_path: Path) -> str:
    """Read ``manifest_path``, set ``kb_version``, write it back, return token."""

    with manifest_path.open() as fh:
        data = yaml.load(fh) or {}
    token = make_version_token()
    data["kb_version"] = token
    with manifest_path.open("w") as fh:
        yaml.dump(data, fh)
    return token


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: bump_kb_version.py <path-to-manifest.yaml>",
            file=sys.stderr,
        )
        return 2
    manifest = Path(argv[1])
    if not manifest.is_file():
        print(f"not found: {manifest}", file=sys.stderr)
        return 1
    token = bump(manifest)
    print(token)
    return 0


if __name__ == "__main__":
    # Tests pass ``__argv__`` through ``runpy``; the real CLI uses sys.argv.
    raw = globals().get("__argv__") or sys.argv
    raise SystemExit(main(list(raw)))
