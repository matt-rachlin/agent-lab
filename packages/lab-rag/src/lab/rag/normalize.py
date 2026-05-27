"""Normalize fetched bytes to markdown with YAML front-matter.

Vendored from kb_builder.normalize.
"""

from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO
from typing import Any

from ruamel.yaml import YAML

_yaml = YAML(typ="safe")
_yaml.default_flow_style = False


def render_frontmatter(meta: dict[str, Any]) -> str:
    buf = StringIO()
    _yaml.dump(meta, buf)
    return f"---\n{buf.getvalue()}---\n"


def make_normalized(
    *,
    body: str,
    source_url: str,
    sha256: str,
    retrieved_at: str | datetime,
    title: str | None = None,
    license: str | None = None,
    authority: str = "community",
    source_type: str = "html-single-page",
    extra: dict[str, Any] | None = None,
) -> str:
    """Body should already be markdown (or close to it)."""
    if isinstance(retrieved_at, datetime):
        retrieved_at = (
            retrieved_at.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        )
    meta: dict[str, Any] = {
        "source_url": source_url,
        "sha256": sha256,
        "retrieved_at": retrieved_at,
        "title": title or "",
        "license": license or "",
        "authority": authority,
        "source_type": source_type,
    }
    if extra:
        meta["extra"] = extra
    return render_frontmatter(meta) + "\n" + body.strip() + "\n"
