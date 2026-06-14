"""Research-scout (ADR-010) scaffolding: assembled lab context + a deduped
recommendation queue.

Web search lives in the agent harness, not the local stack — so this provides the
durable, reusable parts (grounding context, a deduped store, triage list). A
search-capable agent reads `lab scout context`, scans the sources, and calls
`lab scout add` per cited finding. See docs/scout/lab-profile.md + ADR-010.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import psycopg

from lab.core.settings import get_settings

_ROOT = Path(__file__).resolve().parents[4]  # /data/lab/code
_DOCS = _ROOT / "docs"
_TITLE_RE = re.compile(r"^title:\s*['\"]?(.+?)['\"]?\s*$", re.MULTILINE)


def _frontmatter_title(md: str) -> str:
    if md.startswith("---"):
        m = _TITLE_RE.search(md.split("---", 2)[1])
        if m:
            return m.group(1).strip()
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "(untitled)"


def list_recommendations(status: str | None = None) -> list[dict[str, Any]]:
    sql = (
        "SELECT source_url, title, category, why_relevant, confidence, status, found_at "
        "FROM scout_recommendations"
    )
    params: tuple[Any, ...] = ()
    if status:
        sql += " WHERE status = %s"
        params = (status,)
    sql += " ORDER BY found_at DESC"
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description or []]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def add_recommendation(
    *,
    source_url: str,
    title: str,
    category: str,
    why_relevant: str,
    confidence: str = "medium",
) -> str:
    """Insert a recommendation; dedup on source_url. Returns 'added' | 'duplicate'."""
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO scout_recommendations "
            "(source_url, title, category, why_relevant, confidence) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (source_url) DO NOTHING RETURNING id",
            (source_url, title, category, why_relevant, confidence),
        )
        added = cur.fetchone() is not None
        conn.commit()
    return "added" if added else "duplicate"


def context_bundle() -> str:
    """The scout's grounding for a scan: charter + lab-profile (full), ADR/finding
    titles (awareness), and the existing recs (dedup — do not re-add these)."""
    parts: list[str] = ["# SCOUT CONTEXT (read before scanning)", ""]
    for rel in ("charter.md", "scout/lab-profile.md"):
        p = _DOCS / rel
        if p.exists():
            parts.append(f"## {rel}\n\n{p.read_text(encoding='utf-8')}\n")
    for label, sub in (("ADRs", "adr"), ("Findings", "findings")):
        d = _DOCS / sub
        if d.is_dir():
            titles = [
                f"- {_frontmatter_title(f.read_text(encoding='utf-8'))}"
                for f in sorted(d.glob("*.md"))
                if f.name != "index.md"
            ]
            if titles:
                parts.append(f"## {label} (awareness)\n\n" + "\n".join(titles) + "\n")
    parts.append("## Already recommended — DEDUP, do NOT re-add these source_urls\n")
    recs = list_recommendations()
    parts += (
        [f"- [{r['status']}] {r['title']} — {r['source_url']}" for r in recs]
        if recs
        else ["(none yet)"]
    )
    return "\n".join(parts) + "\n"
