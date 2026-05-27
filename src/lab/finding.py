"""Findings registry — sync `docs/findings/F-NNN-*.md` ↔ Postgres `findings` table.

Each finding is a markdown file: `docs/findings/F-NNN-slug.md`.
Required fields parsed from the document:
  - H1: `# F-NNN: <claim>`
  - Frontmatter-style block at top:
      Date: YYYY-MM-DD
      Confidence: low|medium|high
      Source: EXP <slug>
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Literal

import psycopg
from lab.core.settings import get_settings

FINDINGS_DIR_DEFAULT = Path("/data/lab/code/docs/findings")

H1_RE = re.compile(r"^#\s+(F-\d+):\s*(.+?)\s*$", re.MULTILINE)
META_RE = re.compile(
    r"^(date|confidence|source)\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
EXP_REF_RE = re.compile(r"EXP[-_ ]?([A-Z0-9_-]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedFinding:
    file: Path
    slug: str  # the "F-NNN" key
    claim: str  # the short title after the colon
    date: Date | None
    confidence: Literal["low", "medium", "high"] | None
    source_exp_slug: str | None


def parse_finding(path: Path) -> ParsedFinding | None:
    text = path.read_text(encoding="utf-8")
    h1 = H1_RE.search(text)
    if not h1:
        return None
    slug = h1.group(1)
    claim = h1.group(2)

    meta = {m.group(1).lower(): m.group(2).strip() for m in META_RE.finditer(text)}

    date_val: Date | None = None
    if meta.get("date"):
        try:
            date_val = Date.fromisoformat(meta["date"])
        except ValueError:
            date_val = None

    conf_raw = (meta.get("confidence") or "").lower()
    confidence: Literal["low", "medium", "high"] | None
    if conf_raw == "low":
        confidence = "low"
    elif conf_raw == "medium":
        confidence = "medium"
    elif conf_raw == "high":
        confidence = "high"
    else:
        confidence = None

    src = meta.get("source") or ""
    m = EXP_REF_RE.search(src)
    source_exp_slug = m.group(0).upper() if m else None
    if source_exp_slug and source_exp_slug.startswith("EXP "):
        source_exp_slug = source_exp_slug.replace(" ", "-")

    return ParsedFinding(
        file=path,
        slug=slug,
        claim=claim,
        date=date_val,
        confidence=confidence,
        source_exp_slug=source_exp_slug,
    )


def _experiment_id(slug: str | None) -> int | None:
    if not slug:
        return None
    # The source line might be "EXP-SWEEP-SMOKE-001" or just the slug — try direct match first
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT experiment_id FROM experiments WHERE slug = %s", (slug,))
        row = cur.fetchone()
        if row:
            return int(row[0])
        # Try without leading "EXP-"
        if slug.upper().startswith("EXP-"):
            stripped = slug[4:]
            cur.execute("SELECT experiment_id FROM experiments WHERE slug = %s", (stripped,))
            row = cur.fetchone()
            if row:
                return int(row[0])
    return None


_UPSERT = """
INSERT INTO findings
    (slug, claim, confidence, source_exp, doc_path, status, created_at)
VALUES (%(slug)s, %(claim)s, %(confidence)s, %(source_exp)s, %(doc_path)s, 'logged', %(created)s)
ON CONFLICT (slug) DO UPDATE SET
    claim       = EXCLUDED.claim,
    confidence  = EXCLUDED.confidence,
    source_exp  = EXCLUDED.source_exp,
    doc_path    = EXCLUDED.doc_path;
"""


def sync(findings_dir: Path = FINDINGS_DIR_DEFAULT) -> tuple[int, int]:
    """Walk `findings_dir`, upsert one row per F-NNN-*.md. Returns (synced, skipped)."""
    if not findings_dir.is_dir():
        return (0, 0)
    files = sorted(findings_dir.glob("F-*.md"))
    synced = skipped = 0
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        for f in files:
            parsed = parse_finding(f)
            if parsed is None:
                skipped += 1
                continue
            cur.execute(
                _UPSERT,
                {
                    "slug": parsed.slug,
                    "claim": parsed.claim,
                    "confidence": parsed.confidence or "low",
                    "source_exp": _experiment_id(parsed.source_exp_slug),
                    "doc_path": str(f).replace("/data/lab/code/", ""),
                    "created": parsed.date,
                },
            )
            synced += 1
    return (synced, skipped)


def list_findings() -> list[dict[str, object]]:
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.slug, f.claim, f.confidence, f.status, f.doc_path,
                   e.slug AS source_exp_slug, f.created_at
            FROM findings f
            LEFT JOIN experiments e ON e.experiment_id = f.source_exp
            ORDER BY f.slug DESC
            """,
        )
        cols = [d.name for d in cur.description] if cur.description else []
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


_TEMPLATE = """# {slug}: {claim_placeholder}

Date: {today}
Confidence: low
Source: EXP-<slug>

## Claim

## Evidence

## Caveats / limits

## Implications

## Open questions

## Status
- [ ] Logged
- [ ] Replicated
- [ ] Published
"""


def new_finding(
    slug: str,
    claim_placeholder: str = "<one-line claim>",
    *,
    dir_: Path = FINDINGS_DIR_DEFAULT,
) -> Path:
    """Scaffold a new F-NNN markdown file in the findings dir and return its path."""
    if not re.match(r"^F-\d+$", slug):
        raise ValueError(f"slug must look like F-NNN, got {slug!r}")
    dir_.mkdir(parents=True, exist_ok=True)
    safe = claim_placeholder.lower().replace(" ", "-")
    safe = re.sub(r"[^a-z0-9-]+", "", safe)[:48] or "untitled"
    out = dir_ / f"{slug}-{safe}.md"
    if out.exists():
        raise FileExistsError(out)
    from datetime import date as _date

    out.write_text(
        _TEMPLATE.format(
            slug=slug, claim_placeholder=claim_placeholder, today=_date.today().isoformat()
        ),
        encoding="utf-8",
    )
    return out
