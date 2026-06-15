"""Findings registry — sync `docs/findings/F-NNN-*.md` ↔ Postgres `findings` table.

Each finding is a markdown file: `docs/findings/F-NNN-slug.md`.
Required fields parsed from the document:
  - H1: `# F-NNN: <claim>`
  - Frontmatter-style block at top:
      Date: YYYY-MM-DD
      Confidence: low|medium|high
      Source: EXP <slug>
      trust_level: unverified|verified|reliability_confirmed|deployable|retracted
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Literal

import psycopg

from lab.core.settings import get_settings

FINDINGS_DIR_DEFAULT = Path("/data/lab/code/docs/findings")

# ADR-008 finding-doc trust ladder (distinct from the run trust_level in lab.core.trust).
# Promotion must be sequential; --force bypasses the rung-skip guard.
TrustLevel = Literal[
    "unverified",
    "verified",
    "reliability_confirmed",
    "deployable",
    "retracted",
]

TRUST_RUNGS: tuple[TrustLevel, ...] = (
    "unverified",
    "verified",
    "reliability_confirmed",
    "deployable",
    "retracted",
)

# retracted is a terminal state reachable from any rung; not an "advancement".
_TERMINAL: set[str] = {"retracted"}

H1_RE = re.compile(r"^#\s+(F-\d+):\s*(.+?)\s*$", re.MULTILINE)
META_RE = re.compile(
    r"^(date|confidence|source|trust_level)\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
EXP_REF_RE = re.compile(r"EXP[-_ ]?([A-Z0-9_-]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedFinding:
    file: Path
    slug: str  # the "F-NNN" key
    claim: str  # the short title after the colon
    date: Date | None
    confidence: Literal["low", "medium", "high"] | None
    source_exp_slug: str | None
    trust_level: TrustLevel = "unverified"
    depends_on: str | None = None


def parse_finding(path: Path) -> ParsedFinding | None:
    text = path.read_text(encoding="utf-8")
    h1 = H1_RE.search(text)
    if not h1:
        return None
    slug = h1.group(1)
    claim = h1.group(2)

    meta = {m.group(1).lower(): m.group(2).strip() for m in META_RE.finditer(text)}

    # Also parse YAML frontmatter block for depends_on (not in inline meta).
    depends_on: str | None = None
    fm_match = FRONTMATTER_RE.match(text)
    if fm_match:
        fm_body = fm_match.group(1)
        dep_m = re.search(r"^depends_on\s*:\s*(.+?)\s*$", fm_body, re.MULTILINE)
        if dep_m:
            depends_on = dep_m.group(1).strip()

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

    trust_raw = (meta.get("trust_level") or "").lower()
    trust_level: TrustLevel
    if trust_raw in ("verified", "reliability_confirmed", "deployable", "retracted"):
        trust_level = trust_raw  # type: ignore[assignment]
    else:
        trust_level = "unverified"

    src = meta.get("source") or ""
    m_ref = EXP_REF_RE.search(src)
    source_exp_slug = m_ref.group(0).upper() if m_ref else None
    if source_exp_slug and source_exp_slug.startswith("EXP "):
        source_exp_slug = source_exp_slug.replace(" ", "-")

    return ParsedFinding(
        file=path,
        slug=slug,
        claim=claim,
        date=date_val,
        confidence=confidence,
        source_exp_slug=source_exp_slug,
        trust_level=trust_level,
        depends_on=depends_on,
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
    (slug, claim, confidence, source_exp, doc_path, status, created_at, min_trust_seen)
VALUES (%(slug)s, %(claim)s, %(confidence)s, %(source_exp)s, %(doc_path)s, 'logged',
        COALESCE(%(created)s, now()), 'legacy')
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
    successful: list[ParsedFinding] = []
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
            successful.append(parsed)
            synced += 1
    # Phase 15.2: additive MLflow mirror. Best-effort, never blocks.
    _mirror_findings_to_mlflow(successful)
    return (synced, skipped)


_CONFIDENCE_FLOAT = {"low": 0.3, "medium": 0.6, "high": 0.9}


def _mirror_findings_to_mlflow(parsed_findings: list[ParsedFinding]) -> None:
    if not parsed_findings:
        return
    try:
        from lab.observability.mlflow_mirror import MlflowMirror

        mirror = MlflowMirror()
        if not mirror.enabled:
            return
        for p in parsed_findings:
            conf_val = _CONFIDENCE_FLOAT.get(p.confidence or "low", 0.3)
            mlflow_run_id = mirror.log_finding(
                p.slug,
                claim=p.claim,
                importance=3,  # uniform default until plans add an `importance` field
                confidence=conf_val,
                evidence=[p.source_exp_slug] if p.source_exp_slug else None,
            )
            if mlflow_run_id:
                with (
                    psycopg.connect(get_settings().pg_dsn) as conn,
                    conn.cursor() as cur,
                ):
                    cur.execute(
                        "UPDATE findings SET mlflow_run_id = %s WHERE slug = %s",
                        (mlflow_run_id, p.slug),
                    )
    except Exception:  # noqa: S110 — belt-and-suspenders; mirror already logs
        pass


def list_findings(
    findings_dir: Path = FINDINGS_DIR_DEFAULT,
) -> list[dict[str, object]]:
    """Return findings from the DB, enriched with trust_level from the doc files."""
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.slug, f.claim, f.confidence, f.status, f.doc_path,
                   e.slug AS source_exp_slug, f.created_at
            FROM findings f
            LEFT JOIN experiments e ON e.experiment_id = f.source_exp
            ORDER BY f.slug ASC
            """,
        )
        cols = [d.name for d in cur.description] if cur.description else []
        rows = [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    # Enrich each row with the trust_level from its doc file.
    for row in rows:
        doc_path_str = str(row.get("doc_path") or "")
        doc_file: Path | None = None
        if doc_path_str:
            candidate = Path("/data/lab/code") / doc_path_str
            if candidate.exists():
                doc_file = candidate
        if doc_file is None:
            # Fallback: scan findings_dir for the slug prefix.
            slug = str(row["slug"])
            matches = list(findings_dir.glob(f"{slug}-*.md"))
            if matches:
                doc_file = matches[0]
        if doc_file is not None:
            parsed = parse_finding(doc_file)
            row["trust_level"] = parsed.trust_level if parsed else "unverified"
        else:
            row["trust_level"] = "unverified"

    return rows


_TEMPLATE = """# {slug}: {claim_placeholder}

Date: {today}
Confidence: low
Source: EXP-<slug>
trust_level: unverified

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


def _git_user_name() -> str:
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _find_finding_file(slug: str, findings_dir: Path = FINDINGS_DIR_DEFAULT) -> Path | None:
    """Return the Path of the first F-NNN-*.md matching the slug prefix."""
    matches = sorted(findings_dir.glob(f"{slug}-*.md"))
    if matches:
        return matches[0]
    # Also try exact filename (slug with no suffix)
    exact = findings_dir / f"{slug}.md"
    return exact if exact.exists() else None


def promote_finding(
    slug: str,
    target_level: TrustLevel,
    *,
    findings_dir: Path = FINDINGS_DIR_DEFAULT,
    force: bool = False,
) -> Path:
    """Promote a finding doc to a new trust_level (ADR-008).

    Rules:
    - Rungs must be advanced in order (unverified -> verified -> reliability_confirmed
      -> deployable) unless --force is set.
    - retracted is reachable from any rung (terminal; no further advancement).
    - Refuses if the finding lacks a ``depends_on`` evidence link when advancing
      beyond ``unverified`` (importance gate per ADR-004).
    Returns the path of the updated file.
    """
    if target_level not in TRUST_RUNGS:
        raise ValueError(f"unknown trust level {target_level!r}")

    doc = _find_finding_file(slug, findings_dir)
    if doc is None:
        raise FileNotFoundError(f"no finding doc found for slug {slug!r} in {findings_dir}")

    parsed = parse_finding(doc)
    if parsed is None:
        raise ValueError(f"could not parse finding doc at {doc}")

    current = parsed.trust_level

    # Retracted is a terminal state — no further promotion allowed.
    if current == "retracted":
        raise ValueError(
            f"{slug} is already retracted; retraction is terminal (cannot promote further)"
        )

    # Rung-skip check (retracted is always reachable, skip check for it).
    if target_level != "retracted":
        curr_idx = TRUST_RUNGS.index(current) if current in TRUST_RUNGS else 0
        tgt_idx = TRUST_RUNGS.index(target_level)
        if tgt_idx != curr_idx + 1 and not force:
            raise ValueError(
                f"rung skip: {slug} is at {current!r}; cannot jump to {target_level!r} "
                f"without --force. Advance one rung at a time."
            )

    # Evidence gate: non-unverified promotions require depends_on.
    if target_level not in ("unverified", "retracted") and not parsed.depends_on:
        raise ValueError(
            f"{slug} has no 'depends_on' evidence link in frontmatter. "
            f"Promotion to {target_level!r} requires an evidence link (ADR-004). "
            f"Add 'depends_on: EXP-NNN-slug' to the YAML frontmatter."
        )

    text = doc.read_text(encoding="utf-8")
    today = Date.today().isoformat()
    actor = _git_user_name()

    # Update or insert the trust_level inline-meta line.
    trust_pattern = re.compile(r"^trust_level\s*:\s*.+?$", re.MULTILINE)
    new_trust_line = f"trust_level: {target_level}"
    if trust_pattern.search(text):
        text = trust_pattern.sub(new_trust_line, text, count=1)
    else:
        # Insert after the Source: line if present, else after Date: line.
        source_pat = re.compile(r"^(Source\s*:.+)$", re.MULTILINE)
        if source_pat.search(text):
            text = source_pat.sub(r"\1\n" + new_trust_line, text, count=1)
        else:
            date_pat = re.compile(r"^(Date\s*:.+)$", re.MULTILINE)
            if date_pat.search(text):
                text = date_pat.sub(r"\1\n" + new_trust_line, text, count=1)
            else:
                # No metadata lines found: append to top of body.
                text = new_trust_line + "\n" + text

    # Append to ## Promotion history section.
    history_entry = f"- {today}: {current} -> {target_level} (by {actor})"
    history_header = "## Promotion history"
    if history_header in text:
        text = text.rstrip("\n") + "\n" + history_entry + "\n"
    else:
        text = text.rstrip("\n") + f"\n\n{history_header}\n{history_entry}\n"

    doc.write_text(text, encoding="utf-8")
    return doc


def backfill_trust(
    findings_dir: Path = FINDINGS_DIR_DEFAULT,
) -> tuple[int, int]:
    """Set trust_level: unverified on every finding doc that lacks the field.

    Returns (updated, already_set).
    """
    updated = already_set = 0
    for path in sorted(findings_dir.glob("F-*.md")):
        text = path.read_text(encoding="utf-8")
        trust_pat = re.compile(r"^trust_level\s*:\s*.+?$", re.MULTILINE)
        if trust_pat.search(text):
            already_set += 1
            continue
        # Insert after the Source: line if present, else after Date: line.
        source_pat = re.compile(r"^(Source\s*:.+)$", re.MULTILINE)
        if source_pat.search(text):
            text = source_pat.sub(r"\1\ntrust_level: unverified", text, count=1)
        else:
            date_pat = re.compile(r"^(Date\s*:.+)$", re.MULTILINE)
            if date_pat.search(text):
                text = date_pat.sub(r"\1\ntrust_level: unverified", text, count=1)
            else:
                # Last resort: append at bottom.
                text = text.rstrip("\n") + "\ntrust_level: unverified\n"
        path.write_text(text, encoding="utf-8")
        updated += 1
    return (updated, already_set)


def _run_trust_level(run_id: str) -> str | None:
    """The trust_level of a run, or None if the run does not exist (ADR-008)."""
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT trust_level FROM experiment_runs WHERE run_id = %s", (run_id,))
        row = cur.fetchone()
        return str(row[0]) if row else None


def new_finding(
    slug: str,
    claim_placeholder: str = "<one-line claim>",
    *,
    source_run_id: str | None = None,
    dir_: Path = FINDINGS_DIR_DEFAULT,
) -> Path:
    """Scaffold a new F-NNN markdown file in the findings dir and return its path.

    If ``source_run_id`` is given the run must be at trust_level ``verified`` (or
    ``finding``): a finding may only be minted from a verified result (ADR-008).
    Omit the run only for exploratory/methodological notes.
    """
    if not re.match(r"^F-\d+$", slug):
        raise ValueError(f"slug must look like F-NNN, got {slug!r}")
    if source_run_id is not None:
        lvl = _run_trust_level(source_run_id)
        if lvl is None:
            raise ValueError(f"run {source_run_id!r} not found")
        if lvl not in ("verified", "finding"):
            raise ValueError(
                f"run {source_run_id!r} is trust_level={lvl!r}; a finding may only be "
                "minted from a 'verified' run (ADR-008)"
            )
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
    subprocess.run(
        ["git", "add", "-N", str(out)],
        cwd=out.parent,
        check=False,
    )
    return out
