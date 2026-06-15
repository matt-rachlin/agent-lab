"""Experiment pre-registration — validate a plan markdown and record git SHA.

A plan is "pre-registered" when:
  - The markdown contains every required section heading
  - It is committed to git (not dirty)
  - We record (slug, plan_path, plan_git_sha, pre_registered_at) in `experiments`

`lab exp register` enforces this; `lab sweep run --enforce-pre-registration` blocks
sweeps whose experiment row lacks a `plan_git_sha`.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from lab.core.settings import get_settings

# Section headings the plan must contain (case-insensitive, leading `## `).
REQUIRED_SECTIONS = (
    "hypothesis",
    "method",
    "success / failure criteria",
    "kill criteria",
)

# Slug must match this pattern, drawn from the H1 or filename.
# First char uppercase; subsequent chars may be upper/lowercase letters, digits, _ or -.
# Lowercase tail enables suffix variants like EXP-001b, EXP-002c.
SLUG_RE = re.compile(r"^[A-Z][A-Za-z0-9_-]{2,63}$")


@dataclass(frozen=True)
class PlanValidation:
    plan_path: Path
    slug: str
    title: str
    missing_sections: list[str]
    git_sha: str | None
    git_dirty: bool

    @property
    def ok(self) -> bool:
        return not self.missing_sections and self.git_sha is not None and not self.git_dirty


def _section_headings(text: str) -> set[str]:
    """Lower-cased H2 headings, with anything after a trailing parenthetical/em-dash stripped.

    Tolerates variants like "## Success / failure criteria (defined before running)".
    """
    out: set[str] = set()
    for line in text.splitlines():
        if line.startswith("## "):
            head = line[3:].strip().lower()
            # strip trailing parentheticals + em-dash / en-dash subtitles
            head = re.sub(r"\s*\(.*\)\s*$", "", head)  # `(anything)` at end
            head = re.sub(r"\s+[—–-]\s.*$", "", head)  # noqa: RUF001  # em-/en-/hyphen subtitle
            out.add(head.strip())
    return out


def _h1(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return None


def _slug_from_text(h1: str | None, fallback: str) -> str:
    """Extract a slug from the first H1 (assumed to start with the slug) or filename."""
    if h1:
        head = h1.split(":")[0].strip()
        if SLUG_RE.match(head):
            return head
    # Filename fallback: strip leading digits/underscore prefix and .md
    stem = Path(fallback).stem
    # Match SLUG-NNN pattern (letters + dashes + digits, not greedy-trailing-dash)
    m = re.match(r"^([A-Z][A-Z0-9_]*-\d+[a-z]?)", stem)
    if m:
        return m.group(1)
    m2 = re.match(r"^([A-Z][A-Z0-9_]+)", stem)
    if m2:
        return m2.group(1)
    return stem.upper()


def _git_state(path: Path) -> tuple[str | None, bool]:
    """(commit_sha_of_file, repo_dirty_at_path)."""
    try:
        sha = (
            subprocess.check_output(
                ["git", "log", "-n", "1", "--pretty=%H", "--", path.name],
                cwd=path.parent,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            or None
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", path.name],
                cwd=path.parent,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, True


def validate_plan(plan_path: Path) -> PlanValidation:
    text = plan_path.read_text(encoding="utf-8")
    headings = _section_headings(text)
    missing = [s for s in REQUIRED_SECTIONS if s not in headings]
    h1 = _h1(text)
    slug = _slug_from_text(h1, plan_path.name)
    title = (h1 or slug).split(":", 1)[-1].strip() or slug
    git_sha, git_dirty = _git_state(plan_path)
    return PlanValidation(
        plan_path=plan_path,
        slug=slug,
        title=title,
        missing_sections=missing,
        git_sha=git_sha,
        git_dirty=git_dirty,
    )


_SHA_PLACEHOLDER_RE = re.compile(
    r"<commit SHA filled by `?lab exp register`? at registration time>"
)


def _fill_sha_placeholder(plan_path: Path) -> str | None:
    """If the plan doc contains the pre-registration SHA placeholder, substitute HEAD SHA.

    Returns the substituted SHA on success, None if no placeholder was found.
    Writes the file in-place and runs ``git add -N`` so the change is intent-staged.
    """
    text = plan_path.read_text(encoding="utf-8")
    if not _SHA_PLACEHOLDER_RE.search(text):
        return None
    try:
        head_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=plan_path.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    new_text = _SHA_PLACEHOLDER_RE.sub(head_sha, text)
    plan_path.write_text(new_text, encoding="utf-8")
    subprocess.run(
        ["git", "add", "-N", str(plan_path)],
        cwd=plan_path.parent,
        check=False,
    )
    return head_sha


def register_plan(
    plan_path: Path,
    *,
    hypothesis: str | None = None,
    allow_dirty: bool = False,
    note: str | None = None,
) -> PlanValidation:
    """Register an experiment plan. Raises ValueError if validation fails (unless allow_dirty)."""
    # Fill the SHA placeholder before validation so git_state picks up the right SHA.
    _fill_sha_placeholder(plan_path)

    v = validate_plan(plan_path)
    if v.missing_sections:
        raise ValueError(f"plan missing required sections: {v.missing_sections}")
    if v.git_sha is None:
        raise ValueError(f"plan not committed to git: {plan_path}")
    if v.git_dirty and not allow_dirty:
        raise ValueError(
            f"plan has uncommitted changes: {plan_path} (pass --allow-dirty to override)"
        )

    rel_path = str(plan_path).replace("/data/lab/code/", "")

    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO experiments
                (slug, title, hypothesis, status, plan_path, plan_git_sha,
                 pre_registered_at, created_at)
            VALUES (%s, %s, %s, 'planned', %s, %s, %s, NOW())
            ON CONFLICT (slug) DO UPDATE SET
                title = EXCLUDED.title,
                plan_path = EXCLUDED.plan_path,
                plan_git_sha = EXCLUDED.plan_git_sha,
                pre_registered_at = EXCLUDED.pre_registered_at,
                hypothesis = COALESCE(EXCLUDED.hypothesis, experiments.hypothesis)
            """,
            (
                v.slug,
                v.title,
                hypothesis,
                rel_path,
                v.git_sha,
                datetime.now(UTC),
            ),
        )

    # MLflow additive mirror (Phase 15.2). Best-effort: never blocks the
    # canonical Postgres write. Capture the assigned MLflow experiment id
    # back into Postgres so the analysis tier can deep-link to the UI.
    try:
        from lab.observability.mlflow_mirror import MlflowMirror

        mlflow_id = MlflowMirror().upsert_experiment(
            v.slug,
            title=v.title,
            plan_path=rel_path,
            hypothesis=hypothesis,
        )
        if mlflow_id:
            with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE experiments SET mlflow_experiment_id = %s WHERE slug = %s",
                    (mlflow_id, v.slug),
                )
    except Exception:  # noqa: S110 — belt-and-suspenders; mirror already logs
        # Mirror must never break canonical registration; the mirror itself
        # already swallows + logs, this is belt-and-suspenders.
        pass

    _ = note  # reserved for future use (e.g. retroactive registration reason)
    return v


def is_pre_registered(slug: str) -> bool:
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT plan_git_sha FROM experiments WHERE slug = %s",
            (slug,),
        )
        row = cur.fetchone()
    return bool(row and row[0])


def list_experiments() -> list[dict[str, object]]:
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT slug, title, status, plan_path, plan_git_sha, pre_registered_at,
                   started_at, completed_at,
                   (SELECT COUNT(*) FROM experiment_runs WHERE experiment_id = e.experiment_id) AS n_runs
            FROM experiments e
            ORDER BY created_at DESC
            """,
        )
        cols = [d.name for d in cur.description] if cur.description else []
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def get_experiment(slug: str) -> dict[str, object] | None:
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.slug, e.title, e.hypothesis, e.status, e.plan_path, e.plan_git_sha,
                   e.pre_registered_at, e.created_at, e.started_at, e.completed_at,
                   (SELECT COUNT(*) FROM experiment_runs WHERE experiment_id = e.experiment_id) AS n_runs
            FROM experiments e
            WHERE e.slug = %s
            """,
            (slug,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [d.name for d in cur.description] if cur.description else []
        return dict(zip(cols, row, strict=True))
