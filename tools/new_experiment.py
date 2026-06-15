"""Scaffold a new experiment: pre-reg doc, sweep config, analysis dir, DB row.

Usage:
    uv run python tools/new_experiment.py <slug>            # e.g. retrieval-ablation
    uv run python tools/new_experiment.py <slug> --dry-run  # print plan only
    uv run python tools/new_experiment.py <slug> --shape retrieval

Creates:
    docs/exp/EXP-<NNN>-<slug>.md     copy of docs/exp/_template.md
    conf/sweep/EXP-<NNN>.yaml         copy of conf/sweep/_template.yaml
    analysis/EXP-<NNN>/.gitkeep       empty dir for outputs

Registers a placeholder row in `experiments` (status='planned', no
plan_git_sha) so the slug is reserved. The plan is properly
pre-registered later via `lab exp register docs/exp/EXP-<NNN>-<slug>.md`
after the doc has been filled in and committed.

Idempotent: re-running with the same slug is a no-op (refuses with a
non-zero exit if the EXP doc already exists).

NNN auto-assigns to max(existing EXP-NNN) + 1 across DB + filesystem.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import psycopg
import typer
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_EXP = REPO_ROOT / "docs" / "exp"
CONF_SWEEP = REPO_ROOT / "conf" / "sweep"
ANALYSIS = REPO_ROOT / "analysis"
DOC_TEMPLATE = DOCS_EXP / "_template.md"
SWEEP_TEMPLATE = CONF_SWEEP / "_template.yaml"

SLUG_TAIL_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
EXP_FILENAME_RE = re.compile(r"^EXP-(\d{3,4})(?:[a-z])?(?:-.*)?$", re.IGNORECASE)
EXP_DB_SLUG_RE = re.compile(r"^EXP-(\d{3,4})(?:[a-z])?$", re.IGNORECASE)

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _next_exp_number(dsn: str | None = None) -> int:
    """Max(EXP-NNN across DB + docs/exp + conf/sweep) + 1."""
    seen: set[int] = set()
    for f in DOCS_EXP.glob("EXP-*.md"):
        m = EXP_FILENAME_RE.match(f.stem)
        if m:
            seen.add(int(m.group(1)))
    for f in CONF_SWEEP.glob("EXP-*.yaml"):
        m = EXP_FILENAME_RE.match(f.stem)
        if m:
            seen.add(int(m.group(1)))
    if dsn:
        try:
            with psycopg.connect(dsn) as conn, conn.cursor() as cur:
                cur.execute("SELECT slug FROM experiments")
                for (slug,) in cur.fetchall():
                    m = EXP_DB_SLUG_RE.match(slug)
                    if m:
                        seen.add(int(m.group(1)))
        except psycopg.Error:
            # DB unavailable — fall back to filesystem only.
            pass
    return (max(seen) + 1) if seen else 1


def _validate_slug(tail: str) -> str:
    """Validate the user-supplied slug tail (the part after EXP-NNN-)."""
    tail = tail.strip().lower()
    if not SLUG_TAIL_RE.match(tail):
        raise typer.BadParameter(
            f"slug {tail!r} must be lowercase letters/digits/hyphens (2-64 chars), "
            f"start with a letter or digit"
        )
    return tail


def _exp_slug(num: int) -> str:
    return f"EXP-{num:03d}"


def _doc_path(num: int, tail: str) -> Path:
    return DOCS_EXP / f"{_exp_slug(num)}-{tail}.md"


def _sweep_path(num: int) -> Path:
    return CONF_SWEEP / f"{_exp_slug(num)}.yaml"


def _analysis_dir(num: int) -> Path:
    return ANALYSIS / _exp_slug(num)


def _render_doc_template(num: int, tail: str, today: str) -> str:
    """Copy the template, substituting the slug and date markers."""
    text = DOC_TEMPLATE.read_text(encoding="utf-8")
    slug = _exp_slug(num)
    text = text.replace("EXP-NNN", slug, 1)  # only the H1
    text = text.replace("YYYY-MM-DD", today, 1)  # the first Date created
    text = text.replace("EXP-NNN-<slug>", f"{slug}-{tail}")
    return text


def _render_sweep_template(num: int, tail: str) -> str:
    text = SWEEP_TEMPLATE.read_text(encoding="utf-8")
    slug = _exp_slug(num)
    text = text.replace("EXP-NNN-<slug>.md", f"{slug}-{tail}.md")
    text = text.replace("EXP-NNN", slug)
    return text


def _register_placeholder(slug: str, plan_path: Path, dsn: str) -> bool:
    """Insert a placeholder experiments row (status='planned', no plan_git_sha).

    Returns True on insert, False if the row already exists.
    """
    rel_path = str(plan_path).replace(f"{REPO_ROOT}/", "")
    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM experiments WHERE slug = %s", (slug,))
            if cur.fetchone():
                return False
            cur.execute(
                """
                INSERT INTO experiments
                    (slug, title, status, plan_path, created_at)
                VALUES (%s, %s, 'planned', %s, NOW())
                """,
                (slug, f"{slug} (placeholder)", rel_path),
            )
        return True
    except psycopg.Error as e:
        console.print(f"[yellow]DB unreachable, skipping placeholder row[/]: {e}")
        return False


@app.command()
def main(
    slug: str = typer.Argument(..., help="Short slug tail, e.g. 'retrieval-ablation'"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan, don't write"),
    no_stage: bool = typer.Option(False, "--no-stage", help="Skip git add -N after creating files"),
    pg_dsn: str = typer.Option(
        "postgresql://m@/lab",
        "--pg-dsn",
        envvar="LAB_PG_DSN",
        help="Postgres DSN for the lab DB",
    ),
) -> None:
    """Scaffold a new experiment."""
    tail = _validate_slug(slug)

    # Decide the EXP number.
    num = _next_exp_number(dsn=None if dry_run else pg_dsn)
    exp_slug = _exp_slug(num)
    doc_path = _doc_path(num, tail)
    sweep_path = _sweep_path(num)
    analysis_dir = _analysis_dir(num)
    gitkeep = analysis_dir / ".gitkeep"

    # Idempotence: refuse if the doc already exists.
    if doc_path.exists():
        console.print(
            f"[red]ERROR[/]: {doc_path.relative_to(REPO_ROOT)} already exists — "
            f"refusing to overwrite. (No-op.)"
        )
        raise typer.Exit(code=2)

    if not DOC_TEMPLATE.exists() or not SWEEP_TEMPLATE.exists():
        console.print(
            f"[red]ERROR[/]: missing template(s): "
            f"{DOC_TEMPLATE.relative_to(REPO_ROOT)}, "
            f"{SWEEP_TEMPLATE.relative_to(REPO_ROOT)}"
        )
        raise typer.Exit(code=3)

    from datetime import date

    today = date.today().isoformat()
    doc_text = _render_doc_template(num, tail, today)
    sweep_text = _render_sweep_template(num, tail)

    if dry_run:
        console.print(f"[bold]Would create[/] (slug={exp_slug}):")
        console.print(f"  - {doc_path.relative_to(REPO_ROOT)}")
        console.print(f"  - {sweep_path.relative_to(REPO_ROOT)}")
        console.print(f"  - {gitkeep.relative_to(REPO_ROOT)}")
        console.print("[bold]Would insert[/] DB row:")
        console.print(
            f"  experiments(slug={exp_slug!r}, status='planned', "
            f"plan_path={str(doc_path.relative_to(REPO_ROOT))!r})"
        )
        return

    # Write files
    doc_path.write_text(doc_text, encoding="utf-8")
    sweep_path.write_text(sweep_text, encoding="utf-8")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    gitkeep.touch()

    inserted = _register_placeholder(exp_slug, doc_path, pg_dsn)

    if not no_stage:
        subprocess.run(
            ["git", "add", "-N", str(doc_path), str(sweep_path)],
            cwd=REPO_ROOT,
            check=False,
        )

    console.print(f"[green]Created[/] {exp_slug} scaffolding:")
    console.print(f"  doc:   {doc_path.relative_to(REPO_ROOT)}")
    console.print(f"  sweep: {sweep_path.relative_to(REPO_ROOT)}")
    console.print(f"  out:   {analysis_dir.relative_to(REPO_ROOT)}/")
    if inserted:
        console.print(f"  db:    inserted placeholder row in experiments(slug={exp_slug!r})")
    else:
        console.print("  db:    row already exists or DB unreachable — skipped insert")
    if not no_stage:
        console.print("  git:   intent-to-add staged (git add -N)")

    console.print()
    console.print("[bold]Next steps[/]:")
    console.print(f"  1. Edit {doc_path.relative_to(REPO_ROOT)} — fill in all sections")
    console.print(f"  2. Edit {sweep_path.relative_to(REPO_ROOT)} — pick shape + matrix")
    console.print(
        f"  3. git add {doc_path.relative_to(REPO_ROOT)} {sweep_path.relative_to(REPO_ROOT)} && git commit"
    )
    console.print(f"  4. uv run lab exp register {doc_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(app() or 0)
