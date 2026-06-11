"""Generate / refresh tasks/<suite>/CARD.md from the lab.tasks table.

One CARD.md per non-empty suite directory under tasks/. Mirrors the structure
of tools/sync_model_cards.py: preserves hand-edits between
<!-- BEGIN HAND --> and <!-- END HAND --> markers, regenerates everything
between <!-- BEGIN AUTOGEN --> and <!-- END AUTOGEN -->.

Hand-editable sections:
- Purpose (one paragraph)
- Known limitations (hand-curated; see EXP-002 F-005 call-outs about http
  tasks for PBS-Agent-v0.1)

Run after `lab tasks load` (or whenever the task set changes).
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import psycopg
import typer
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "tasks"
FINDINGS_DIR = REPO_ROOT / "docs" / "findings"

HAND_BEGIN = "<!-- BEGIN HAND -->"
HAND_END = "<!-- END HAND -->"
GEN_BEGIN = "<!-- BEGIN AUTOGEN -->"
GEN_END = "<!-- END AUTOGEN -->"

# Map DB suite slug -> directory name on disk.
SUITE_TO_DIR = {
    "smoke": "smoke",
    "PBS-v0.1": "pbs",
    "pbs-agent-v0.1": "pbs-agent-v0.1",
    "pbs-agent-rag-v0.1": "pbs-agent-rag-v0.1",
    "pbs-agent-rag-v0.2": "pbs-agent-rag-v0.2",
    "agent-smoke": "agent-smoke",
}

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _fetch_tasks(dsn: str) -> list[dict[str, Any]]:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT suite, slug, category, difficulty, payload, added_at
            FROM tasks
            WHERE retired_at IS NULL
            ORDER BY suite, slug
            """
        )
        cols = [d.name for d in cur.description] if cur.description else []
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _fetch_experiments_using_suite(dsn: str, suite: str) -> list[str]:
    """Distinct experiment slugs that ran any task from this suite."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT e.slug
            FROM experiment_runs r
            JOIN tasks t ON t.task_id = r.task_id
            JOIN experiments e ON e.experiment_id = r.experiment_id
            WHERE t.suite = %s
            ORDER BY e.slug
            """,
            (suite,),
        )
        return [r[0] for r in cur.fetchall()]


def _find_findings_citing_suite(suite: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not FINDINGS_DIR.exists():
        return out
    needle = suite
    for f in sorted(FINDINGS_DIR.glob("F-*.md")):
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        if needle not in text:
            continue
        slug = f.stem.split("-", 2)
        f_id = f"{slug[0]}-{slug[1]}" if len(slug) >= 2 else f.stem
        title = ""
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        out.append((f_id, title))
    return out


def _yamlval(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, str):
        if any(c in v for c in (":", "#", "@", "'")) or v.strip() != v:
            return '"' + v.replace('"', '\\"') + '"'
        return v
    return str(v)


def _summarize_suite(suite: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate stats for the suite from its task rows."""
    cat_counter: Counter[str] = Counter()
    tools_set: set[str] = set()
    success_pred_types: Counter[str] = Counter()
    rubric_types: Counter[str] = Counter()
    difficulty: Counter[str] = Counter()
    earliest: datetime | None = None
    for t in tasks:
        cat = t.get("category") or "uncategorized"
        cat_counter[cat] += 1
        diff = t.get("difficulty") or "unspecified"
        difficulty[diff] += 1
        payload = t.get("payload") or {}
        # tools list (agent tasks)
        for tool in payload.get("tools") or []:
            name = tool.get("name") if isinstance(tool, dict) else tool
            if name:
                tools_set.add(str(name))
        sp = payload.get("success_predicate")
        if isinstance(sp, dict) and sp.get("type"):
            success_pred_types[str(sp["type"])] += 1
        rb = payload.get("rubric")
        if isinstance(rb, dict) and rb.get("type"):
            rubric_types[str(rb["type"])] += 1
        added = t.get("added_at")
        if isinstance(added, datetime) and (earliest is None or added < earliest):
            earliest = added
    return {
        "categories": dict(cat_counter),
        "tools": sorted(tools_set),
        "success_predicate_types": dict(success_pred_types),
        "rubric_types": dict(rubric_types),
        "difficulty": dict(difficulty),
        "earliest": earliest,
        "task_count": len(tasks),
    }


def _extract_hand_blocks(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not text:
        return out
    current_h2: str | None = None
    in_hand = False
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            current_h2 = line[3:].strip().lower()
            in_hand = False
            buf = []
            continue
        if HAND_BEGIN in line:
            in_hand = True
            buf = []
            continue
        if HAND_END in line:
            if in_hand and current_h2 is not None:
                out[current_h2] = "\n".join(buf).strip()
            in_hand = False
            continue
        if in_hand:
            buf.append(line)
    return out


def _render_card(
    suite: str,
    suite_dir_name: str,
    summary: dict[str, Any],
    experiments: list[str],
    findings: list[tuple[str, str]],
    hand_blocks: dict[str, str],
) -> str:
    earliest = summary["earliest"]
    created = earliest.date().isoformat() if isinstance(earliest, datetime) else "unknown"
    today = date.today().isoformat()
    cats = summary["categories"]
    cats_inline = "[" + ", ".join(repr(c) for c in sorted(cats)) + "]"
    last_used_inline = "[" + ", ".join(repr(s) for s in experiments) + "]"

    fm = [
        "---",
        f"doc_id: task-suite-{suite_dir_name}",
        f"title: {suite} — task suite",
        "kind: card",
        "status: active",
        "owner: m",
        f"created: {created}",
        f"last_updated: {today}",
        f"suite: {_yamlval(suite)}",
        f"task_count: {summary['task_count']}",
        f"categories: {cats_inline}",
        f"last_used_in: {last_used_inline}",
        "---",
    ]
    frontmatter = "\n".join(fm) + "\n"

    # Body — purpose first (hand block), then autogen
    purpose = hand_blocks.get("purpose") or (
        f"TODO: one-paragraph description of what {suite} measures."
    )
    purpose_section = f"## Purpose\n\n{HAND_BEGIN}\n{purpose}\n{HAND_END}\n"

    autogen: list[str] = [GEN_BEGIN, ""]
    autogen.append("## Categories")
    autogen.append("")
    for cat, n in sorted(cats.items()):
        autogen.append(f"- `{cat}` — {n} task(s)")
    autogen.append("")
    autogen.append("## Difficulty distribution")
    autogen.append("")
    for diff, n in sorted(summary["difficulty"].items()):
        autogen.append(f"- {diff}: {n}")
    autogen.append("")
    autogen.append("## Tools used (union across tasks)")
    autogen.append("")
    if summary["tools"]:
        for tool in summary["tools"]:
            autogen.append(f"- `{tool}`")
    else:
        autogen.append("None (text-only suite).")
    autogen.append("")
    autogen.append("## Pre-reg shape")
    autogen.append("")
    sp = summary["success_predicate_types"]
    autogen.append("- success_predicate types:")
    if sp:
        for t, n in sorted(sp.items()):
            autogen.append(f"  - `{t}` — {n} task(s)")
    else:
        autogen.append("  - (none — single-turn tasks rely on rubric only)")
    rb = summary["rubric_types"]
    autogen.append("- rubric types:")
    if rb:
        for t, n in sorted(rb.items()):
            autogen.append(f"  - `{t}` — {n} task(s)")
    else:
        autogen.append("  - (none)")
    autogen.append("")
    autogen.append("## Experiments using this suite")
    autogen.append("")
    if experiments:
        for slug in experiments:
            autogen.append(f"- `{slug}`")
    else:
        autogen.append("None on record (no `experiment_runs` rows reference this suite).")
    autogen.append("")
    autogen.append("## Findings citing this suite")
    autogen.append("")
    if findings:
        for fid, title in findings:
            autogen.append(f"- [{fid}](../../docs/findings/{fid}-*.md) — {title}")
    else:
        autogen.append("No findings yet.")
    autogen.append("")
    autogen.append(GEN_END)
    autogen_section = "\n".join(autogen) + "\n"

    known = hand_blocks.get("known limitations") or (
        "_Hand-curated list. Add limits, gotchas, and known-bad behaviour here._"
    )
    known_section = f"## Known limitations\n\n{HAND_BEGIN}\n{known}\n{HAND_END}\n"

    return frontmatter + "\n" + purpose_section + "\n" + autogen_section + "\n" + known_section


@app.command()
def main(
    pg_dsn: str = typer.Option("postgresql://m@/lab", "--pg-dsn", envvar="LAB_PG_DSN"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Sync tasks/<suite>/CARD.md from lab.tasks."""
    all_tasks = _fetch_tasks(pg_dsn)
    # Bucket by suite
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for t in all_tasks:
        by_suite.setdefault(t["suite"], []).append(t)

    written = 0
    skipped: list[str] = []
    for suite, tasks in sorted(by_suite.items()):
        dir_name = SUITE_TO_DIR.get(suite)
        if not dir_name:
            skipped.append(suite)
            continue
        suite_dir = TASKS_DIR / dir_name
        if not suite_dir.exists():
            skipped.append(f"{suite} (no dir {suite_dir})")
            continue

        summary = _summarize_suite(suite, tasks)
        experiments = _fetch_experiments_using_suite(pg_dsn, suite)
        findings = _find_findings_citing_suite(suite)
        card_path = suite_dir / "CARD.md"
        existing = card_path.read_text(encoding="utf-8") if card_path.exists() else ""
        hand_blocks = _extract_hand_blocks(existing)
        new_text = _render_card(suite, dir_name, summary, experiments, findings, hand_blocks)

        if dry_run:
            console.print(
                f"[bold]Would write[/] {card_path.relative_to(REPO_ROOT)} "
                f"({len(new_text)} chars)"
            )
        else:
            card_path.write_text(new_text, encoding="utf-8")
            written += 1

    if dry_run:
        console.print(f"[bold]Dry run[/]: {len(by_suite) - len(skipped)} card(s) would be written")
    else:
        console.print(
            f"[green]Wrote[/] {written} suite card(s) under {TASKS_DIR.relative_to(REPO_ROOT)}/"
        )
    if skipped:
        console.print(f"[yellow]Skipped suites[/] (no mapping or dir): {skipped}")


if __name__ == "__main__":
    app()
