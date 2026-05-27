"""Generate / refresh docs/model-cards/<litellm_id>.md from the lab.models table.

One markdown file per `lab.models` row, with a YAML frontmatter block and a
machine-generated body (description, usage, performance, references). Hand-
written content is preserved between HTML comment markers:

    <!-- BEGIN HAND -->
    Whatever you wrote here is not touched.
    <!-- END HAND -->

Sections that may contain hand-edits:
- Description (one-paragraph blurb)
- Known issues (hand-curated list)

Run after a `lab models sync` (or whenever the registry changes).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import psycopg
import typer
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = REPO_ROOT / "docs" / "model-cards"
FINDINGS_DIR = REPO_ROOT / "docs" / "findings"

HAND_BEGIN = "<!-- BEGIN HAND -->"
HAND_END = "<!-- END HAND -->"
GEN_BEGIN = "<!-- BEGIN AUTOGEN -->"
GEN_END = "<!-- END AUTOGEN -->"

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _fetch_models(dsn: str) -> list[dict[str, Any]]:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT model_id, publisher, name, variant, quant, backend,
                   litellm_id, source_url, ollama_tag, vram_gb, context_max,
                   output_max, license, capabilities, notes, pulled_at,
                   retired_at
            FROM models
            ORDER BY litellm_id
            """
        )
        cols = [d.name for d in cur.description] if cur.description else []
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _fetch_usage(dsn: str, model_id: int) -> dict[str, Any]:
    """Top experiment slugs (last 30d) and aggregate perf stats for a model."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        # Top experiments by run count (last 30d)
        cur.execute(
            """
            SELECT e.slug, COUNT(*) AS n
            FROM experiment_runs r
            JOIN experiments e ON e.experiment_id = r.experiment_id
            WHERE r.model_id = %s
              AND r.started_at IS NOT NULL
              AND r.started_at > NOW() - INTERVAL '30 days'
            GROUP BY e.slug
            ORDER BY n DESC
            LIMIT 5
            """,
            (model_id,),
        )
        top_exps = [(r[0], int(r[1])) for r in cur.fetchall()]

        # Aggregate perf across runs (lifetime, where data exists)
        cur.execute(
            """
            SELECT
              AVG(latency_ms)::float                          AS mean_latency_ms,
              AVG(tokens_in)::float                           AS mean_tokens_in,
              AVG(tokens_out)::float                          AS mean_tokens_out,
              AVG(cost_usd)::float                            AS mean_cost_usd,
              COUNT(*)                                        AS n_runs,
              COUNT(*) FILTER (WHERE status='done')           AS n_done,
              COUNT(*) FILTER (WHERE status='error')          AS n_error
            FROM experiment_runs
            WHERE model_id = %s
            """,
            (model_id,),
        )
        perf_row = cur.fetchone()
        cols = [d.name for d in cur.description] if cur.description else []
        perf = dict(zip(cols, perf_row, strict=True)) if perf_row else {}
    return {"top_exps": top_exps, "perf": perf}


def _find_findings_citing(litellm_id: str) -> list[tuple[str, str]]:
    """Scan docs/findings/F-*.md for files that mention this litellm_id verbatim.

    Returns [(F-NNN, title), ...].
    """
    out: list[tuple[str, str]] = []
    if not FINDINGS_DIR.exists():
        return out
    for f in sorted(FINDINGS_DIR.glob("F-*.md")):
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        if litellm_id not in text:
            continue
        # Extract F-NNN from filename and title from H1
        slug = f.stem.split("-", 2)
        f_id = f"{slug[0]}-{slug[1]}" if len(slug) >= 2 else f.stem
        title = ""
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        out.append((f_id, title))
    return out


def _format_frontmatter(m: dict[str, Any], last_used_in: list[str]) -> str:
    pulled_at = m["pulled_at"]
    created = pulled_at.date().isoformat() if isinstance(pulled_at, datetime) else str(pulled_at)

    caps = m.get("capabilities") or []
    caps_inline = "[" + ", ".join(repr(c) for c in caps) + "]"
    last_used_inline = "[" + ", ".join(repr(s) for s in last_used_in) + "]"

    def _yamlval(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, str):
            # Quote if contains special chars
            if any(c in v for c in (":", "#", "@", "'")) or v.strip() != v:
                return '"' + v.replace('"', '\\"') + '"'
            return v
        return str(v)

    today = date.today().isoformat()
    lines = [
        "---",
        f"doc_id: model-{m['litellm_id']}",
        f"title: {_yamlval(m['publisher'])} / {_yamlval(m['name'])}"
        + (f" {_yamlval(m['variant'])}" if m.get("variant") else ""),
        "kind: card",
        f"status: {'retired' if m.get('retired_at') else 'active'}",
        "owner: m",
        f"created: {created}",
        f"last_updated: {today}",
        f"litellm_id: {_yamlval(m['litellm_id'])}",
        f"backend: {_yamlval(m['backend'])}",
        f"publisher: {_yamlval(m['publisher'])}",
        f"vram_gb: {_yamlval(m.get('vram_gb'))}",
        f"context_max: {_yamlval(m.get('context_max'))}",
        f"capabilities: {caps_inline}",
        f"ollama_tag: {_yamlval(m.get('ollama_tag'))}",
        f"source_url: {_yamlval(m.get('source_url'))}",
        f"license: {_yamlval(m.get('license'))}",
        "known_issues: []",
        f"last_used_in: {last_used_inline}",
        "---",
    ]
    return "\n".join(lines) + "\n"


def _autogen_body(
    m: dict[str, Any],
    usage: dict[str, Any],
    findings: list[tuple[str, str]],
) -> str:
    perf = usage.get("perf") or {}
    top_exps = usage.get("top_exps") or []

    def _fmt(v: Any, suffix: str = "", precision: int = 2) -> str:
        if v is None:
            return "n/a"
        try:
            return f"{float(v):.{precision}f}{suffix}"
        except (TypeError, ValueError):
            return str(v)

    n_runs = perf.get("n_runs") or 0
    n_done = perf.get("n_done") or 0
    n_error = perf.get("n_error") or 0

    body = [GEN_BEGIN]
    body.append(f"# {m['publisher']} / {m['name']}")
    if m.get("variant"):
        body[-1] += f" {m['variant']}"
    body.append("")
    body.append(
        f"`litellm_id`: `{m['litellm_id']}` · "
        f"backend: `{m['backend']}` · "
        f"vram_gb: `{m.get('vram_gb') or 'n/a'}` · "
        f"context_max: `{m.get('context_max') or 'n/a'}`"
    )
    body.append("")
    body.append("## Usage")
    body.append("")
    if top_exps:
        body.append(f"Most-used in (last 30d, top {len(top_exps)}):")
        for slug, n in top_exps[:3]:
            body.append(f"- `{slug}` — {n} run(s)")
    else:
        body.append("No `experiment_runs` rows in the last 30 days.")
    body.append("")
    body.append("## Performance (lifetime aggregate)")
    body.append("")
    body.append(f"- runs: {n_runs} (done={n_done}, error={n_error})")
    body.append(f"- mean latency: {_fmt(perf.get('mean_latency_ms'), ' ms', 1)}")
    body.append(f"- mean tokens_in: {_fmt(perf.get('mean_tokens_in'), '', 1)}")
    body.append(f"- mean tokens_out: {_fmt(perf.get('mean_tokens_out'), '', 1)}")
    body.append(f"- mean cost: {_fmt(perf.get('mean_cost_usd'), ' USD', 6)}")
    body.append("")
    body.append("## References")
    body.append("")
    if findings:
        for fid, title in findings:
            body.append(f"- [{fid}](../findings/{fid}-*.md) — {title}")
    else:
        body.append("No findings cite this model yet.")
    body.append("")
    if m.get("source_url"):
        body.append(f"Source: <{m['source_url']}>")
    body.append(GEN_END)
    return "\n".join(body) + "\n"


def _description_section(m: dict[str, Any]) -> str:
    """Default description (preserved between HAND markers)."""
    notes = m.get("notes")
    blurb = notes if notes else "TODO: write a one-paragraph description."
    return f"## Description\n\n{HAND_BEGIN}\n{blurb}\n{HAND_END}\n"


def _known_issues_section() -> str:
    return (
        f"## Known issues\n\n{HAND_BEGIN}\n"
        f"_Hand-curated list. Add entries as they're discovered._\n"
        f"{HAND_END}\n"
    )


def _extract_hand_blocks(text: str) -> dict[str, str]:
    """Extract hand-edited blocks from an existing card, keyed by preceding H2."""
    out: dict[str, str] = {}
    if not text:
        return out
    lines = text.splitlines()
    current_h2: str | None = None
    in_hand = False
    buf: list[str] = []
    for line in lines:
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
    m: dict[str, Any],
    usage: dict[str, Any],
    findings: list[tuple[str, str]],
    hand_blocks: dict[str, str],
) -> str:
    last_used_in = [slug for slug, _ in (usage.get("top_exps") or [])]
    frontmatter = _format_frontmatter(m, last_used_in)
    autogen = _autogen_body(m, usage, findings)

    description = m.get("notes") or "TODO: write a one-paragraph description."
    hand_desc = hand_blocks.get("description")
    if hand_desc:
        description = hand_desc

    known_issues = hand_blocks.get("known issues") or (
        "_Hand-curated list. Add entries as they're discovered._"
    )

    description_section = f"## Description\n\n{HAND_BEGIN}\n{description}\n{HAND_END}\n"
    known_issues_section = f"## Known issues\n\n{HAND_BEGIN}\n{known_issues}\n{HAND_END}\n"

    return frontmatter + "\n" + autogen + "\n" + description_section + "\n" + known_issues_section


@app.command()
def main(
    pg_dsn: str = typer.Option("postgresql://m@/lab", "--pg-dsn", envvar="LAB_PG_DSN"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Sync docs/model-cards/<litellm_id>.md from lab.models."""
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    models = _fetch_models(pg_dsn)
    if not models:
        console.print("[yellow]no models found in lab.models[/]")
        raise typer.Exit(code=1)

    written = 0
    for m in models:
        card_path = CARDS_DIR / f"{m['litellm_id']}.md"
        usage = _fetch_usage(pg_dsn, m["model_id"])
        findings = _find_findings_citing(m["litellm_id"])
        existing = card_path.read_text(encoding="utf-8") if card_path.exists() else ""
        hand_blocks = _extract_hand_blocks(existing)
        new_text = _render_card(m, usage, findings, hand_blocks)

        if dry_run:
            console.print(
                f"[bold]Would write[/] {card_path.relative_to(REPO_ROOT)} "
                f"({len(new_text)} chars)"
            )
        else:
            card_path.write_text(new_text, encoding="utf-8")
            written += 1

    if dry_run:
        console.print(f"[bold]Dry run[/]: {len(models)} card(s) would be written")
    else:
        console.print(
            f"[green]Wrote[/] {written} card(s) under {CARDS_DIR.relative_to(REPO_ROOT)}/"
        )
    _ = UTC  # silence unused-import warning if removed elsewhere


if __name__ == "__main__":
    app()
