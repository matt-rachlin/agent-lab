"""Lab CLI — `uv run lab ...` or `lab ...` after install."""

from __future__ import annotations

import sys
from datetime import datetime as _datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from lab.analyze.report import make_report
from lab.daily_log import ensure_today, open_in_editor
from lab.eval import apply_to_experiment, get_registry, load_evaluators_from
from lab.eval.builtin import register_all as register_builtin_evaluators
from lab.experiment import (
    get_experiment,
    is_pre_registered,
    list_experiments,
    register_plan,
    validate_plan,
)
from lab.finding import list_findings, new_finding
from lab.finding import sync as sync_findings
from lab.manifest import capture as capture_manifest
from lab.sweep.config import load_sweep
from lab.sweep.runner import run_sweep
from lab.tasks.registry import list_suites, load_tasks, register_tasks

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

tasks_app = typer.Typer(help="Task suite management")
app.add_typer(tasks_app, name="tasks")


@tasks_app.command("load")
def tasks_load(path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Load a YAML/JSON/JSONL task file into the lab.tasks table."""
    rows = load_tasks(path)
    n = register_tasks(rows)
    console.print(f"[green]registered[/] {n} task(s) from [bold]{path}[/]")


@tasks_app.command("list")
def tasks_list() -> None:
    """List task suites in the lab DB."""
    table = Table("Suite", "Count")
    for suite, count in list_suites():
        table.add_row(suite, str(count))
    console.print(table)


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

sweep_app = typer.Typer(help="Comparison sweeps")
app.add_typer(sweep_app, name="sweep")


@sweep_app.command("run")
def sweep_run(
    config: Path = typer.Argument(..., exists=True, readable=True),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't actually call models"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip already-done runs"),
    enforce_pre_registration: bool = typer.Option(
        False,
        "--enforce-pre-registration",
        help="Refuse to start if the experiment slug is not pre-registered (no plan_git_sha).",
    ),
    key_file: Path = typer.Option(
        Path("/data/lab/services/litellm-master-key"),
        "--key-file",
        help="Path to LiteLLM master key file",
    ),
) -> None:
    """Run a sweep from a YAML config."""
    spec = load_sweep(config)
    if enforce_pre_registration and not is_pre_registered(spec.experiment.slug):
        console.print(
            f"[red]ERROR[/]: experiment {spec.experiment.slug!r} is not pre-registered. "
            f"Run `lab exp register {spec.experiment.plan_path}` first, "
            f"or omit --enforce-pre-registration."
        )
        raise typer.Exit(code=2)
    litellm_key = key_file.read_text().strip()
    summary = run_sweep(spec, litellm_key=litellm_key, resume=resume, dry_run=dry_run)
    console.print(f"[bold green]summary[/]: {summary}")
    if summary.get("errors", 0) > 0:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

analyze_app = typer.Typer(help="Analyze sweep results")
app.add_typer(analyze_app, name="analyze")


@analyze_app.command("report")
def analyze_report(
    experiment: str = typer.Argument(..., help="Experiment slug"),
    out: Path | None = typer.Option(None, "--out", help="Write to file instead of stdout"),
) -> None:
    """Generate a markdown report for an experiment's runs."""
    md = make_report(experiment)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        console.print(f"wrote {len(md):,} bytes to [bold]{out}[/]")
    else:
        sys.stdout.write(md)


# ---------------------------------------------------------------------------
# exp — experiment plan pre-registration
# ---------------------------------------------------------------------------

exp_app = typer.Typer(help="Experiment plans: pre-registration, list, show")
app.add_typer(exp_app, name="exp")


@exp_app.command("validate")
def exp_validate(plan: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Check whether a plan is ready for `lab exp register` without committing it."""
    v = validate_plan(plan)
    table = Table("Field", "Value")
    table.add_row("slug", v.slug)
    table.add_row("title", v.title)
    table.add_row("plan_path", str(v.plan_path))
    table.add_row("git_sha", v.git_sha or "[red](uncommitted)[/]")
    table.add_row("git_dirty", "[red]yes[/]" if v.git_dirty else "no")
    table.add_row(
        "missing_sections",
        "[red]" + ", ".join(v.missing_sections) + "[/]" if v.missing_sections else "(none)",
    )
    table.add_row("ready", "[green]yes[/]" if v.ok else "[red]no[/]")
    console.print(table)
    if not v.ok:
        raise typer.Exit(code=1)


@exp_app.command("register")
def exp_register(
    plan: Path = typer.Argument(..., exists=True, readable=True),
    hypothesis: str = typer.Option("", "--hypothesis", help="One-line hypothesis to store"),
    allow_dirty: bool = typer.Option(False, "--allow-dirty", help="Permit uncommitted plan files"),
    note: str = typer.Option("", "--note", help="Retroactive-registration reason, if any"),
) -> None:
    """Pre-register an experiment plan (records slug + git SHA + timestamp)."""
    try:
        v = register_plan(
            plan,
            hypothesis=hypothesis or None,
            allow_dirty=allow_dirty,
            note=note or None,
        )
    except ValueError as exc:
        console.print(f"[red]registration failed[/]: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]registered[/] {v.slug} (sha={v.git_sha[:12] if v.git_sha else '?'}…) "
        f"from {v.plan_path}"
    )


@exp_app.command("list")
def exp_list() -> None:
    """List all experiments in the lab DB."""
    rows = list_experiments()
    if not rows:
        console.print("(no experiments yet)")
        return
    table = Table("Slug", "Status", "Pre-reg", "Runs", "Plan path")
    for r in rows:
        sha = r.get("plan_git_sha")
        pre = f"[green]{str(sha)[:8]}[/]" if sha else "[red]no[/]"
        table.add_row(
            str(r["slug"]),
            str(r.get("status") or ""),
            pre,
            str(r.get("n_runs") or 0),
            str(r.get("plan_path") or ""),
        )
    console.print(table)


@exp_app.command("show")
def exp_show(slug: str = typer.Argument(..., help="Experiment slug")) -> None:
    """Show one experiment's full record."""
    row = get_experiment(slug)
    if not row:
        console.print(f"[red]not found[/]: {slug}")
        raise typer.Exit(code=1)
    table = Table("Field", "Value")
    for k, v in row.items():
        table.add_row(k, str(v) if v is not None else "")
    console.print(table)


# ---------------------------------------------------------------------------
# finding — registry mirror of docs/findings/
# ---------------------------------------------------------------------------

finding_app = typer.Typer(help="Findings: new, sync, list")
app.add_typer(finding_app, name="finding")


@finding_app.command("new")
def finding_new(
    slug: str = typer.Argument(..., help="F-NNN slug (e.g. F-042)"),
    claim: str = typer.Argument("<one-line claim>", help="Short claim text"),
) -> None:
    """Scaffold a new findings markdown file from the template."""
    try:
        path = new_finding(slug, claim)
    except (ValueError, FileExistsError) as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]created[/] {path}")


@finding_app.command("sync")
def finding_sync() -> None:
    """Walk docs/findings/ and upsert each F-NNN-*.md into the findings table."""
    synced, skipped = sync_findings()
    console.print(f"synced {synced} finding(s); skipped {skipped} unparseable file(s)")


@finding_app.command("list")
def finding_list() -> None:
    """List all findings in the lab DB."""
    rows = list_findings()
    if not rows:
        console.print("(no findings yet)")
        return
    table = Table("Slug", "Confidence", "Source EXP", "Status", "Claim")
    for r in rows:
        table.add_row(
            str(r["slug"]),
            str(r["confidence"]),
            str(r.get("source_exp_slug") or ""),
            str(r["status"]),
            str(r["claim"])[:70],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

eval_app = typer.Typer(help="Evaluator framework")
app.add_typer(eval_app, name="eval")


@eval_app.command("list")
def eval_list(
    extra: Path | None = typer.Option(None, "--from", help="Also load user evaluators from PATH"),
) -> None:
    """List registered evaluators (built-ins + optional user dir)."""
    register_builtin_evaluators()
    if extra:
        new = load_evaluators_from(extra)
        console.print(f"loaded {len(new)} user evaluator(s) from {extra}")
    table = Table("Name", "Version", "Category", "Threshold", "Description")
    for entry in sorted(get_registry().values(), key=lambda e: e.name):
        table.add_row(
            entry.name, entry.version, entry.category, str(entry.threshold), entry.description
        )
    console.print(table)


@eval_app.command("apply")
def eval_apply(
    experiment: str = typer.Argument(..., help="Experiment slug"),
    only: list[str] = typer.Option([], "--only", help="Restrict to named evaluators (repeatable)"),
    extra: Path | None = typer.Option(None, "--from", help="Also load user evaluators from PATH"),
    judge_model: str = typer.Option(
        "gpt-oss-20b-cloud", "--judge", help="LiteLLM model_name for LLM-judge evaluators"
    ),
    no_judge: bool = typer.Option(False, "--no-judge", help="Disable judge (llm_judge evals skip)"),
) -> None:
    """Apply registered evaluators to every done run in an experiment."""
    from lab.eval.judge import make_judge

    register_builtin_evaluators()
    if extra:
        loaded = load_evaluators_from(extra)
        console.print(f"loaded {len(loaded)} user evaluator(s) from {extra}")
    names = list(only) if only else None
    judge = None if no_judge else make_judge(model=judge_model)
    reports = apply_to_experiment(experiment, evaluator_names=names, judge=judge)
    table = Table("Evaluator", "Runs", "Scored", "Skipped", "Passed", "Failed")
    for r in reports:
        table.add_row(
            r.evaluator,
            str(r.n_runs),
            str(r.n_scored),
            str(r.n_skipped),
            f"[green]{r.n_passed}[/]",
            f"[red]{r.n_failed}[/]" if r.n_failed else "0",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# manifest / models — wrappers for parity with `uv run python -m lab.manifest`
# ---------------------------------------------------------------------------


@app.command("manifest")
def manifest_capture(
    extra: str = typer.Option("", "--extra", help="JSON string to merge into manifest.extra"),
) -> None:
    """Capture an environment manifest and persist to lab DB + MinIO."""
    import json

    extra_dict = json.loads(extra) if extra else None
    m = capture_manifest(extra=extra_dict)
    console.print(f"[green]manifest[/] sha={m.sha}")


docs_app = typer.Typer(help="Documentation helpers")
app.add_typer(docs_app, name="docs")


@docs_app.command("recent")
def docs_recent(
    n: int = typer.Option(10, "--n", "-n", help="Number of recent files to list"),
    root: Path = typer.Option(Path("/data/lab/code/docs"), "--root", help="Docs root directory"),
) -> None:
    """List the N most recently modified docs (excluding _templates/)."""
    files: list[tuple[float, Path]] = []
    for p in root.rglob("*.md"):
        if "_templates" in p.parts:
            continue
        if p.name.startswith("_") or p.name == "index.md":
            continue
        files.append((p.stat().st_mtime, p))
    files.sort(reverse=True)
    table = Table("Modified", "Path")
    for mtime, p in files[:n]:
        when = _datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        rel = str(p).replace("/data/lab/code/", "")
        table.add_row(when, rel)
    console.print(table)


@docs_app.command("serve")
def docs_serve(
    port: int = typer.Option(8001, "--port", help="Local port for `mkdocs serve`"),
) -> None:
    """Local-only MkDocs Material dev server (no publishing target — by design)."""
    import subprocess

    console.print(f"[green]serving docs at[/] http://localhost:{port}")
    subprocess.call(["uv", "run", "mkdocs", "serve", "--dev-addr", f"127.0.0.1:{port}"])


@docs_app.command("build")
def docs_build() -> None:
    """Build the docs site to ./site (local-only, not published)."""
    import subprocess

    subprocess.call(["uv", "run", "mkdocs", "build", "--strict"])
    console.print("[green]built[/] /data/lab/code/site")


@app.command("today")
def today_command(
    no_editor: bool = typer.Option(
        False, "--no-editor", help="Don't spawn $EDITOR; just print the path"
    ),
) -> None:
    """Open (or create) today's daily log; pre-fills from yesterday's `## Tomorrow` section."""
    path, created = ensure_today()
    console.print(
        f"[{'green' if created else 'yellow'}]{'created' if created else 'opened'}[/] {path}"
    )
    if no_editor:
        return
    rc = open_in_editor(path)
    if rc != 0:
        raise typer.Exit(code=rc)


@app.command("version")
def version() -> None:
    """Print the lab version."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        console.print(pkg_version("lab"))
    except PackageNotFoundError:
        console.print("0.0.1+local")


if __name__ == "__main__":
    app()
