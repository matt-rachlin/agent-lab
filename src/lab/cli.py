"""Lab CLI — `uv run lab ...` or `lab ...` after install."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from lab.analyze.report import make_report
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
    key_file: Path = typer.Option(
        Path("/data/lab/services/litellm-master-key"),
        "--key-file",
        help="Path to LiteLLM master key file",
    ),
) -> None:
    """Run a sweep from a YAML config."""
    spec = load_sweep(config)
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
