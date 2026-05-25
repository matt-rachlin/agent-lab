"""Lab CLI — `uv run lab ...` or `lab ...` after install."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from lab.analyze.report import make_report
from lab.eval import apply_to_experiment, get_registry, load_evaluators_from
from lab.eval.builtin import register_all as register_builtin_evaluators
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
    only: list[str] = typer.Option(
        [], "--only", help="Restrict to named evaluators (repeatable)"
    ),
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
