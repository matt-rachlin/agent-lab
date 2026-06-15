"""Lab CLI — `uv run lab ...` or `lab ...` after install."""

from __future__ import annotations

import sys
from datetime import datetime as _datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from lab.analyze.report import make_report
from lab.core.daily_log import ensure_today, open_in_editor
from lab.core.manifest import capture as capture_manifest
from lab.core.notify import get_ntfy_url, notify
from lab.eval import apply_to_experiment, get_registry, load_evaluators_from
from lab.eval.builtin import register_all as register_builtin_evaluators
from lab.experiment import (
    get_experiment,
    is_pre_registered,
    list_experiments,
    register_plan,
    validate_plan,
)
from lab.finding import TrustLevel, backfill_trust, list_findings, new_finding, promote_finding
from lab.finding import sync as sync_findings
from lab.observability.log import configure_logging as _configure_logging
from lab.observability.quota import alert_if_high as quota_alert
from lab.observability.quota import usage_window as quota_window
from lab.observability.spend import backfill as spend_backfill
from lab.observability.tracing import configure_tracing as _configure_tracing
from lab.sweep.config import load_sweep
from lab.sweep.runner import cancel_sweep, get_sweep_status, run_sweep
from lab.tasks.registry import list_suites, load_tasks, register_tasks

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.callback()
def _root(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        envvar="LAB_LOG_LEVEL",
        help="stdlib log level for the structured logger (DEBUG/INFO/WARNING/ERROR).",
    ),
    log_json: bool | None = typer.Option(
        None,
        "--log-json/--no-log-json",
        envvar="LAB_LOG_JSON",
        help="Force JSON-mode logs. Default: auto (JSON off-TTY, console on-TTY).",
    ),
) -> None:
    """Lab CLI entrypoint — wires structured logging + OTel tracing once.

    Idempotent: subcommands that re-enter the lab through the same process
    won't re-wire. The OTel exporter target defaults to
    ``http://localhost:4317`` (Tempo); override with
    ``LAB_OTEL_EXPORTER_URL`` or set it to ``none`` to disable export
    while still keeping span creation in-process.
    """

    _configure_logging(level=log_level, json_mode=log_json)
    _configure_tracing()


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
# data — external benchmark downloads (Phase 17.5)
# ---------------------------------------------------------------------------

data_app = typer.Typer(help="External benchmark data: download + register tasks")
app.add_typer(data_app, name="data")


@data_app.command("add-benchmark")
def data_add_benchmark(
    name: str = typer.Argument(..., help="Benchmark slug (e.g. 'bfcl-v3')"),
    categories: list[str] = typer.Option(
        [],
        "--category",
        help="Restrict to named categories (repeatable; default: all supported)",
    ),
    limit_per_category: int | None = typer.Option(
        None,
        "--limit-per-category",
        help="Cap examples per category (smallest id first; default: full)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Load and report counts but do not register to the DB"
    ),
) -> None:
    """Download + register an external benchmark as a lab task suite.

    Currently supported:

    * ``bfcl-v3`` — Berkeley Function Calling Leaderboard v3 (Python AST
      categories: simple, multiple, parallel, parallel_multiple).
    * ``tau2-bench`` — Sierra τ²-bench (loads the vendored domain tasks into
      a registerable suite; multi-turn dual-control *execution* is a
      deferred runner lane — rows carry the τ² bookkeeping for it).
    * ``harbor`` — Terminal-Bench corpus (loads the vendored task dirs into a
      registerable suite; *execution* is via the Harbor CLI +
      ``lab.agent.harbor_adapter.LabReactAgent``, not ``lab sweep run``).
    """

    if name in {"bfcl-v3", "bfcl"}:
        from lab.eval.external.bfcl import DEFAULT_CATEGORIES, load_bfcl_tasks

        cats = categories if categories else list(DEFAULT_CATEGORIES)
        unknown = [c for c in cats if c not in DEFAULT_CATEGORIES]
        if unknown:
            console.print(
                f"[red]error[/] unknown BFCL categor{'ies' if len(unknown) > 1 else 'y'}: "
                f"{', '.join(unknown)} (known: {', '.join(DEFAULT_CATEGORIES)})"
            )
            raise typer.Exit(code=2)
        console.print(
            f"[bold]bfcl-v3[/] loading {len(cats)} categor{'ies' if len(cats) > 1 else 'y'}: {', '.join(cats)}"
        )
        tasks = load_bfcl_tasks(cats, limit_per_category=limit_per_category)
        by_cat: dict[str, int] = {}
        for t in tasks:
            by_cat[t.category or "?"] = by_cat.get(t.category or "?", 0) + 1
        table = Table("Category", "Count")
        for cat in cats:
            table.add_row(cat, str(by_cat.get(cat, 0)))
        console.print(table)
        console.print(f"[green]loaded[/] {len(tasks)} task(s)")
        if dry_run:
            console.print("[yellow]dry-run[/] — skipping DB write")
            return
        n = register_tasks(tasks)
        console.print(f"[green]registered[/] {n} task(s) into suite 'bfcl-v3-ast'")
        return

    if name in {"tau2-bench", "tau2"}:
        from lab.eval.external.tau2 import DEFAULT_DOMAINS, load_tau2_tasks
        from lab.eval.external.tau2 import SUITE_NAME as TAU2_SUITE

        doms = categories if categories else list(DEFAULT_DOMAINS)
        console.print(f"[bold]tau2-bench[/] loading {len(doms)} domain(s): {', '.join(doms)}")
        try:
            tasks = load_tau2_tasks(doms, limit_per_domain=limit_per_category)
        except FileNotFoundError as exc:
            console.print(f"[red]error[/] {exc}")
            raise typer.Exit(code=2) from exc
        by_dom: dict[str, int] = {}
        for t in tasks:
            by_dom[t.category or "?"] = by_dom.get(t.category or "?", 0) + 1
        table = Table("Domain", "Count")
        for dom in doms:
            table.add_row(dom, str(by_dom.get(dom, 0)))
        console.print(table)
        console.print(f"[green]loaded[/] {len(tasks)} task(s)")
        console.print(
            "[yellow]note[/] τ²-bench execution (user-simulator + reward) is a "
            "deferred runner lane; registration makes the suite selectable now."
        )
        if dry_run:
            console.print("[yellow]dry-run[/] — skipping DB write")
            return
        n = register_tasks(tasks)
        console.print(f"[green]registered[/] {n} task(s) into suite {TAU2_SUITE!r}")
        return

    if name in {"harbor", "terminal-bench"}:
        from lab.eval.external.harbor import SUITE_NAME as HARBOR_SUITE
        from lab.eval.external.harbor import load_harbor_tasks

        console.print("[bold]harbor[/] loading Terminal-Bench task corpus")
        try:
            tasks = load_harbor_tasks(limit=limit_per_category)
        except FileNotFoundError as exc:
            console.print(f"[red]error[/] {exc}")
            raise typer.Exit(code=2) from exc
        console.print(f"[green]loaded[/] {len(tasks)} task(s)")
        console.print(
            "[yellow]note[/] Harbor execution runs via the Harbor CLI + "
            "LabReactAgent, not `lab sweep run`; registration makes the suite "
            "selectable as a cohort manifest."
        )
        if dry_run:
            console.print("[yellow]dry-run[/] — skipping DB write")
            return
        n = register_tasks(tasks)
        console.print(f"[green]registered[/] {n} task(s) into suite {HARBOR_SUITE!r}")
        return

    console.print(f"[red]unknown benchmark[/] {name!r}; known: bfcl-v3, tau2-bench, harbor")
    raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

models_app = typer.Typer(help="Model library management")
app.add_typer(models_app, name="models")


@models_app.command("sync")
def models_sync(
    no_cloud: bool = typer.Option(
        False, "--no-cloud", help="Skip the curated cloud catalog; only sync local Ollama tags."
    ),
) -> None:
    """Refresh `lab.models` from `ollama list` (no pulls)."""
    from lab.models.register import sync_models

    summary = sync_models(include_cloud=not no_cloud)
    console.print(
        f"[green]synced[/] {summary['total']} model(s) "
        f"({summary['local']} local, {summary['cloud']} cloud)"
    )


@models_app.command("show")
def models_show(
    litellm_id: str = typer.Argument(..., help="The canonical litellm_id, e.g. qwen3-30b-a3b-moe"),
) -> None:
    """Show a single model registry row.

    Phase 19a: added so each new model is queryable via
    `lab models show <litellm_id>` per the phase plan.
    """
    import psycopg

    from lab.core.settings import get_settings

    sql = """
    SELECT model_id, publisher, name, variant, quant, backend, litellm_id,
           source_url, ollama_tag, vram_gb, context_max, output_max,
           license, capabilities, notes, pulled_at, retired_at,
           mlflow_model_uri
    FROM models
    WHERE litellm_id = %s
    """
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, (litellm_id,))
        row = cur.fetchone()
        if row is None:
            console.print(f"[red]no models row for litellm_id={litellm_id!r}[/]")
            raise typer.Exit(code=2)
        cols = [d[0] for d in (cur.description or [])]
    table = Table("Field", "Value", show_lines=False)
    for k, v in zip(cols, row, strict=True):
        if v is None:
            disp = "[dim]—[/]"
        elif isinstance(v, list):
            disp = ", ".join(map(str, v)) if v else "[dim](empty)[/]"
        else:
            disp = str(v)
        table.add_row(k, disp)
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
    allow_slow_models: bool = typer.Option(
        False,
        "--allow-slow-models",
        help=(
            "Phase 19e: opt into ceiling-class models tagged `slow_mode` in "
            "lab.models (e.g. llama-3.3-70b-q4 at 6-10 tok/s). Default off."
        ),
    ),
    key_file: Path = typer.Option(
        Path("/data/lab/services/litellm-master-key"),
        "--key-file",
        help="Path to LiteLLM master key file",
    ),
    queue: bool = typer.Option(
        False,
        "--queue",
        help=(
            "Enqueue into the pueue gpu group (labeled with the experiment slug) "
            "instead of running in-process. Serializes GPU work and surfaces the "
            "sweep in the Bridge Queue + Lab tabs with live console output."
        ),
    ),
) -> None:
    """Run a sweep from a YAML config."""
    import shutil
    import subprocess
    import sys

    from lab.sweep.runner import SlowModelGateError

    spec = load_sweep(config)

    if queue:
        # Re-invoke this same command (minus --queue) under pueue's gpu group so
        # it serializes against other GPU work and shows up in the Queue tab. The
        # slug label is what Bridge uses to link the pueue task <-> experiment.
        pueue = shutil.which("pueue") or "/home/m/.local/bin/pueue"
        lab_bin = str(Path(sys.executable).with_name("lab"))
        inner = [lab_bin, "sweep", "run", str(config.resolve())]
        if not resume:
            inner.append("--no-resume")
        if dry_run:
            inner.append("--dry-run")
        if enforce_pre_registration:
            inner.append("--enforce-pre-registration")
        if allow_slow_models:
            inner.append("--allow-slow-models")
        inner += ["--key-file", str(key_file)]
        result = subprocess.run(
            [
                pueue,
                "add",
                "-g",
                "gpu",
                "-w",
                "/data/lab/code",
                "--label",
                spec.experiment.slug,
                "--",
                *inner,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (result.stdout + result.stderr).strip()
        console.print(out or f"enqueued {spec.experiment.slug}")
        raise typer.Exit(code=result.returncode)

    if enforce_pre_registration and not is_pre_registered(spec.experiment.slug):
        console.print(
            f"[red]ERROR[/]: experiment {spec.experiment.slug!r} is not pre-registered. "
            f"Run `lab exp register {spec.experiment.plan_path}` first, "
            f"or omit --enforce-pre-registration."
        )
        raise typer.Exit(code=2)
    litellm_key = key_file.read_text().strip()
    try:
        summary = run_sweep(
            spec,
            litellm_key=litellm_key,
            resume=resume,
            dry_run=dry_run,
            allow_slow_models=allow_slow_models,
        )
    except SlowModelGateError as exc:
        console.print(f"[red]ERROR[/]: {exc}")
        raise typer.Exit(code=2) from exc
    console.print(f"[bold green]summary[/]: {summary}")
    if summary.get("errors", 0) > 0:
        raise typer.Exit(code=1)


@sweep_app.command("status")
def sweep_status() -> None:
    """Show in-flight sweeps: active experiment_runs, GPU lease, running PIDs."""
    s = get_sweep_status()
    if s.sweep_pids:
        pid_table = Table("Slug", "PID")
        for slug, pid in s.sweep_pids:
            pid_table.add_row(slug, str(pid))
        console.print(pid_table)
    else:
        console.print("[dim]no sweep pidfiles[/]")
    holder = s.gpu_lease_holder or "[dim](free)[/]"
    console.print(f"[bold]gpu lease[/]: {holder} (ttl {s.gpu_lease_ttl}s)")
    if s.in_progress:
        table = Table("Run", "Slug", "Model", "Seed", "Started")
        for row in s.in_progress:
            table.add_row(
                str(row["run_id"])[:12],
                str(row["experiment_slug"]),
                str(row["model"]),
                str(row["seed"]),
                str(row["started_at"]),
            )
        console.print(table)
    else:
        console.print("[dim]no runs in_progress[/]")


@sweep_app.command("cancel")
def sweep_cancel(
    slug: str = typer.Argument(..., help="Experiment slug whose sweep to cancel"),
    release_lease: bool = typer.Option(
        True, "--release-lease/--no-release-lease", help="Also force-release the GPU lease."
    ),
) -> None:
    """Signal the running sweep (SIGTERM) and release the GPU lease."""
    result = cancel_sweep(slug, release_lease=release_lease)
    if result["signaled"] is None:
        console.print(f"[yellow]no active sweep pid found for[/] {slug}")
    else:
        console.print(f"[green]signaled[/] pid {result['signaled']} for {slug}")
    if result["released_lease"]:
        console.print("[green]released[/] GPU lease")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

analyze_app = typer.Typer(help="Analyze sweep results")
app.add_typer(analyze_app, name="analyze")


@analyze_app.command("scoreboard")
def analyze_scoreboard(
    require_finding_trust: str | None = typer.Option(
        None,
        "--require-finding-trust",
        help=(
            "Stricter gate: only include entries whose associated finding doc is at or "
            "above this trust level. Valid values: verified, reliability_confirmed, "
            "deployable. Default: off (run trust_level='verified' is the only gate)."
        ),
    ),
) -> None:
    """Multi-axis scoreboard over verified results (ADR-009): capability/
    reliability/safety gate, safety veto, cost reported. Sparse until baselines.

    By default, gates on run trust_level='verified'. Pass --require-finding-trust
    to additionally gate on the finding-doc trust_level (a stricter requirement).
    """
    from lab.analyze.scoreboard import render_scoreboard
    from lab.finding import TRUST_RUNGS

    finding_trust_level: TrustLevel | None = None
    if require_finding_trust is not None:
        valid_levels = set(TRUST_RUNGS) - {"unverified"}
        if require_finding_trust not in valid_levels:
            console.print(
                f"[red]invalid finding trust level[/] {require_finding_trust!r}. "
                f"Valid: {', '.join(sorted(valid_levels))}"
            )
            raise typer.Exit(code=2)
        finding_trust_level = require_finding_trust

    sys.stdout.write(render_scoreboard(require_finding_trust=finding_trust_level))


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
# scout — research-scout recommendation queue (ADR-010)
# ---------------------------------------------------------------------------

scout_app = typer.Typer(help="Research scout: context + recommendation queue")
app.add_typer(scout_app, name="scout")


@scout_app.command("context")
def scout_context_cmd() -> None:
    """Emit the scout's grounding (charter + profile + doc titles + dedup list)."""
    from lab.scout import context_bundle

    sys.stdout.write(context_bundle())


@scout_app.command("add")
def scout_add_cmd(
    source_url: str = typer.Argument(..., help="Source URL (dedup key)"),
    title: str = typer.Option(..., "--title"),
    category: str = typer.Option(
        ..., "--category", help="model|architecture|software|paper|method|benchmark"
    ),
    why: str = typer.Option(..., "--why", help="why relevant to us"),
    confidence: str = typer.Option("medium", "--confidence", help="low|medium|high"),
) -> None:
    """Add a cited recommendation (deduped on source_url)."""
    from lab.scout import add_recommendation

    result = add_recommendation(
        source_url=source_url,
        title=title,
        category=category,
        why_relevant=why,
        confidence=confidence,
    )
    console.print(f"[{'green' if result == 'added' else 'yellow'}]{result}[/] {source_url}")


@scout_app.command("scan")
def scout_scan_cmd(
    focus: str = typer.Argument(..., help="What to scan for"),
    model: str = typer.Option("qwen3-4b-ft-toolcall-q4-latest", "--model"),
    max_recs: int = typer.Option(6, "--max-recs"),
    max_tool_calls: int = typer.Option(24, "--max-tool-calls"),
    num_ctx: int | None = typer.Option(
        None, "--num-ctx", help="Ollama context window; raise for reasoning drivers (gpt-oss)"
    ),
) -> None:
    """Autonomous scout scan (ADR-011): the model searches sources + logs cited recs."""
    from lab.scout_scan import run_scan

    out = run_scan(
        focus=focus,
        model=model,
        max_recs=max_recs,
        max_tool_calls=max_tool_calls,
        num_ctx=num_ctx,
    )
    console.print(f"[green]scan done[/]: {out}")


@scout_app.command("list")
def scout_list_cmd(
    status: str = typer.Option(None, "--status", help="new|triaged|actioned|rejected"),
) -> None:
    """List the recommendation queue."""
    from lab.scout import list_recommendations

    rows = list_recommendations(status)
    if not rows:
        console.print("(no recommendations)")
        return
    table = Table("status", "conf", "category", "title", "source")
    for r in rows:
        table.add_row(
            str(r["status"]),
            str(r["confidence"]),
            str(r["category"]),
            str(r["title"])[:48],
            str(r["source_url"])[:48],
        )
    console.print(table)


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
    sha_short = v.git_sha[:12] if v.git_sha else "?"
    console.print(f"[green]registered[/] {v.slug} at {sha_short}… from {v.plan_path}")
    console.print("commit to seal pre-registration.")


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

finding_app = typer.Typer(help="Findings: new, sync, list, promote")
app.add_typer(finding_app, name="finding")

# Trust-level color map for TTY output.
_TRUST_COLORS: dict[str, str] = {
    "unverified": "yellow",
    "verified": "green",
    "reliability_confirmed": "bright_green",
    "deployable": "cyan",
    "retracted": "red",
}


@finding_app.command("new")
def finding_new(
    slug: str = typer.Argument(..., help="F-NNN slug (e.g. F-042)"),
    claim: str = typer.Argument("<one-line claim>", help="Short claim text"),
    run: str = typer.Option(
        None, "--run", help="source run_id; must be a 'verified' run (ADR-008)"
    ),
) -> None:
    """Scaffold a new findings markdown file from the template."""
    try:
        path = new_finding(slug, claim, source_run_id=run)
    except (ValueError, FileExistsError) as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(code=1) from exc
    if run is None:
        console.print(
            "[yellow]warning:[/] no --run given; this finding is unlinked/exploratory "
            "and is not eligible for verified promotion (ADR-008)."
        )
    console.print(f"[green]created[/] {path}")


@finding_app.command("sync")
def finding_sync() -> None:
    """Walk docs/findings/ and upsert each F-NNN-*.md into the findings table."""
    synced, skipped = sync_findings()
    console.print(f"synced {synced} finding(s); skipped {skipped} unparseable file(s)")


@finding_app.command("list")
def finding_list() -> None:
    """List all findings in the lab DB with trust_level column."""
    rows = list_findings()
    if not rows:
        console.print("(no findings yet)")
        return
    tty = console.is_terminal
    table = Table("Slug", "Trust", "Confidence", "Source EXP", "Status", "Claim")
    for r in rows:
        trust = str(r.get("trust_level") or "unverified")
        if tty:
            color = _TRUST_COLORS.get(trust, "white")
            trust_cell = f"[{color}]{trust}[/{color}]"
        else:
            trust_cell = trust
        table.add_row(
            str(r["slug"]),
            trust_cell,
            str(r["confidence"]),
            str(r.get("source_exp_slug") or ""),
            str(r["status"]),
            str(r["claim"])[:70],
        )
    console.print(table)


@finding_app.command("promote")
def finding_promote(
    slug: str = typer.Argument(..., help="Finding slug, e.g. F-005"),
    level: str = typer.Argument(..., help="Target trust level"),
    force: bool = typer.Option(False, "--force", help="Allow skipping rungs (use with caution)"),
) -> None:
    """Promote a finding to the next trust level (ADR-008).

    Valid levels: unverified -> verified -> reliability_confirmed -> deployable.
    retracted is reachable from any rung (terminal).

    Refuses to skip rungs without --force. Requires depends_on evidence link
    in frontmatter for levels beyond unverified.
    """
    valid: set[str] = {"unverified", "verified", "reliability_confirmed", "deployable", "retracted"}
    if level not in valid:
        console.print(f"[red]unknown level[/] {level!r}. Valid: {', '.join(sorted(valid))}")
        raise typer.Exit(code=2)
    target: TrustLevel = level  # type: ignore[assignment]
    try:
        path = promote_finding(slug, target, force=force)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]promote failed[/]: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]promoted[/] {slug} -> {level} in {path}")


@finding_app.command("backfill-trust")
def finding_backfill_trust() -> None:
    """Set trust_level: unverified on all finding docs that lack the field."""
    updated, already_set = backfill_trust()
    console.print(f"backfill complete: {updated} updated, {already_set} already had trust_level")


# ---------------------------------------------------------------------------
# quota — Ollama Cloud usage estimate
# ---------------------------------------------------------------------------

quota_app = typer.Typer(help="Ollama Cloud quota tracker (rough estimate)")
app.add_typer(quota_app, name="quota")


@quota_app.command("status")
def quota_status(
    tier: str = typer.Option("pro", "--tier", help="free | pro | max"),
    window_hours: int = typer.Option(
        168, "--window", help="Rolling window hours (default 168 = 7d)"
    ),
) -> None:
    """Show estimated cloud usage over a rolling window."""
    u = quota_window(tier=tier, window_hours=window_hours)  # type: ignore[arg-type]
    table = Table("Field", "Value")
    table.add_row("tier", u.tier)
    table.add_row("window_hours", str(u.window_hours))
    table.add_row("runs", str(u.runs))
    table.add_row("tokens_in", f"{u.tokens_in:,}")
    table.add_row("tokens_out", f"{u.tokens_out:,}")
    table.add_row("weighted_units", str(u.weighted_units))
    table.add_row("budget", str(u.budget))
    color = "red" if u.pct_consumed >= 95 else "yellow" if u.pct_consumed >= 80 else "green"
    table.add_row("pct_consumed", f"[{color}]{u.pct_consumed:.1f}%[/]")
    console.print(table)


@quota_app.command("backfill-cost")
def quota_backfill_cost(
    limit: int = typer.Option(1000, "--limit", help="Max runs to update in one pass"),
) -> None:
    """Backfill experiment_runs.cost_usd from the LiteLLM proxy spend ledger."""
    report = spend_backfill(limit=limit)
    table = Table("Field", "Value")
    table.add_row("runs_examined", str(report.runs_examined))
    table.add_row("spends_found", str(report.spends_found))
    table.add_row("runs_updated", str(report.runs_updated))
    table.add_row("total_cost_usd", f"${report.total_cost_usd:.6f}")
    console.print(table)


@quota_app.command("check")
def quota_check(
    tier: str = typer.Option("pro", "--tier", help="free | pro | max"),
    threshold: float = typer.Option(80.0, "--threshold"),
    window_hours: int = typer.Option(168, "--window"),
) -> None:
    """Compute usage and send an ntfy alert if above threshold. For cron use."""
    u = quota_alert(threshold_pct=threshold, tier=tier, window_hours=window_hours)  # type: ignore[arg-type]
    console.print(f"{u.pct_consumed:.1f}% of budget (threshold {threshold:.0f}%)")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

eval_app = typer.Typer(help="Evaluator framework")
app.add_typer(eval_app, name="eval")

# Phase 16.4 follow-up: prompts subgroup wires the lab.eval.prompts +
# lab.eval.prompt_tests modules into the CLI. Lives in eval_cli.py to
# keep this module from growing past its current ~1.2k lines.
from lab.eval_cli import prompts_app as _prompts_app  # noqa: E402

eval_app.add_typer(_prompts_app, name="prompts")


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


# ---------------------------------------------------------------------------
# agent — Phase 6 agent harness (stubs; bodies in 6b/6c/6d)
# ---------------------------------------------------------------------------

agent_app = typer.Typer(help="Agent harness (Phase 6 — stubs)")
app.add_typer(agent_app, name="agent")

agent_tools_app = typer.Typer(help="Agent tool servers")
agent_app.add_typer(agent_tools_app, name="tools")

agent_sandbox_app = typer.Typer(help="Agent sandbox image")
agent_app.add_typer(agent_sandbox_app, name="sandbox")


@agent_app.command("run")
def agent_run(
    task: str = typer.Option(..., "--task", help="Task slug to run"),
    model: str = typer.Option(..., "--model", help="LiteLLM model id"),
    suite: str = typer.Option(
        "agent",
        "--suite",
        help="Task suite to load from (e.g. pbs-agent-v0.1, smoke)",
    ),
    max_turns: int | None = typer.Option(None, "--max-turns", help="Override the task's max_turns"),
    tool_budget: int | None = typer.Option(
        None, "--tool-budget", help="Override the task's tool_budget"
    ),
    temperature: float = typer.Option(0.0, "--temperature", help="Sampling temperature"),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Max response tokens"),
    no_persist: bool = typer.Option(
        False, "--no-persist", help="Skip MinIO/Postgres write — useful for local debugging"
    ),
    pipeline_plan_only: bool = typer.Option(
        False,
        "--pipeline-plan-only",
        help=(
            "Print the lab.platform.model_pool PipelineModelPlan JSON that "
            "would be declared for this task+model (Phase 19c) and exit "
            "without loading anything."
        ),
    ),
    allow_slow_models: bool = typer.Option(
        False,
        "--allow-slow-models",
        help=(
            "Phase 19e: opt into ceiling-class models tagged `slow_mode` in "
            "lab.models (e.g. llama-3.3-70b-q4 at 6-10 tok/s). Default off; "
            "agent_run refuses to dispatch without this flag."
        ),
    ),
) -> None:
    """Run a single agent cell end-to-end via the Inspect harness.

    Loads the task from the registry, opens a Podman+gVisor sandbox, runs
    the multi-turn agent loop, mirrors the result into MinIO + Postgres, and
    prints a Rich summary.
    """
    import json as _json
    import uuid as _uuid

    from inspect_ai import eval as inspect_eval

    from lab.agent.sandbox import Sandbox
    from lab.inspect_bridge.adapter import lab_task_to_inspect
    from lab.inspect_bridge.logwriter import SweepContext, write_run_from_inspect_log
    from lab.tasks.registry import Task as LabTask
    from lab.tasks.registry import get_tasks

    # Phase 19e — slow-model gate. Reuses the same DB-driven capability
    # check as the sweep runner so a single registration controls both
    # entry points. `--allow-slow-models` opts in explicitly.
    if not allow_slow_models:
        from lab.sweep.runner import _slow_models_in

        slow = _slow_models_in([model])
        if slow:
            console.print(
                f"[red]ERROR[/]: model {model!r} is tagged `slow_mode` in lab.models "
                "(ceiling-class, 6-10 tok/s). Pass --allow-slow-models to opt in."
            )
            raise typer.Exit(code=2)

    rows = get_tasks(suite, [task])
    if not rows:
        console.print(f"[red]task {task!r} not found in suite {suite!r}")
        raise typer.Exit(code=2)
    row = rows[0]
    payload = row["payload"]
    if isinstance(payload, str):
        payload = _json.loads(payload)
    lab_task = LabTask.model_validate(
        {
            "suite": row["suite"],
            "slug": row["slug"],
            "category": row.get("category"),
            "difficulty": row.get("difficulty"),
            "input": payload["input"],
            "system": payload.get("system"),
            # Phase 16.4: tasks can reference a prompt by id; resolved at
            # adapter-build time via PromptRegistry.
            "system_prompt_id": payload.get("system_prompt_id"),
            "tools": payload.get("tools"),
            "max_turns": max_turns if max_turns is not None else payload.get("max_turns", 1),
            "tool_budget": tool_budget
            if tool_budget is not None
            else payload.get("tool_budget", 0),
            "success_predicate": payload.get("success_predicate"),
            "sandbox": payload.get("sandbox"),
            "gold_answer": payload.get("gold_answer"),
            "rubric": payload.get("rubric"),
            "description": payload.get("description"),
        }
    )

    # Phase 19c — `--pipeline-plan-only` short-circuits before any sandbox
    # or model load. Prints the JSON plan to stdout so an operator (or a
    # debugging script) can confirm what the model_pool would pre-flight
    # without paying any actual cost.
    #
    # The `is True` check is deliberate: when this function is invoked
    # directly from Python (as the unit tests do), Typer's default
    # OptionInfo object is truthy and would erroneously trigger the
    # short-circuit. Strict-True keeps tests that pass nothing here
    # working unchanged.
    if pipeline_plan_only is True:
        from lab.platform.model_pool import plan_for_cell

        plan = plan_for_cell(
            pipeline_id=f"adhoc-{lab_task.slug}",
            model_id=model,
            tools=lab_task.tools,
        )
        console.print_json(plan.model_dump_json())
        return

    sandbox_cfg = lab_task.sandbox or {}
    network = sandbox_cfg.get("network", "none")
    env = dict(sandbox_cfg.get("env", {}))
    workspace_files_raw = sandbox_cfg.get("workspace_files") or {}
    workspace_files = {
        k: v.encode("utf-8") if isinstance(v, str) else v for k, v in workspace_files_raw.items()
    }

    from lab.agent.tools import task_needs_hf_cache_mount, task_needs_kb_mount
    from lab.core.settings import get_settings as _get_settings_kb

    kb_root_mount: Path | None = None
    if task_needs_kb_mount(lab_task.tools):
        kb_root_mount = _get_settings_kb().kb_root
        env.setdefault("LAB_KB_ROOT", "/kb")
        # The kb_query MCP tool needs to embed the query via Ollama, which
        # lives on the host. Inside the sandbox `localhost` is the container,
        # so point the vendored embedder at the podman-provided alias for
        # the host. Allow callers to override (e.g. tests) by setting
        # OLLAMA_HOST in task.sandbox.env beforehand. See `lab.rag.embedder`.
        env.setdefault("OLLAMA_HOST", "http://host.containers.internal:11434")
        # Also force the sandbox into bridge-network mode with
        # host.containers.internal as an allow-listed host. Podman injects
        # the /etc/hosts entry automatically; sandbox.py knows to skip the
        # `_resolve_host_ipv4` step for this magic name. Without this the
        # container has no network and can never reach Ollama.
        if network == "none":
            network = ["host.containers.internal"]
        elif isinstance(network, list) and "host.containers.internal" not in network:
            network = [*network, "host.containers.internal"]

    # Phase 7 reranker: share the HF cache across cells so the ~1.5 GB
    # cross-encoder weights download exactly once. Only mounted when the
    # task can actually trigger the reranker (kb_query in tools AND
    # LAB_RAG_RERANKER != "none"). The setting carries the host directory;
    # we create it on first use so a clean clone JIT-bootstraps. We force
    # ``HF_HUB_OFFLINE=1`` because the sandbox network is locked to
    # ``host.containers.internal`` only (no huggingface.co reachability) —
    # the cache MUST be warm host-side first, or the reranker silently
    # falls back to stage-1.
    hf_cache_mount: Path | None = None
    if task_needs_hf_cache_mount(lab_task.tools):
        hf_cache_root = _get_settings_kb().hf_cache_root
        hf_cache_root.mkdir(parents=True, exist_ok=True)
        hf_cache_mount = hf_cache_root
        env.setdefault("HF_HOME", "/hf-cache")
        env.setdefault("TRANSFORMERS_CACHE", "/hf-cache/transformers")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Propagate the host's LAB_RAG_RERANKER (if any) into the sandbox so
    # the reranker can be disabled / pinned globally without per-task env
    # surgery. Done unconditionally — the value is harmless when the tool
    # never fires.
    import os as _os

    _host_reranker = _os.environ.get("LAB_RAG_RERANKER")
    if _host_reranker is not None:
        env.setdefault("LAB_RAG_RERANKER", _host_reranker)

    # Phase 7.1: point the in-sandbox LabReranker at the host-side HTTP
    # service when kb_query is in play AND the reranker isn't disabled.
    # The sandbox image no longer carries sentence-transformers/torch, so
    # without this URL the reranker silently falls through to pass-through.
    # ``host.containers.internal`` is already in the network allow-list
    # above (the kb_query branch added it for Ollama); the rerank server
    # binds 127.0.0.1 host-side but podman maps it on the same alias.
    if task_needs_hf_cache_mount(lab_task.tools, reranker_env=env.get("LAB_RAG_RERANKER")):
        _rerank_port = _os.environ.get("LAB_RAG_RERANKER_PORT", "8401")
        env.setdefault(
            "LAB_RAG_RERANKER_URL",
            f"http://host.containers.internal:{_rerank_port}",
        )

    run_id_ = f"adhoc-{lab_task.slug}-{_uuid.uuid4().hex[:8]}"
    console.print(
        f"[bold]agent run[/]: task={lab_task.slug} model={model} "
        f"max_turns={lab_task.max_turns} tool_budget={lab_task.tool_budget}"
    )

    import shutil
    import tempfile

    parent_dir = tempfile.mkdtemp(prefix="lab-inspect-parent-")
    log_dir = str(Path(parent_dir) / "inspect")
    try:
        with Sandbox(
            network=network,
            env=env,
            workspace_files=workspace_files,
            kb_root_mount=kb_root_mount,
            hf_cache_mount=hf_cache_mount,
            hf_cache_target="/hf-cache",
        ) as sandbox:
            inspect_task = lab_task_to_inspect(
                lab_task,
                model=model,
                sandbox=sandbox,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logs = inspect_eval(
                inspect_task,
                display="plain",
                log_samples=True,
                log_dir=log_dir,
                log_format="json",
                log_realtime=False,
            )
    finally:
        shutil.rmtree(parent_dir, ignore_errors=True)
    if not logs:
        console.print("[red]inspect_ai.eval returned no logs")
        raise typer.Exit(code=1)
    log = logs[0]
    samples = getattr(log, "samples", None) or []
    if not samples:
        console.print("[red]inspect log has no samples")
        raise typer.Exit(code=1)
    sample = samples[0]
    lab_agent = (sample.metadata or {}).get("lab_agent") or {}

    trace_uri = None
    if not no_persist:
        import psycopg as _psycopg

        from lab.core.settings import get_settings as _get_settings

        with _psycopg.connect(_get_settings().pg_dsn) as pg, pg.cursor() as cur:
            cur.execute(
                "SELECT model_id FROM models WHERE litellm_id = %s LIMIT 1",
                (model,),
            )
            mrow = cur.fetchone()
            cur.execute("SELECT manifest_sha FROM manifests ORDER BY captured_at DESC LIMIT 1")
            manr = cur.fetchone()
        if mrow is None:
            console.print(f"[yellow]no models row for litellm_id={model!r} — skipping persist")
        else:
            ctx = SweepContext(
                run_id=run_id_,
                experiment_id=None,
                experiment_slug="adhoc",
                model_id=int(mrow[0]),
                model_litellm_id=model,
                task_id=int(row["task_id"]),
                task_slug=lab_task.slug,
                config_hash="adhoc",
                config={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                seed=0,
                manifest_sha=str(manr[0]) if manr is not None else "adhoc",
            )
            try:
                trace_uri = write_run_from_inspect_log(log, ctx)
            except Exception as exc:
                console.print(f"[yellow]logwriter failed: {exc} — continuing with summary only")

    summary = Table(title=f"agent run — {lab_task.slug}")
    summary.add_column("metric")
    summary.add_column("value")
    summary.add_row("turns used", str(lab_agent.get("actual_turns")))
    summary.add_row("tool calls", str(lab_agent.get("tool_call_count")))
    summary.add_row("terminated", str(lab_agent.get("terminated_reason")))
    summary.add_row("total latency (ms)", str(lab_agent.get("total_latency_ms")))
    summary.add_row("error", str(lab_agent.get("error") or "—"))
    # Show every scorer the adapter ran. Inspect stores them keyed by
    # registered scorer name.
    for scorer_name, scored in (sample.scores or {}).items():
        summary.add_row(f"scorer:{scorer_name}", str(scored.value))
    if trace_uri:
        summary.add_row("trajectory", trace_uri)
    console.print(summary)


@agent_tools_app.command("list")
def agent_tools_list() -> None:
    """List registered agent tools with their parameter schemas.

    Schemas come from the FastMCP servers themselves — never hand-coded — so
    this is the authoritative view of the tool surface the agent will see.
    """
    from lab.inspect_bridge.tools import discover_tool_schemas

    schemas = discover_tool_schemas()
    table = Table("Tool", "Required", "Optional", "Description")
    for name in sorted(schemas):
        s = schemas[name]
        props = s.input_schema.get("properties", {})
        required = set(s.input_schema.get("required", []))
        req_str = ", ".join(sorted(p for p in props if p in required))
        opt_str = ", ".join(sorted(p for p in props if p not in required))
        desc = (s.description.splitlines()[0] if s.description else "").strip()
        table.add_row(name, req_str or "(none)", opt_str or "(none)", desc[:60])
    console.print(table)


@agent_tools_app.command("test")
def agent_tools_test(
    name: str = typer.Argument(..., help="Tool name to smoke-test"),
) -> None:
    """Run a tool's smoke test end-to-end inside the sandbox.

    Each tool has a hard-coded minimal-effort call that exercises the
    happy path (e.g. `fs_write` writes a file, `fs_read` reads it back).
    Useful for validating that a freshly-built sandbox image is wired up
    correctly without spinning up a whole eval.
    """
    from lab.agent.sandbox import Sandbox, gvisor_available
    from lab.agent.tools import TOOL_SERVERS
    from lab.inspect_bridge.tools import _invoke_tool_via_sandbox_sync

    if name not in TOOL_SERVERS:
        console.print(f"[red]unknown tool[/]: {name} (known: {sorted(TOOL_SERVERS)})")
        raise typer.Exit(code=2)
    if not gvisor_available():
        console.print("[red]gVisor not available — run `just sandbox-build` and install runsc[/]")
        raise typer.Exit(code=2)
    smoke_args: dict[str, dict[str, object]] = {
        "fs_write": {"path": "smoke.txt", "content": "hello\n", "mode": "overwrite"},
        "fs_read": {"path": "smoke.txt"},
        "fs_grep": {"pattern": "hello", "path": "."},
        "shell_exec": {"command": "echo ok"},
        "http_fetch": {"url": "https://example.com/"},
        "python_eval": {"code": "print(2+2)"},
        "kb_query": {"kb_name": "bash", "question": "how do I redirect stderr to stdout", "k": 3},
    }
    args = smoke_args[name]
    network: str | list[str] = "none"
    env: dict[str, str] = {}
    workspace_files: dict[str, bytes] = {}
    kb_root_mount: Path | None = None
    if name == "http_fetch":
        network = ["example.com"]
        env = {"LAB_HTTP_ALLOWLIST": "example.com"}
    elif name in {"fs_read", "fs_grep"}:
        # Stage a file so the read/grep have something to look at.
        workspace_files = {"smoke.txt": b"hello smoke\n"}
    elif name == "kb_query":
        # Mount the lab KB root read-only. The smoke test runs against the
        # `bash` KB, which is `enrichment_pending` with 0 indexed chunks, so
        # `kb_query` short-circuits to the empty-KB path before touching
        # Ollama — no network needed.
        from lab.core.settings import get_settings as _get_settings_kb

        kb_root_mount = _get_settings_kb().kb_root
        env["LAB_KB_ROOT"] = "/kb"
    with Sandbox(
        network=network,
        env=env,
        workspace_files=workspace_files,
        kb_root_mount=kb_root_mount,
    ) as sb:
        try:
            result = _invoke_tool_via_sandbox_sync(sb, TOOL_SERVERS[name], name, args)
        except Exception as exc:
            console.print(f"[red]tool {name} failed[/]: {exc}")
            raise typer.Exit(code=1) from exc
    import json as _json

    console.print(f"[green]{name} OK[/]")
    console.print(_json.dumps(result, indent=2, default=str))


@agent_sandbox_app.command("build")
def agent_sandbox_build(
    image: str = typer.Option("lab-agent-sandbox:0.1", "--image", help="Image tag to build"),
    containerfile: Path = typer.Option(
        Path("containers/Containerfile.agent-sandbox"),
        "--containerfile",
        help="Containerfile path (relative to repo root)",
    ),
    context: Path = typer.Option(Path("."), "--context", help="Build context directory"),
    digest_out: Path = typer.Option(
        Path("conf/sandbox-image.sha"),
        "--digest-out",
        help="Where to write the resulting image digest",
    ),
) -> None:
    """Build the agent sandbox image and record its digest.

    Calls `podman build` and writes the resulting image ID to
    `conf/sandbox-image.sha` so `experiment_runs.sandbox_image_hash` can be
    populated from the same source of truth.
    """
    import subprocess

    build_cmd = [
        "podman",
        "build",
        "-t",
        image,
        "-f",
        str(containerfile),
        str(context),
    ]
    console.print(f"[dim]$ {' '.join(build_cmd)}[/]")
    try:
        subprocess.run(build_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]podman build failed (exit {exc.returncode})[/]")
        raise typer.Exit(code=exc.returncode) from exc
    inspect = subprocess.run(
        ["podman", "image", "inspect", image, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = inspect.stdout.strip()
    digest_out.parent.mkdir(parents=True, exist_ok=True)
    digest_out.write_text(digest + "\n", encoding="utf-8")
    console.print(f"[green]built[/] {image} -> [bold]{digest}[/]")
    console.print(f"[dim]digest written to {digest_out}[/]")


@app.command("notify")
def notify_command(
    message: str = typer.Argument(..., help="Notification body"),
    title: str = typer.Option("lab", "--title", "-T", help="Notification title"),
    priority: str = typer.Option("default", "--priority", "-p", help="min|low|default|high|max"),
    tags: list[str] = typer.Option([], "--tag", help="ntfy tag (repeatable)"),
) -> None:
    """Send a notification (ntfy.sh + best-effort notify-send)."""
    url = get_ntfy_url()
    if url:
        console.print(f"[dim]ntfy → {url}[/]")
    ok = notify(
        message,
        title=title,
        priority=priority,  # type: ignore[arg-type]
        tags=tags or None,
    )
    if not ok:
        console.print("[yellow]no notification channel reachable[/]")
        raise typer.Exit(code=1)


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


# ---------------------------------------------------------------------------
# kb — vendored kb-builder (Phase 6h-a). Query/inspect; building stays out of
# 6h-a scope (the bash KB is built externally and consumed read-only here).
# ---------------------------------------------------------------------------

kb_app = typer.Typer(help="RAG knowledge bases (lab.rag)")
app.add_typer(kb_app, name="kb")


def _kb_root_path() -> Path:
    from lab.core.settings import get_settings

    return Path(get_settings().kb_root).expanduser()


def _kb_dir(name: str) -> Path:
    return _kb_root_path() / name


@kb_app.command("list")
def kb_list() -> None:
    """List KBs under LAB_KB_ROOT (reads each manifest.yaml)."""
    from lab.rag.manifest import load_manifest

    root = _kb_root_path()
    if not root.exists():
        console.print(f"[yellow]no KB root at[/] {root}")
        return
    rows: list[tuple[str, str, str, int]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        mf = child / "manifest.yaml"
        if not mf.exists():
            continue
        try:
            m = load_manifest(mf)
        except Exception as e:
            rows.append((child.name, f"[red]load-failed[/] {e}", "-", 0))
            continue
        rows.append((m.name, m.status, m.models.embedding.name, len(m.sources)))
    table = Table("Name", "Status", "Embed model", "Sources")
    for name, status, model, n_sources in rows:
        table.add_row(name, status, model, str(n_sources))
    console.print(table)


@kb_app.command("show")
def kb_show(name: str = typer.Argument(..., help="KB name")) -> None:
    """Print a manifest summary for KB <name>."""
    from lab.rag.index import count_rows, index_bytes
    from lab.rag.manifest import load_manifest

    kb_dir = _kb_dir(name)
    mf = kb_dir / "manifest.yaml"
    if not mf.exists():
        console.print(f"[red]no manifest at[/] {mf}")
        raise typer.Exit(code=2)
    m = load_manifest(mf)
    n_rows = count_rows(kb_dir)
    idx_bytes = index_bytes(kb_dir)
    table = Table("Field", "Value")
    table.add_row("name", m.name)
    table.add_row("status", m.status)
    table.add_row("description", (m.description or "").strip()[:200])
    table.add_row("kb_format_version", str(m.kb_format_version))
    table.add_row("chunk_format_version", str(m.chunk_format_version))
    table.add_row("created_at", m.created_at)
    table.add_row("last_refreshed_at", m.last_refreshed_at or "-")
    table.add_row("embedding.model", m.models.embedding.name)
    table.add_row("embedding.dims", str(m.models.embedding.dimensions))
    table.add_row("chunker", f"{m.models.chunker.name} v{m.models.chunker.version}")
    table.add_row("sources", str(len(m.sources)))
    table.add_row("stats.chunks", str(m.stats.chunk_count))
    table.add_row("stats.embedded_tokens", str(m.stats.embedded_token_count))
    table.add_row("index rows", str(n_rows))
    table.add_row("index bytes", str(idx_bytes))
    table.add_row("eval.retrieval@1", f"{m.eval.retrieval_at_1:.3f}")
    table.add_row("eval.retrieval@5", f"{m.eval.retrieval_at_5:.3f}")
    console.print(table)


@kb_app.command("query")
def kb_query(
    name: str = typer.Argument(..., help="KB name"),
    question: str = typer.Argument(..., help="Free-text query"),
    k: int = typer.Option(5, "--k", help="Top-k hits"),
    alpha: float = typer.Option(
        0.5, "--alpha", help="Dense weight (1.0=pure dense, 0.0=pure sparse)"
    ),
    authority: str = typer.Option(
        "", "--authority", help="Optional authority filter (e.g. official, manpage)"
    ),
) -> None:
    """Hybrid dense+sparse query against KB <name>."""
    from lab.rag.index import count_rows, hybrid_query

    kb_dir = _kb_dir(name)
    if not (kb_dir / "manifest.yaml").exists():
        console.print(f"[red]no KB at[/] {kb_dir}")
        raise typer.Exit(code=2)
    if count_rows(kb_dir) == 0:
        console.print(
            f"[yellow]KB {name!r} has no indexed chunks yet[/] "
            f"(status likely enrichment_pending/embedding_pending). "
            f"Skipping query — no Ollama call made."
        )
        return
    hits = hybrid_query(
        kb_dir,
        question,
        k=k,
        alpha=alpha,
        filter_authority=authority or None,
    )
    if not hits:
        console.print("[yellow]no hits[/]")
        return
    table = Table("#", "Score", "Dense", "Sparse", "Section", "Title", "Source")
    for i, h in enumerate(hits, 1):
        section = " / ".join(h.section_path) if h.section_path else "-"
        table.add_row(
            str(i),
            f"{h.score:.3f}",
            f"{h.dense_score:.3f}",
            f"{h.sparse_score:.3f}",
            section[:60],
            (h.title or "-")[:40],
            (h.source_url or "-")[:50],
        )
    console.print(table)


@kb_app.command("eval")
def kb_eval(
    name: str = typer.Argument(..., help="KB name"),
    n: int = typer.Option(30, "--n", help="Number of synthetic queries"),
    k: int = typer.Option(5, "--k", help="Top-k for hit@k"),
    model: str | None = typer.Option(None, "--model", help="Eval model (default qwen3:14b-q4_K_M)"),
) -> None:
    """Run the synthetic-query retrieval eval. Gated on the shared GPU lease.

    Refuses to run if `lab:gpu:lease:0` is held (e.g. by an active sweep). Wait
    for the sweep to finish, then retry.
    """
    import redis

    from lab.core.settings import get_settings
    from lab.rag.eval_retrieval import DEFAULT_EVAL_MODEL, run_eval

    settings = get_settings()
    try:
        client = redis.from_url(settings.redis_url)
        lease_holder = client.get("lab:gpu:lease:0")
    except Exception as e:
        console.print(
            f"[yellow]could not reach Valkey[/] ({e}); refusing to run eval to avoid "
            f"clashing with a sweep we can't see. Re-run when Valkey is reachable."
        )
        raise typer.Exit(code=2) from None
    if lease_holder:
        holder = lease_holder.decode() if isinstance(lease_holder, bytes) else str(lease_holder)
        console.print(
            f"[red]GPU lease held[/] by {holder!r} — refusing to run eval. "
            f"Wait for the running sweep to finish, then retry."
        )
        raise typer.Exit(code=2)

    kb_dir = _kb_dir(name)
    if not (kb_dir / "manifest.yaml").exists():
        console.print(f"[red]no KB at[/] {kb_dir}")
        raise typer.Exit(code=2)
    summary = run_eval(kb_dir, n=n, k=k, model=model or DEFAULT_EVAL_MODEL)
    console.print(summary)


@kb_app.command("cache-stats")
def kb_cache_stats() -> None:
    """Print Phase 8 RAG cache hit/miss counters (process-local snapshot).

    Counters live in-process; this command spins up a :class:`RagCache` so the
    Valkey connection is exercised and the snapshot reflects whatever the
    current process has accumulated. For long-running services use the
    Prometheus exporter instead.
    """
    from lab.rag.cache import RagCache

    cache = RagCache()
    snap = cache.stats_snapshot()
    table = Table("counter", "value")
    for k, v in sorted(snap.items()):
        table.add_row(k, str(v))
    console.print(table)


if __name__ == "__main__":
    app()
