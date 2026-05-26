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
from lab.notify import get_ntfy_url, notify
from lab.quota import alert_if_high as quota_alert
from lab.quota import usage_window as quota_window
from lab.spend import backfill as spend_backfill
from lab.sweep.config import load_sweep
from lab.sweep.runner import cancel_sweep, get_sweep_status, run_sweep
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

    sandbox_cfg = lab_task.sandbox or {}
    network = sandbox_cfg.get("network", "none")
    env = dict(sandbox_cfg.get("env", {}))
    workspace_files_raw = sandbox_cfg.get("workspace_files") or {}
    workspace_files = {
        k: v.encode("utf-8") if isinstance(v, str) else v for k, v in workspace_files_raw.items()
    }

    from lab.agent.tools import task_needs_kb_mount
    from lab.settings import get_settings as _get_settings_kb

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

        from lab.settings import get_settings as _get_settings

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
        from lab.settings import get_settings as _get_settings_kb

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
    from lab.settings import get_settings

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

    from lab.rag.eval_retrieval import DEFAULT_EVAL_MODEL, run_eval
    from lab.settings import get_settings

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
