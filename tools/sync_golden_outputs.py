"""Generate or refresh frozen golden outputs for the eval suites.

Walks `tasks/<suite>/*.yaml`, runs a canonical model against each task
(via `lab agent run` or its single-turn equivalent), and writes the
captured response + trajectory + scorer outcomes to
`evals/golden/<suite>/<task_slug>/<model>.json`.

Defaults are conservative:

* `--dry-run` lists what would be generated without running any model;
  this is the safe mode and what CI / nightly should run by default.
* Without `--dry-run`, the script refuses to launch if `lab:gpu:lease:0`
  is held — sibling sweeps may be using the GPU. Pass `--force-lease` to
  bypass (e.g. when capturing a cloud-only run).
* Existing golden files are skipped UNLESS their stored `config_hash`
  differs from the current matrix (a config change forces recapture).
  Pass `--force` to overwrite unconditionally.

Initial population targets:

* pbs-v0.1: qwen3-14b-q4 + gpt-oss-120b-cloud (~48 files)
* pbs-agent-v0.1: llama3.1-8b-q4 + gpt-oss-120b-cloud (~24 files)
* pbs-agent-rag-v0.1: glm-5.1-cloud (~6 files)

Total ~78 goldens.

Usage:

    # See what would be generated; touches no model
    uv run python tools/sync_golden_outputs.py --dry-run

    # Capture goldens for one suite + model (real model run)
    uv run python tools/sync_golden_outputs.py \\
        --suite pbs-v0.1 --model qwen3-14b-q4

    # Re-capture even if files exist
    uv run python tools/sync_golden_outputs.py \\
        --suite pbs-v0.1 --model qwen3-14b-q4 --force
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Canonical (suite, model) capture matrix.
#
# Kept here so `--dry-run` always shows the same expected file list. Edit
# this dict (rather than the CLI) when the target matrix changes.
# ---------------------------------------------------------------------------

DEFAULT_MATRIX: dict[str, list[str]] = {
    "pbs-v0.1": ["qwen3-14b-q4", "gpt-oss-120b-cloud"],
    "pbs-agent-v0.1": ["llama3.1-8b-q4", "gpt-oss-120b-cloud"],
    "pbs-agent-rag-v0.1": ["glm-5.1-cloud"],
}


# ---------------------------------------------------------------------------
# Suite layout helpers — find the task files on disk.
# ---------------------------------------------------------------------------

# Map suite name -> tasks/ directory.
SUITE_DIRS: dict[str, str] = {
    "pbs-v0.1": "tasks/pbs",
    "pbs-agent-v0.1": "tasks/pbs-agent-v0.1",
    "pbs-agent-rag-v0.1": "tasks/pbs-agent-rag-v0.1",
    "agent-smoke": "tasks/agent-smoke",
}

# Map our label (used in output paths) -> canonical suite name in the
# `tasks` table. The pbs-v0.1 directory is "pbs-v0.1" on disk and in
# `evals/golden/`, but the DB row stores `PBS-v0.1` (uppercase) — that's
# the only one that differs.
SUITE_DB_NAMES: dict[str, str] = {
    "pbs-v0.1": "PBS-v0.1",
    "pbs-agent-v0.1": "pbs-agent-v0.1",
    "pbs-agent-rag-v0.1": "pbs-agent-rag-v0.1",
    "agent-smoke": "agent-smoke",
}

# Cap on total cloud calls. Sub-cent each via the Ollama Cloud Pro
# subscription, but bail hard if the matrix balloons — a sweep blowing
# through here unattended would be a problem.
CLOUD_CALL_BUDGET = 200

# Single-turn capture defaults — match the EXP-001 baseline so any future
# replay through the sweep runner produces the same config_hash and the
# goldens stay relevant.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_REQUEST_TIMEOUT_SEC = 600


def _load_task_slugs(suite_dir: Path) -> list[str]:
    """Collect every `slug:` entry from every YAML file in `suite_dir`."""
    import yaml

    slugs: list[str] = []
    for yaml_path in sorted(suite_dir.glob("*.yaml")):
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        tasks = raw.get("tasks") if isinstance(raw, dict) else None
        if isinstance(tasks, list):
            for entry in tasks:
                if isinstance(entry, dict) and "slug" in entry:
                    slugs.append(str(entry["slug"]))
    return slugs


def _config_hash(suite: str, model: str) -> str:
    """Short hash of the capture-relevant config (suite + model + sampling).

    Includes the sampling parameters that affect reproducibility so a
    capture taken at different defaults is distinguishable.
    """
    payload = json.dumps(
        {
            "suite": suite,
            "model": model,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _gpu_lease_held() -> bool:
    """True iff `lab:gpu:lease:0` is currently held by anything.

    Soft-imports lab.core so the script remains useful in --dry-run mode
    even when Postgres / Valkey isn't reachable.
    """
    try:
        from lab.core.gpu_lease import status as gpu_status

        holder, _ttl = gpu_status()
        return holder is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Backend / service preflight
# ---------------------------------------------------------------------------


def _model_backend(model: str) -> str:
    """Return the backend string ('ollama-local', 'ollama-cloud', etc).

    Soft-imports the DB so --dry-run still works without Postgres. If the
    model can't be looked up, default to 'unknown' which is treated as
    cloud (no GPU lease) — local runs hit their own preflight via the
    sweep runner.
    """
    try:
        import psycopg

        from lab.core.settings import get_settings

        with (
            psycopg.connect(get_settings().pg_dsn) as conn,
            conn.cursor() as cur,
        ):
            cur.execute(
                "SELECT backend FROM models WHERE litellm_id = %s LIMIT 1",
                (model,),
            )
            row = cur.fetchone()
            return str(row[0]) if row else "unknown"
    except Exception:
        return "unknown"


def _is_local_backend(backend: str) -> bool:
    return backend == "ollama-local"


def _is_cloud_backend(backend: str) -> bool:
    return backend in {"ollama-cloud", "ollama_cloud"}


def _ollama_reachable() -> bool:
    try:
        import httpx

        from lab.core.settings import get_settings

        r = httpx.get(
            get_settings().ollama_local_url.rstrip("/") + "/api/version",
            timeout=5.0,
        )
        return r.status_code == 200
    except Exception:
        return False


def _rerank_reachable() -> bool:
    port = os.environ.get("LAB_RAG_RERANKER_PORT", "8401")
    try:
        import httpx

        r = httpx.get(f"http://localhost:{port}/healthz", timeout=5.0)
        if r.status_code != 200:
            return False
        body = r.json()
        return bool(body.get("ok"))
    except Exception:
        return False


def _read_litellm_key() -> str:
    """Same source-of-truth as the sweep runner."""
    from lab.core.settings import get_settings

    settings = get_settings()
    if settings.litellm_key:
        return settings.litellm_key
    candidate = Path("/data/lab/services/litellm-master-key")
    if candidate.exists():
        return candidate.read_text().strip()
    return ""


# ---------------------------------------------------------------------------
# Rubric scoring (single-turn fast path)
# ---------------------------------------------------------------------------


def _score_exact_match(response_text: str, gold: str, case_sensitive: bool) -> float:
    if not gold or not response_text:
        return 0.0
    resp = response_text if case_sensitive else response_text.lower()
    target = gold if case_sensitive else gold.lower()
    pattern = re.compile(r"(?<!\w)" + re.escape(target) + r"(?!\w)")
    return 1.0 if pattern.search(resp) else 0.0


def _score_regex_match(response_text: str, pattern: str, case_sensitive: bool) -> float:
    if not response_text:
        return 0.0
    flags = re.DOTALL
    if not case_sensitive:
        flags |= re.IGNORECASE
    try:
        compiled = re.compile(pattern, flags)
    except re.error:
        return 0.0
    return 1.0 if compiled.search(response_text) else 0.0


def _score_single_turn(task_payload: dict[str, Any], response_text: str) -> dict[str, float]:
    """Score a pbs-v0.1-style task using its rubric.

    Returns a dict like {"exact_match": 1.0} or {"regex_match": 0.0}.
    Unknown rubric types yield an empty dict (the comparator handles
    missing scorers as max drift — explicit is fine here).
    """
    rubric = task_payload.get("rubric") or {}
    rtype = rubric.get("type")
    case_sensitive = bool(rubric.get("case_sensitive", False))
    if rtype == "exact_match":
        gold = task_payload.get("gold_answer") or ""
        return {"exact_match": _score_exact_match(response_text, str(gold), case_sensitive)}
    if rtype == "regex":
        pattern = rubric.get("pattern") or ""
        return {"regex_match": _score_regex_match(response_text, str(pattern), case_sensitive)}
    return {}


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _build_messages(task_payload: dict[str, Any]) -> list[dict[str, str]]:
    """Build chat messages from the task payload.

    Mirrors `lab.sweep.runner._build_messages` precedence (task system >
    nothing else, since we don't carry sweep-level defaults here).
    """
    messages: list[dict[str, str]] = []
    system = task_payload.get("system")
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(task_payload["input"])})
    return messages


def _capture_single_turn(
    *,
    suite_db: str,
    suite_label: str,
    task_slug: str,
    model: str,
    backend: str,
    timeout: int,
    force_lease: bool,
) -> dict[str, Any]:
    """Single-turn capture for pbs-v0.1-style tasks."""

    from lab.core.llm import call_litellm_chat
    from lab.core.settings import get_settings
    from lab.tasks.registry import get_tasks

    rows = get_tasks(suite_db, [task_slug])
    if not rows:
        raise RuntimeError(f"no task row for suite={suite_db!r} slug={task_slug!r}")
    payload = rows[0]["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)

    messages = _build_messages(payload)
    settings = get_settings()
    litellm_key = _read_litellm_key()

    def _do_call() -> tuple[dict[str, Any], int]:
        return call_litellm_chat(
            settings=settings,
            litellm_key=litellm_key,
            model=model,
            messages=messages,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            timeout=timeout,
        )

    if _is_local_backend(backend) and not force_lease:
        from lab.core.gpu_lease import gpu_lease

        with gpu_lease(f"goldens:{model}:{task_slug}", ttl_sec=timeout + 60):
            resp_json, _ = _do_call()
    else:
        resp_json, _ = _do_call()

    message = ((resp_json.get("choices") or [{}])[0]).get("message") or {}
    response_text = str(message.get("content") or "")
    scorers = _score_single_turn(payload, response_text)

    captured_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "task_slug": task_slug,
        "model": model,
        "suite": suite_label,
        "config_hash": _config_hash(suite_label, model),
        "captured_at": captured_at,
        "response_text": response_text,
        "tool_calls": [],
        "scorer_outcomes": scorers,
        "trajectory_summary": {
            "actual_turns": 1,
            "tool_call_count": 0,
            "terminated_reason": "model_finished",
        },
    }


def _capture_agent(
    *,
    suite_db: str,
    suite_label: str,
    task_slug: str,
    model: str,
    backend: str,
    timeout: int,
    force_lease: bool,
) -> dict[str, Any]:
    """Multi-turn agent capture for pbs-agent-* tasks.

    Mirrors `lab agent run --no-persist`: build the lab Task from the
    registry row, hand it to `lab_task_to_inspect`, run via `inspect_eval`
    inside a Sandbox. We extract response/tool_calls/scorers from the
    resulting EvalLog.
    """

    import shutil
    import tempfile

    from inspect_ai import eval as inspect_eval

    from lab.agent.sandbox import Sandbox
    from lab.agent.tools import (
        task_needs_hf_cache_mount,
        task_needs_kb_mount,
    )
    from lab.core.settings import get_settings
    from lab.inspect_bridge.adapter import lab_task_to_inspect
    from lab.tasks.registry import Task as LabTask
    from lab.tasks.registry import get_tasks

    rows = get_tasks(suite_db, [task_slug])
    if not rows:
        raise RuntimeError(f"no task row for suite={suite_db!r} slug={task_slug!r}")
    row = rows[0]
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)

    lab_task = LabTask.model_validate(
        {
            "suite": row["suite"],
            "slug": row["slug"],
            "category": row.get("category"),
            "difficulty": row.get("difficulty"),
            "input": payload["input"],
            "system": payload.get("system"),
            "tools": payload.get("tools"),
            "max_turns": payload.get("max_turns", 1),
            "tool_budget": payload.get("tool_budget", 0),
            "success_predicate": payload.get("success_predicate"),
            "sandbox": payload.get("sandbox"),
            "gold_answer": payload.get("gold_answer"),
            "rubric": payload.get("rubric"),
            "description": payload.get("description"),
        }
    )

    sandbox_cfg = lab_task.sandbox or {}
    network: Any = sandbox_cfg.get("network", "none")
    env: dict[str, str] = dict(sandbox_cfg.get("env", {}))
    workspace_files_raw = sandbox_cfg.get("workspace_files") or {}
    workspace_files: dict[str, bytes] = {
        k: v.encode("utf-8") if isinstance(v, str) else v for k, v in workspace_files_raw.items()
    }

    kb_root_mount: Path | None = None
    if task_needs_kb_mount(lab_task.tools):
        kb_root_mount = get_settings().kb_root
        env.setdefault("LAB_KB_ROOT", "/kb")
        env.setdefault("OLLAMA_HOST", "http://host.containers.internal:11434")
        if network == "none":
            network = ["host.containers.internal"]
        elif isinstance(network, list) and "host.containers.internal" not in network:
            network = [*network, "host.containers.internal"]

    hf_cache_mount: Path | None = None
    host_reranker = os.environ.get("LAB_RAG_RERANKER")
    if host_reranker is not None:
        env.setdefault("LAB_RAG_RERANKER", host_reranker)
    if task_needs_hf_cache_mount(lab_task.tools, reranker_env=env.get("LAB_RAG_RERANKER")):
        hf_cache_root = get_settings().hf_cache_root
        hf_cache_root.mkdir(parents=True, exist_ok=True)
        hf_cache_mount = hf_cache_root
        env.setdefault("HF_HOME", "/hf-cache")
        env.setdefault("TRANSFORMERS_CACHE", "/hf-cache/transformers")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        rerank_port = os.environ.get("LAB_RAG_RERANKER_PORT", "8401")
        env.setdefault(
            "LAB_RAG_RERANKER_URL",
            f"http://host.containers.internal:{rerank_port}",
        )

    def _run_eval(log_dir: str) -> Any:
        with Sandbox(
            network=network,
            env=env,
            workspace_files=workspace_files,
            time_limit_sec=timeout,
            kb_root_mount=kb_root_mount,
            hf_cache_mount=hf_cache_mount,
            hf_cache_target="/hf-cache",
        ) as sandbox:
            inspect_task = lab_task_to_inspect(
                lab_task,
                model=model,
                sandbox=sandbox,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            logs = inspect_eval(
                inspect_task,
                display="none",
                log_samples=True,
                log_dir=log_dir,
                log_format="json",
                log_realtime=False,
            )
        return logs

    parent_dir = tempfile.mkdtemp(prefix="lab-goldens-")
    log_dir = str(Path(parent_dir) / "inspect")
    try:
        if _is_local_backend(backend) and not force_lease:
            from lab.core.gpu_lease import gpu_lease

            with gpu_lease(f"goldens:{model}:{task_slug}", ttl_sec=timeout + 60):
                logs = _run_eval(log_dir)
        else:
            logs = _run_eval(log_dir)
    finally:
        shutil.rmtree(parent_dir, ignore_errors=True)

    if not logs:
        raise RuntimeError("inspect_ai.eval returned no logs")
    log = logs[0]
    samples = getattr(log, "samples", None) or []
    if not samples:
        raise RuntimeError("inspect log has no samples")
    sample = samples[0]
    metadata = sample.metadata or {}
    lab_agent = metadata.get("lab_agent") or {}

    # Flatten tool calls across all turns into the canonical
    # [{tool, args}] list the comparator expects.
    tool_calls: list[dict[str, Any]] = []
    for turn in lab_agent.get("turns") or []:
        for tc in turn.get("tool_calls") or []:
            tool_calls.append(
                {
                    "tool": tc.get("tool") or "",
                    "args": tc.get("args") or {},
                }
            )

    # Pull the final assistant turn out of state.messages.
    response_text = ""
    for msg in reversed(list(sample.messages or [])):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "assistant":
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if isinstance(content, str):
                response_text = content
            elif isinstance(content, list):
                # Inspect rich-content lists — flatten to text.
                parts: list[str] = []
                for c in content:
                    parts.append(getattr(c, "text", None) or str(c))
                response_text = "\n".join(parts)
            else:
                response_text = str(content or "")
            break

    # Per-scorer outcomes. The Inspect Score's `.value` may be a float, a
    # NOANSWER sentinel ("N"), or a string. Coerce floats; drop the rest.
    scorer_outcomes: dict[str, float] = {}
    for name, score in (sample.scores or {}).items():
        value = getattr(score, "value", None)
        if isinstance(value, str) and value == "N":
            continue
        try:
            scorer_outcomes[str(name)] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue

    captured_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "task_slug": task_slug,
        "model": model,
        "suite": suite_label,
        "config_hash": _config_hash(suite_label, model),
        "captured_at": captured_at,
        "response_text": response_text,
        "tool_calls": tool_calls,
        "scorer_outcomes": scorer_outcomes,
        "trajectory_summary": {
            "actual_turns": int(lab_agent.get("actual_turns") or 0),
            "tool_call_count": int(lab_agent.get("tool_call_count") or 0),
            "terminated_reason": lab_agent.get("terminated_reason") or "unknown",
        },
    }


def _capture(
    *,
    suite_db: str,
    suite_label: str,
    task_slug: str,
    model: str,
    backend: str,
    timeout: int,
    force_lease: bool,
) -> dict[str, Any]:
    """Run the model against (suite, task_slug) and return a golden payload.

    Dispatches between the single-turn fast path (pbs-v0.1) and the
    multi-turn agent path (pbs-agent-*).
    """
    if suite_label.startswith("pbs-agent"):
        return _capture_agent(
            suite_db=suite_db,
            suite_label=suite_label,
            task_slug=task_slug,
            model=model,
            backend=backend,
            timeout=timeout,
            force_lease=force_lease,
        )
    return _capture_single_turn(
        suite_db=suite_db,
        suite_label=suite_label,
        task_slug=task_slug,
        model=model,
        backend=backend,
        timeout=timeout,
        force_lease=force_lease,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_matrix(
    args: argparse.Namespace,
) -> list[tuple[str, str, str]]:
    """Return [(suite, task_slug, model), ...] for the requested scope."""
    suites = [args.suite] if args.suite is not None else list(DEFAULT_MATRIX.keys())
    triples: list[tuple[str, str, str]] = []
    repo_root = Path(__file__).resolve().parent.parent
    for suite in suites:
        models = [args.model] if args.model is not None else DEFAULT_MATRIX.get(suite, [])
        if not models:
            print(f"warning: no models in matrix for suite={suite!r}; skipping")
            continue
        suite_dir_name = SUITE_DIRS.get(suite)
        if suite_dir_name is None:
            print(f"warning: no tasks dir mapping for suite={suite!r}; skipping")
            continue
        suite_dir = repo_root / suite_dir_name
        if not suite_dir.exists():
            print(f"warning: suite dir missing: {suite_dir}")
            continue
        slugs = _load_task_slugs(suite_dir)
        if args.task is not None:
            slugs = [s for s in slugs if s == args.task]
        for slug in slugs:
            for model in models:
                triples.append((suite, slug, model))
    return triples


def _existing_golden_matches(target: Path, expected_hash: str) -> bool:
    """True iff the file at `target` exists AND its config_hash matches."""
    if not target.exists():
        return False
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(raw.get("config_hash") or "") == expected_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate frozen golden outputs for eval suites")
    parser.add_argument(
        "--suite",
        choices=sorted(DEFAULT_MATRIX),
        help="limit to one suite (default: all)",
    )
    parser.add_argument(
        "--model",
        help="limit to one model (default: matrix entries per suite)",
    )
    parser.add_argument(
        "--task",
        help="limit to one task slug",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("evals/golden"),
        help="output root (default: evals/golden/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be generated; do not touch any model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing golden files even when hash matches",
    )
    parser.add_argument(
        "--force-lease",
        action="store_true",
        help="proceed even if lab:gpu:lease:0 is held",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SEC,
        help=f"per-cell request timeout (default: {DEFAULT_REQUEST_TIMEOUT_SEC})",
    )
    args = parser.parse_args(argv)

    triples = _build_matrix(args)
    if not triples:
        print("nothing to do (matrix empty)", file=sys.stderr)
        return 0

    # First pass: pick which cells need capture vs. skip.
    to_capture: list[tuple[str, str, str]] = []
    skip_count = 0
    for suite, slug, model in triples:
        target = args.root / suite / slug / f"{model}.json"
        expected_hash = _config_hash(suite, model)
        if not args.force and _existing_golden_matches(target, expected_hash):
            print(f"skip (exists, hash match): {target}")
            skip_count += 1
            continue
        if not args.force and target.exists():
            print(f"recapture (config_hash drift): {target}")
        else:
            print(f"{'would-write' if args.dry_run else 'write'}: {target}")
        to_capture.append((suite, slug, model))

    print(
        f"\nsummary: {len(to_capture)} to capture, {skip_count} already present "
        f"(total triples: {len(triples)})"
    )

    if args.dry_run:
        return 0

    if not to_capture:
        return 0

    # Cloud-call budget guard. Backends are looked up once so we don't
    # hit Postgres in the inner loop.
    backends: dict[str, str] = {}
    cloud_calls = 0
    for _suite, _slug, model in to_capture:
        if model not in backends:
            backends[model] = _model_backend(model)
        if _is_cloud_backend(backends[model]):
            cloud_calls += 1
    if cloud_calls > CLOUD_CALL_BUDGET:
        print(
            f"refusing to capture: cloud-call budget exceeded "
            f"({cloud_calls} > {CLOUD_CALL_BUDGET}). Trim the matrix "
            f"or raise CLOUD_CALL_BUDGET.",
            file=sys.stderr,
        )
        return 4
    print(f"cloud-call budget: {cloud_calls} cells (limit {CLOUD_CALL_BUDGET})")

    if not args.force_lease and _gpu_lease_held():
        print(
            "\nrefusing to capture: lab:gpu:lease:0 is held by another "
            "process. Pass --force-lease to override, or retry later.",
            file=sys.stderr,
        )
        return 2

    # Preflight: Ollama must be reachable (local AND cloud route through
    # the proxy via Ollama). Rerank only required if any RAG task is in
    # the to_capture set.
    if not _ollama_reachable():
        print(
            "refusing to capture: Ollama not reachable at "
            f"{os.environ.get('LAB_OLLAMA_LOCAL_URL', 'http://localhost:11434')}. "
            "Start ollama before running.",
            file=sys.stderr,
        )
        return 5
    if any(s == "pbs-agent-rag-v0.1" for s, _, _ in to_capture) and not _rerank_reachable():
        print(
            "refusing to capture: rerank service not reachable at "
            "http://localhost:${LAB_RAG_RERANKER_PORT:-8401}/healthz. "
            "Start the reranker (or set LAB_RAG_RERANKER=none if you "
            "intend to capture without it).",
            file=sys.stderr,
        )
        return 6

    captured_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    print(f"\ncapture mode (captured_at={captured_at}):")
    n_written = 0
    n_errored = 0
    errors: list[tuple[str, str, str, str]] = []
    wall_start = time.monotonic()
    for suite, slug, model in to_capture:
        target = args.root / suite / slug / f"{model}.json"
        suite_db = SUITE_DB_NAMES.get(suite, suite)
        backend = backends.get(model) or _model_backend(model)
        cell_start = time.monotonic()
        try:
            payload = _capture(
                suite_db=suite_db,
                suite_label=suite,
                task_slug=slug,
                model=model,
                backend=backend,
                timeout=args.timeout,
                force_lease=args.force_lease,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cell_ms = int((time.monotonic() - cell_start) * 1000)
            n_written += 1
            print(f"  + {target} ({cell_ms} ms)")
        except Exception as exc:
            n_errored += 1
            tb = traceback.format_exc()
            print(
                f"  ! {target}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            errors.append((suite, slug, model, f"{type(exc).__name__}: {exc}\n{tb}"))

    wall_sec = int(time.monotonic() - wall_start)
    print(
        f"\ndone: {n_written} written, {n_errored} errored, {skip_count} skipped — wall {wall_sec}s"
    )

    if n_errored:
        # If more than 5% of cells errored, fail loudly. Otherwise return
        # 0 — partial captures are acceptable when most cells succeed and
        # the caller can re-run for the few stragglers.
        error_pct = n_errored / max(len(to_capture), 1)
        if error_pct > 0.05:
            print(
                f"\n{n_errored}/{len(to_capture)} cells errored "
                f"({100 * error_pct:.1f}%, > 5% threshold)",
                file=sys.stderr,
            )
            return 7
        print(f"  ({n_errored}/{len(to_capture)} cells errored — under 5% threshold)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
