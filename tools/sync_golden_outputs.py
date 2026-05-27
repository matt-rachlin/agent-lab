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
* Existing golden files are skipped; pass `--force` to overwrite.

Initial population targets (planned):

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
import sys
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
    """Short hash of the capture-relevant config (suite + model)."""
    payload = json.dumps({"suite": suite, "model": model}, sort_keys=True).encode("utf-8")
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
# Capture
# ---------------------------------------------------------------------------


def _capture(suite: str, task_slug: str, model: str) -> dict[str, Any]:
    """Run the model against (suite, task_slug) and return a golden payload.

    Implementation note: a real capture path delegates to `lab agent run`
    (or the single-turn equivalent for non-agent suites). That dispatch
    lives in `lab.sweep.runner` / `lab.inspect_bridge.solver` — both of
    which are owned by sibling 16.1+16.2. Rather than reach across that
    boundary in this Phase 16 batch, we raise NotImplementedError so the
    user is forced to wire the runner in manually when they actually want
    to capture (which is deferred per the plan).

    This keeps `--dry-run` fully functional without depending on
    not-yet-stable observability runtime hooks.
    """
    raise NotImplementedError(
        "real-model capture is deferred to a manual user trigger; "
        "see evals/golden/README.md for the planned dispatch path"
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
        help="overwrite existing golden files",
    )
    parser.add_argument(
        "--force-lease",
        action="store_true",
        help="proceed even if lab:gpu:lease:0 is held",
    )
    args = parser.parse_args(argv)

    triples = _build_matrix(args)
    if not triples:
        print("nothing to do (matrix empty)", file=sys.stderr)
        return 0

    new_count = 0
    skip_count = 0
    for suite, slug, model in triples:
        target = args.root / suite / slug / f"{model}.json"
        if target.exists() and not args.force:
            print(f"skip (exists): {target}")
            skip_count += 1
            continue
        new_count += 1
        print(f"{'would-write' if args.dry_run else 'write'}: {target}")

    print(
        f"\nsummary: {new_count} to capture, {skip_count} already present "
        f"(total triples: {len(triples)})"
    )

    if args.dry_run:
        return 0

    if not args.force_lease and _gpu_lease_held():
        print(
            "\nrefusing to capture: lab:gpu:lease:0 is held by another "
            "process. Pass --force-lease to override, or retry later.",
            file=sys.stderr,
        )
        return 2

    # If we got here, a real capture is requested. Per the plan, this
    # path is intentionally not implemented yet — the runner integration
    # belongs to sibling 16.1+16.2. Fail clearly so the user can wire it
    # in manually when ready.
    captured_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    print(f"\ncapture mode (captured_at={captured_at}):")
    for suite, slug, model in triples:
        target = args.root / suite / slug / f"{model}.json"
        if target.exists() and not args.force:
            continue
        try:
            payload = _capture(suite, slug, model)
        except NotImplementedError as exc:
            print(f"  ! {target}: {exc}", file=sys.stderr)
            return 3
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"  + {target}")
    _ = _config_hash  # imported for use by future capture path
    return 0


if __name__ == "__main__":
    sys.exit(main())
