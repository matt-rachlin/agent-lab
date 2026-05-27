"""One-time backfill of existing lab Postgres data into MLflow.

Usage::

    python -m tools.backfill_mlflow --dry-run     # preview counts
    python -m tools.backfill_mlflow               # do it
    python -m tools.backfill_mlflow --force       # re-mirror rows that already have ids
    python -m tools.backfill_mlflow --runs-limit 100  # cap (debug)

Idempotent: rows that already carry the relevant ``mlflow_*_id`` are
skipped unless ``--force`` is passed. Errors on individual rows are
recorded but don't stop the walk.
"""

from __future__ import annotations

import argparse
import json
import sys

from lab.observability.mlflow_backfill import backfill_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview counts, write nothing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-mirror rows that already carry an mlflow_*_id",
    )
    parser.add_argument(
        "--runs-limit",
        type=int,
        default=None,
        help="Cap the number of experiment_runs to mirror (debug aid)",
    )
    args = parser.parse_args(argv)

    summary = backfill_all(
        dry_run=args.dry_run,
        force=args.force,
        runs_limit=args.runs_limit,
    )
    print(json.dumps(summary.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
