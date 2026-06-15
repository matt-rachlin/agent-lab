"""Harbor verifier → eval_results writer (M3 workstream).

Reads a Harbor verifier output file (JSON Lines) and writes one row per task
into ``eval_results``.  The input format is intentionally minimal — one JSON
object per line:

.. code-block:: json

    {"task": "adaptive-rejection-sampler", "passed": true, "score": 1.0}
    {"task": "matrix-multiply", "passed": false, "score": 0.0}

Fields:

* ``task``   — the Terminal-Bench task slug (must exist in ``tasks`` table under
               suite ``harbor``).
* ``passed`` — bool; whether the verifier declared success.
* ``score``  — float in [0, 1]; numeric grade (passed→1.0 if omitted).

Idempotency policy: ``INSERT … ON CONFLICT (run_id, evaluator_id) DO NOTHING``
— a second ingest of the same ``run_id`` x task pair is silently skipped.
``rows_written`` counts only the rows actually inserted (not the skipped ones).

Usage (CLI surface in ``lab eval ingest-harbor``)::

    from lab.eval.external.harbor_ingest import ingest_harbor_run
    counts = ingest_harbor_run(
        Path("harbor-results.jsonl"),
        run_id="EXP-013-r0001",
        suite="harbor",
        trust_level="unverified",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Json

from lab.core.settings import get_settings
from lab.observability.log import get_logger

log = get_logger(__name__)

# Evaluator name + version registered by this module.
_EVALUATOR_NAME = "harbor_verifier"
_EVALUATOR_VERSION = "1.0"


def _ensure_evaluator(cur: psycopg.Cursor[Any]) -> int:
    """Upsert the harbor_verifier evaluator row and return its evaluator_id."""
    cur.execute(
        """
        INSERT INTO evaluators (name, version, category, module_path)
        VALUES (%s, %s, 'external', 'lab.eval.external.harbor_ingest')
        ON CONFLICT (name, version) DO NOTHING;
        """,
        (_EVALUATOR_NAME, _EVALUATOR_VERSION),
    )
    cur.execute(
        "SELECT evaluator_id FROM evaluators WHERE name = %s AND version = %s;",
        (_EVALUATOR_NAME, _EVALUATOR_VERSION),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(
            f"evaluator {_EVALUATOR_NAME!r}/{_EVALUATOR_VERSION!r} vanished after INSERT"
        )
    return int(row[0])


def _task_id_for(
    cur: psycopg.Cursor[Any],
    slug: str,
    suite: str,
    run_id: str,
) -> int | None:
    """Return task_id for (suite, slug) or None if not found (with warning)."""
    cur.execute(
        "SELECT task_id FROM tasks WHERE suite = %s AND slug = %s AND retired_at IS NULL;",
        (suite, slug),
    )
    row = cur.fetchone()
    if row is None:
        log.warning(
            "harbor_ingest_unknown_task",
            task_slug=slug,
            suite=suite,
            run_id=run_id,
        )
        return None
    return int(row[0])


def ingest_harbor_run(
    results_path: Path,
    *,
    run_id: str,
    suite: str = "harbor",
    trust_level: str = "unverified",
) -> dict[str, int]:
    """Read a Harbor verifier output file, write one row per task to eval_results.

    The input file must be JSON Lines; each line is a JSON object with at
    minimum ``{"task": "<slug>", "passed": <bool>}``.  An optional ``"score"``
    float is used when present; otherwise ``score = 1.0 if passed else 0.0``.

    The ``run_id`` must already exist in ``experiment_runs``; if it does not,
    the INSERT will raise a foreign-key violation and the whole call fails.

    The ``trust_level`` parameter is recorded in the ``raw`` JSONB column for
    downstream trust-lifecycle queries (ADR-006/ADR-008).

    Returns:
        dict with keys ``rows_written``, ``passed``, ``failed``,
        ``skipped_unknown_task``.
    """
    counts: dict[str, int] = {
        "rows_written": 0,
        "passed": 0,
        "failed": 0,
        "skipped_unknown_task": 0,
    }

    lines = results_path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("harbor_ingest_bad_line", line_no=i + 1, error=str(exc))
            continue
        if not isinstance(obj, dict):
            log.warning("harbor_ingest_non_object_line", line_no=i + 1)
            continue
        if "task" not in obj:
            log.warning("harbor_ingest_missing_task_key", line_no=i + 1)
            continue
        records.append(obj)

    if not records:
        log.info("harbor_ingest_no_records", path=str(results_path), run_id=run_id)
        return counts

    dsn = get_settings().pg_dsn
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        evaluator_id = _ensure_evaluator(cur)

        for rec in records:
            slug = str(rec["task"])
            passed = bool(rec.get("passed", False))
            score_raw = rec.get("score")
            score = float(score_raw) if score_raw is not None else (1.0 if passed else 0.0)

            task_id = _task_id_for(cur, slug, suite, run_id)
            if task_id is None:
                counts["skipped_unknown_task"] += 1
                continue

            raw_payload: dict[str, Any] = {
                "harbor": rec,
                "trust_level": trust_level,
                "task_id": task_id,
            }

            cur.execute(
                """
                INSERT INTO eval_results (run_id, evaluator_id, score, passed, raw)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (run_id, evaluator_id) DO NOTHING;
                """,
                (run_id, evaluator_id, score, passed, Json(raw_payload)),
            )
            inserted = cur.rowcount
            if inserted > 0:
                counts["rows_written"] += 1
                if passed:
                    counts["passed"] += 1
                else:
                    counts["failed"] += 1

    log.info(
        "harbor_ingest_complete",
        path=str(results_path),
        run_id=run_id,
        **counts,
    )
    return counts


__all__ = ["ingest_harbor_run"]
