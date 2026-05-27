"""Integration: one end-to-end sweep cell run via the real stack.

Touches: Postgres (experiments/tasks/models/experiment_runs), Valkey (GPU lease),
MinIO (trace upload), LiteLLM (chat completion). Skips cleanly when any of
those services or required rows are not present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from lab.sweep.config import RunConfig, SweepConfig
from lab.sweep.runner import execute_cell, expand_matrix
from lab.tasks.registry import Task, register_tasks

pytestmark = pytest.mark.integration


TEST_SUITE = "_it"
TEST_SLUG = "_it_sweep_e2e"
TEST_TASK_SLUG = "smoke-1"


def _resolve_test_model(pg: Any, served: set[str]) -> tuple[str, str] | None:
    """Pick a local model that is registered in lab.models AND served by LiteLLM."""
    with pg.cursor() as cur:
        cur.execute(
            "SELECT litellm_id, backend FROM models "
            "WHERE backend = 'ollama-local' AND litellm_id NOT LIKE '%%-' "
            "ORDER BY vram_gb NULLS LAST"
        )
        for row in cur.fetchall():
            if row[0] in served:
                return (row[0], row[1])
    return None


def _ensure_test_task(pg: Any) -> int:
    register_tasks(
        [
            Task(
                suite=TEST_SUITE,
                slug=TEST_TASK_SLUG,
                input="Reply with the single word: ok",
                gold_answer="ok",
            )
        ]
    )
    with pg.cursor() as cur:
        cur.execute(
            "SELECT task_id FROM tasks WHERE suite = %s AND slug = %s",
            (TEST_SUITE, TEST_TASK_SLUG),
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _read_litellm_key() -> str | None:
    p = Path("/data/lab/services/litellm-master-key")
    if not p.exists():
        return None
    return p.read_text().strip() or None


def test_sweep_cell_end_to_end(pg: Any, valkey: Any, minio_client: Any, litellm_url: str) -> None:
    litellm_key = _read_litellm_key()
    if litellm_key is None:
        pytest.skip("no LiteLLM master key file present")
    try:
        m_resp = httpx.get(
            f"{litellm_url}/v1/models",
            headers={"Authorization": f"Bearer {litellm_key}"},
            timeout=3,
        )
    except httpx.HTTPError as exc:
        pytest.skip(f"litellm /v1/models unreachable: {exc}")
    if m_resp.status_code != 200:
        pytest.skip(f"litellm /v1/models returned {m_resp.status_code}")
    served = {m.get("id") for m in m_resp.json().get("data", []) if isinstance(m.get("id"), str)}
    model = _resolve_test_model(pg, served)
    if model is None:
        pytest.skip(
            f"no local model registered in lab.models is served by litellm; served={sorted(served)[:5]}…"
        )

    # Ensure the test task is registered, then build a 1-cell sweep
    _ensure_test_task(pg)
    spec = SweepConfig(
        experiment={"slug": TEST_SLUG, "title": "integration self-test"},  # type: ignore[arg-type]
        tasks={"suite": TEST_SUITE, "slugs": [TEST_TASK_SLUG]},  # type: ignore[arg-type]
        models=[model[0]],
        configs=[RunConfig(name="greedy", temperature=0.0, top_p=1.0, max_tokens=8)],
        seeds=[1],
    )
    from lab.sweep.runner import _ensure_experiment, _models_lookup

    eid = _ensure_experiment(spec)
    models = _models_lookup([model[0]])
    cells = expand_matrix(spec, eid, models)
    assert len(cells) == 1
    cell = cells[0]
    result = execute_cell(cell, litellm_key=litellm_key, timeout=120)
    assert result.status in {"done", "error"}, f"unexpected status: {result.status}"
    # Verify row landed in DB
    with pg.cursor() as cur:
        cur.execute(
            "SELECT status, trace_path FROM experiment_runs WHERE run_id = %s", (cell.run_id,)
        )
        row = cur.fetchone()
    assert row is not None, "experiment_run row not inserted"
    db_status, trace_path = row
    assert db_status == result.status
    if result.status == "done":
        assert trace_path is not None
        assert trace_path.startswith("s3://")
        # Verify the trace blob is downloadable
        bucket, _, key = trace_path.removeprefix("s3://").partition("/")
        resp = minio_client.get_object(bucket, key)
        try:
            data = resp.read()
        finally:
            resp.close()
            resp.release_conn()
        first = json.loads(data.decode("utf-8").splitlines()[0])
        assert first["run_id"] == cell.run_id
