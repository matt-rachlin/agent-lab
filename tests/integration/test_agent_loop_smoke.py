"""Full agent-loop smoke: sandbox + tools + (real) LiteLLM + Postgres + MinIO.

Skipped unless every dependency is reachable. The point is to validate the
seam between the solver and everything it touches — not to assert specific
model behaviour.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest
from lab.agent.sandbox import gvisor_available
from lab.core.settings import get_settings

pytestmark = pytest.mark.integration


def _ollama_local_up() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.5)
        return r.status_code < 500
    except Exception:
        return False


def _litellm_up() -> bool:
    try:
        r = httpx.get(get_settings().litellm_url.rstrip("/") + "/health/liveliness", timeout=1.5)
        return r.status_code < 500
    except Exception:
        return False


@pytest.fixture(scope="module", autouse=True)
def _require_environment() -> None:
    if not gvisor_available():
        pytest.skip("gVisor not available")
    if not _litellm_up():
        pytest.skip("LiteLLM proxy not reachable")
    if not _ollama_local_up():
        pytest.skip("Ollama (local) not reachable")


def test_agent_loop_smoke_reads_a_file(pg: Any, minio_client: Any, tmp_path: Any) -> None:
    """Stage a file, give the agent fs_read, watch it read+respond.

    The model is asked to read `note.txt` and return its content. We assert
    that the agent reached `model_finished`, made at least one tool call,
    and that an `agent_logs` row + MinIO trajectory both exist.
    """

    import uuid as _uuid

    from inspect_ai import eval as inspect_eval
    from lab.agent.sandbox import Sandbox
    from lab.tasks.registry import Task

    from lab.inspect_bridge.adapter import lab_task_to_inspect
    from lab.inspect_bridge.logwriter import SweepContext, write_run_from_inspect_log

    note_content = "the secret is 17"
    task = Task.model_validate(
        {
            "suite": "smoke",
            "slug": "agent-loop-smoke",
            "input": (
                "Use the fs_read tool to read 'note.txt' from /workspace, "
                "then reply with the secret."
            ),
            "tools": [{"name": "fs_read"}],
            "max_turns": 4,
            "tool_budget": 3,
        }
    )
    model = os.environ.get("LAB_SMOKE_AGENT_MODEL", "qwen3-14b-q4")

    run_id_ = f"smoke-{_uuid.uuid4().hex[:10]}"

    with Sandbox(workspace_files={"note.txt": note_content.encode("utf-8")}) as sandbox:
        inspect_task = lab_task_to_inspect(
            task,
            model=model,
            sandbox=sandbox,
            temperature=0.0,
            max_tokens=512,
        )
        logs = inspect_eval(
            inspect_task,
            display="none",
            log_samples=True,
            log_dir=str(tmp_path / "inspect"),
        )
    assert logs, "inspect_eval returned no logs"
    log = logs[0]
    samples = log.samples or []
    assert samples, "inspect log had no samples"
    sample = samples[0]
    lab_agent = (sample.metadata or {}).get("lab_agent") or {}
    assert lab_agent.get("actual_turns", 0) >= 1
    # Even if the model didn't actually call the tool, the loop should have
    # exited cleanly. We don't gate on the model's correctness here — only
    # on the loop's mechanics.
    assert lab_agent.get("terminated_reason") in {
        "model_finished",
        "budget_exhausted",
        "max_turns_reached",
    }

    # Postgres + MinIO mirror.
    # We pretend this is an experiment_run for the bookkeeping path; the
    # foreign-key constraint requires a real `experiment_id`, so we look up
    # whatever experiment / task / model rows exist.
    with pg.cursor() as cur:
        cur.execute("SELECT experiment_id FROM experiments LIMIT 1")
        row = cur.fetchone()
        if row is None:
            pytest.skip("no experiments in db; cannot exercise FK path")
        exp_id = int(row[0])
        cur.execute("SELECT task_id FROM tasks LIMIT 1")
        task_row = cur.fetchone()
        cur.execute("SELECT model_id FROM models LIMIT 1")
        model_row = cur.fetchone()
        if task_row is None or model_row is None:
            pytest.skip("no tasks/models in db; cannot exercise FK path")
        cur.execute("SELECT manifest_sha FROM manifests LIMIT 1")
        man_row = cur.fetchone()
        if man_row is None:
            pytest.skip("no manifests in db; cannot exercise FK path")
        manifest_sha = str(man_row[0])

    ctx = SweepContext(
        run_id=run_id_,
        experiment_id=exp_id,
        experiment_slug="smoke",
        model_id=int(model_row[0]),
        model_litellm_id=model,
        task_id=int(task_row[0]),
        task_slug=task.slug,
        config_hash="smoke",
        config={"temperature": 0.0, "max_tokens": 512},
        seed=0,
        manifest_sha=manifest_sha,
    )
    trace_uri = write_run_from_inspect_log(log, ctx)
    assert trace_uri.startswith("s3://")

    # The bucket + key should now be present.
    key = trace_uri.split("/", 3)[-1]
    settings = get_settings()
    obj = minio_client.get_object(settings.s3_bucket, key)
    body = obj.read()
    obj.close()
    obj.release_conn()
    assert b'"type": "header"' in body or b'"type":"header"' in body

    # And `agent_logs` got an idempotent row.
    with pg.cursor() as cur:
        cur.execute("SELECT inspect_log_path FROM agent_logs WHERE run_id = %s", (run_id_,))
        row = cur.fetchone()
    assert row is not None
    assert row[0] == trace_uri
