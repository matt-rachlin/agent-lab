"""End-to-end scorer test: real Inspect harness + real sandbox + scorers fire.

Exercises a task with `success_predicate: workspace_file_contains` and a
`tool_call` rubric, runs through `inspect_eval`, then asserts the
`end_state` and `tool_correctness` scorers landed on the EvalSample.

We don't gate on the model actually solving the task — local 12 GB
models are flaky on tool-call schemas. The test gates on the scorer
*mechanism* — they must produce a Score (pass or fail), and one of them
must have produced a non-NOANSWER value.

Skipped cleanly when LiteLLM, Ollama, or gVisor are not reachable.
"""

from __future__ import annotations

import os
import uuid as _uuid
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


def test_scorers_fire_e2e_workspace_predicate(pg: Any, tmp_path: Any) -> None:
    """Run an agent task with success_predicate + tool_call rubric.

    The task asks the model to write 'PASS' to `/workspace/answer.txt`.
    The `end_state` scorer reads the file from the workspace snapshot;
    `tool_correctness` asserts the fs_write tool was invoked.
    """
    from inspect_ai import eval as inspect_eval

    from lab.agent.sandbox import Sandbox
    from lab.inspect_bridge.adapter import lab_task_to_inspect
    from lab.tasks.registry import Task

    task = Task.model_validate(
        {
            "suite": "smoke",
            "slug": "scorer-e2e-write-pass",
            "input": (
                "Use the fs_write tool to write the literal text 'PASS' to a "
                "file at path 'answer.txt' (relative to /workspace) using mode "
                "'overwrite'. Then reply 'done'."
            ),
            "system": "You are an assistant with tool access. Call tools when asked.",
            "tools": [{"name": "fs_write"}],
            "max_turns": 4,
            "tool_budget": 3,
            "success_predicate": {
                "type": "workspace_file_contains",
                "path": "answer.txt",
                "substring": "PASS",
            },
            "rubric": {
                "type": "tool_call",
                "target_tool": "fs_write",
                "expected_args": {"path": "answer.txt"},
            },
        }
    )
    model = os.environ.get("LAB_SMOKE_AGENT_MODEL", "qwen3-14b-q4")

    with Sandbox() as sandbox:
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

    # The scorer mechanism must have produced entries for end_state,
    # tool_correctness, and budget_respected (no judge — include_judge
    # not set). We don't require any specific scorer to PASS — only that
    # they fired and at least one returned a numeric (non-NOANSWER)
    # value.
    scores = sample.scores or {}
    score_names = list(scores.keys())
    assert any("end_state" in n for n in score_names), (
        f"end_state missing from scores: {score_names}"
    )
    assert any("tool_correctness" in n for n in score_names), (
        f"tool_correctness missing from scores: {score_names}"
    )
    assert any("budget_respected" in n for n in score_names), (
        f"budget_respected missing from scores: {score_names}"
    )

    # At least one scorer must have produced a numeric value (not NOANSWER).
    numeric_seen = False
    for s in scores.values():
        v = getattr(s, "value", None)
        if isinstance(v, (int, float)):
            numeric_seen = True
            break
        if isinstance(v, str) and v != "N":
            numeric_seen = True
            break
    assert numeric_seen, f"every scorer returned NOANSWER: {scores}"

    # The trajectory must have a workspace_snapshot stash even if empty.
    lab_agent = (sample.metadata or {}).get("lab_agent") or {}
    assert "workspace_snapshot" in lab_agent


def test_scorers_persist_to_postgres(pg: Any, minio_client: Any, tmp_path: Any) -> None:
    """A successful run with multiple scorers must land in agent_logs.turns.score_breakdown."""
    from inspect_ai import eval as inspect_eval

    from lab.agent.sandbox import Sandbox
    from lab.inspect_bridge.adapter import lab_task_to_inspect
    from lab.inspect_bridge.logwriter import SweepContext, write_run_from_inspect_log
    from lab.tasks.registry import Task

    task = Task.model_validate(
        {
            "suite": "smoke",
            "slug": "scorer-e2e-persist",
            "input": "Reply 'ok'.",
            "max_turns": 1,
            "tool_budget": 0,
            "success_predicate": {
                "type": "workspace_file_exists",
                "path": "never-written.txt",
            },
        }
    )
    model = os.environ.get("LAB_SMOKE_AGENT_MODEL", "qwen3-14b-q4")

    run_id_ = f"scorer-e2e-{_uuid.uuid4().hex[:10]}"
    with Sandbox() as sandbox:
        inspect_task = lab_task_to_inspect(
            task,
            model=model,
            sandbox=sandbox,
            temperature=0.0,
            max_tokens=64,
        )
        logs = inspect_eval(
            inspect_task,
            display="none",
            log_samples=True,
            log_dir=str(tmp_path / "inspect"),
        )

    log = logs[0]

    with pg.cursor() as cur:
        cur.execute("SELECT experiment_id FROM experiments LIMIT 1")
        row = cur.fetchone()
        if row is None:
            pytest.skip("no experiments rows in db")
        exp_id = int(row[0])
        cur.execute("SELECT task_id FROM tasks LIMIT 1")
        task_row = cur.fetchone()
        cur.execute("SELECT model_id FROM models LIMIT 1")
        model_row = cur.fetchone()
        cur.execute("SELECT manifest_sha FROM manifests LIMIT 1")
        man_row = cur.fetchone()
        if not (task_row and model_row and man_row):
            pytest.skip("required FK rows missing in db")
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
        config={"temperature": 0.0, "max_tokens": 64},
        seed=0,
        manifest_sha=manifest_sha,
    )
    trace_uri = write_run_from_inspect_log(log, ctx)
    assert trace_uri.startswith("s3://")

    with pg.cursor() as cur:
        cur.execute("SELECT turns FROM agent_logs WHERE run_id = %s", (run_id_,))
        row = cur.fetchone()
    assert row is not None
    persisted = row[0]
    # Either bare list (no score_breakdown) or dict with score_breakdown.
    if isinstance(persisted, dict):
        breakdown = persisted.get("score_breakdown") or {}
        # At minimum budget_respected + end_state should have fired.
        scorer_names = set(breakdown.keys())
        assert any("budget_respected" in n for n in scorer_names), scorer_names
        assert any("end_state" in n for n in scorer_names), scorer_names
