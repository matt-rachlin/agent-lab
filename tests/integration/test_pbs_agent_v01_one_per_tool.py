"""PBS-Agent v0.1 — one task per tool, end-to-end through the harness.

Picks six tasks (one per tool in TOOL_SERVERS) and runs each through
`lab_task_to_inspect` + `inspect_eval`. The point is to assert the
*harness* completes cleanly for every tool category — we do NOT assert
that the model under test scored 1.0. A scorer value of 0.0 is fine; a
traceback, hung loop, or missing trajectory is not.

Skips cleanly when gVisor, LiteLLM, or Ollama are not reachable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from lab.agent.sandbox import gvisor_available
from lab.agent.tools import TOOL_SERVERS
from lab.core.settings import get_settings
from lab.tasks.registry import Task, load_tasks

pytestmark = pytest.mark.integration

SUITE_DIR = Path(__file__).resolve().parents[2] / "tasks" / "pbs-agent-v0.1"


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


def _all_tasks() -> list[Task]:
    out: list[Task] = []
    for f in sorted(SUITE_DIR.glob("*.yaml")):
        out.extend(load_tasks(f))
    return out


def _select_one_task_per_tool() -> dict[str, Task]:
    """Greedy: walk the 12 tasks once and assign each to the first
    listed tool that doesn't yet have a representative.

    This guarantees we cover all 6 tools with at most 6 tasks (assuming
    every tool is touched by some task — enforced by the unit suite).
    """

    tasks = _all_tasks()
    picks: dict[str, Task] = {}
    # Sort to make selection deterministic.
    for task in sorted(tasks, key=lambda t: t.slug):
        for spec in task.tools or []:
            name = spec.get("name")
            if name and name in TOOL_SERVERS and name not in picks:
                picks[name] = task
                break
    return picks


def _ids() -> list[str]:
    picks = _select_one_task_per_tool()
    return [f"{tool}:{task.slug}" for tool, task in sorted(picks.items())]


@pytest.mark.parametrize(
    ("tool_name", "task"),
    sorted(_select_one_task_per_tool().items()),
    ids=_ids(),
)
def test_one_task_per_tool_runs_through_harness(tool_name: str, task: Task, tmp_path: Any) -> None:
    """Run `task` end-to-end. Pass means the harness completed cleanly.

    Scorer outcomes are recorded in the trajectory but NOT asserted — the
    point is to validate the loop, not the model. We DO assert:
      * at least one turn happened
      * the loop terminated for a sanctioned reason (no traceback)
      * the scorers slot is populated (the adapter ran them)
    """

    from inspect_ai import eval as inspect_eval

    from lab.agent.sandbox import Sandbox
    from lab.inspect_bridge.adapter import lab_task_to_inspect

    model = os.environ.get("LAB_SMOKE_AGENT_MODEL", "llama3.1-8b-q4")

    sandbox_cfg = task.sandbox or {}
    network = sandbox_cfg.get("network", "none")
    env = dict(sandbox_cfg.get("env", {}))
    workspace_files_raw = sandbox_cfg.get("workspace_files") or {}
    workspace_files = {
        k: v.encode("utf-8") if isinstance(v, str) else v for k, v in workspace_files_raw.items()
    }

    with Sandbox(network=network, env=env, workspace_files=workspace_files) as sandbox:
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

    assert logs, f"{task.slug}: inspect_eval returned no logs"
    log = logs[0]
    samples = log.samples or []
    assert samples, f"{task.slug}: inspect log had no samples"
    sample = samples[0]
    lab_agent = (sample.metadata or {}).get("lab_agent") or {}

    # The loop made at least one turn (even if the model refused, we
    # still get one turn-entry for the refusal).
    assert lab_agent.get("actual_turns", 0) >= 1, (
        f"{task.slug}: zero turns recorded — harness did not run the loop"
    )

    # Termination is one of the known reasons; a traceback would land here
    # too but `terminated_reason` would not be a sanctioned value.
    sanctioned = {
        "model_finished",
        "budget_exhausted",
        "max_turns_reached",
    }
    assert lab_agent.get("terminated_reason") in sanctioned, (
        f"{task.slug}: terminated_reason={lab_agent.get('terminated_reason')!r}"
    )

    # The adapter wired up at least one scorer (end_state, tool_correctness,
    # or budget_respected — budget_respected is always added).
    scores = sample.scores or {}
    assert scores, f"{task.slug}: no scorers populated"
    assert "budget_respected" in scores, (
        f"{task.slug}: budget_respected scorer missing — adapter may be broken"
    )
