"""PBS-Agent v0.1 task suite — schema and shape checks.

Loads every YAML file under `tasks/pbs-agent-v0.1/` and asserts each task
has the fields the harness needs (input, tools, max_turns, tool_budget,
success_predicate, sandbox.workspace_files where applicable). Also
asserts the cross-suite invariants from the 6f plan:

  * suite name is consistent
  * every tool referenced exists in the agent's TOOL_SERVERS registry
  * each of the 6 tools is touched by at least one task
  * at least one task uses success_predicate: db_query
  * at least one task uses include_judge: true
  * total task count == 12
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.agent.tools import TOOL_SERVERS
from lab.tasks.registry import Task, load_tasks

SUITE_DIR = Path(__file__).resolve().parents[2] / "tasks" / "pbs-agent-v0.1"
EXPECTED_SUITE = "pbs-agent-v0.1"
EXPECTED_TOTAL = 12


def _all_tasks() -> list[Task]:
    files = sorted(SUITE_DIR.glob("*.yaml"))
    assert files, f"no YAML files under {SUITE_DIR}"
    out: list[Task] = []
    for f in files:
        out.extend(load_tasks(f))
    return out


def test_suite_directory_exists() -> None:
    assert SUITE_DIR.is_dir(), f"{SUITE_DIR} missing"


def test_task_count_is_twelve() -> None:
    tasks = _all_tasks()
    assert len(tasks) == EXPECTED_TOTAL, f"expected {EXPECTED_TOTAL} tasks, got {len(tasks)}"


def test_every_task_has_required_fields() -> None:
    for task in _all_tasks():
        assert task.suite == EXPECTED_SUITE, f"{task.slug} has wrong suite {task.suite!r}"
        assert task.input, f"{task.slug} missing input"
        assert task.tools, f"{task.slug} missing tools list"
        assert task.max_turns >= 2, f"{task.slug} max_turns={task.max_turns} < 2"
        assert task.tool_budget >= 1, f"{task.slug} tool_budget={task.tool_budget} < 1"


def test_slugs_are_unique() -> None:
    slugs = [t.slug for t in _all_tasks()]
    assert len(set(slugs)) == len(slugs), f"duplicate slugs: {slugs}"


def test_difficulty_set_on_every_task() -> None:
    for task in _all_tasks():
        assert task.difficulty in {"easy", "medium", "hard"}, (
            f"{task.slug} difficulty={task.difficulty!r}"
        )


def test_every_referenced_tool_exists() -> None:
    for task in _all_tasks():
        for spec in task.tools or []:
            name = spec.get("name")
            assert name in TOOL_SERVERS, (
                f"{task.slug} references unknown tool {name!r}; "
                f"known: {sorted(TOOL_SERVERS)}"
            )


def test_every_tool_is_touched_by_some_task() -> None:
    touched: set[str] = set()
    for task in _all_tasks():
        for spec in task.tools or []:
            n = spec.get("name")
            if n:
                touched.add(n)
    missing = set(TOOL_SERVERS) - touched
    assert not missing, f"tools never touched by any task: {sorted(missing)}"


def test_success_predicate_shape() -> None:
    valid_types = {
        "workspace_file_contains",
        "workspace_file_equals",
        "workspace_file_exists",
        "db_query",
    }
    for task in _all_tasks():
        sp = task.success_predicate
        assert sp is not None, f"{task.slug} missing success_predicate"
        assert sp.get("type") in valid_types, (
            f"{task.slug} predicate type {sp.get('type')!r} not in {valid_types}"
        )


def test_at_least_one_db_query_predicate() -> None:
    hits = [t.slug for t in _all_tasks() if (t.success_predicate or {}).get("type") == "db_query"]
    assert hits, "no task uses success_predicate type=db_query"


def test_at_least_one_judge_task() -> None:
    hits = [t.slug for t in _all_tasks() if (t.success_predicate or {}).get("include_judge")]
    assert hits, "no task uses success_predicate.include_judge"


def test_tool_call_rubrics_have_target_tool() -> None:
    for task in _all_tasks():
        if task.rubric is not None and task.rubric.type == "tool_call":
            assert task.rubric.target_tool, (
                f"{task.slug} tool_call rubric missing target_tool"
            )
            # target_tool must be in the task's allowed tools list
            allowed = {spec.get("name") for spec in (task.tools or [])}
            assert task.rubric.target_tool in allowed, (
                f"{task.slug} target_tool {task.rubric.target_tool!r} not in tools={allowed}"
            )


def test_http_tasks_use_fixture_dir() -> None:
    """Each http_fetch task MUST run in offline-fixture mode."""

    for task in _all_tasks():
        names = {spec.get("name") for spec in (task.tools or [])}
        if "http_fetch" not in names:
            continue
        sb = task.sandbox or {}
        env = sb.get("env") or {}
        assert env.get("LAB_HTTP_FIXTURE_DIR"), (
            f"{task.slug} uses http_fetch but no LAB_HTTP_FIXTURE_DIR in sandbox.env"
        )
        # Must also have the http allow-list and a network list.
        assert env.get("LAB_HTTP_ALLOWLIST"), (
            f"{task.slug} uses http_fetch but no LAB_HTTP_ALLOWLIST in sandbox.env"
        )
        network = sb.get("network")
        assert isinstance(network, list), (
            f"{task.slug} uses http_fetch but sandbox.network is not a list"
        )
        assert network, (
            f"{task.slug} uses http_fetch but sandbox.network is empty"
        )


@pytest.mark.parametrize(
    "expected_difficulty", ["easy", "medium", "hard"],
)
def test_difficulty_mix_present(expected_difficulty: str) -> None:
    """Every difficulty bucket has at least one task."""

    hits = [t.slug for t in _all_tasks() if t.difficulty == expected_difficulty]
    assert hits, f"no tasks with difficulty={expected_difficulty}"
