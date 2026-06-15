"""PBS-Agent RAG v0.1 task suite — schema + shape checks.

Mirrors `test_pbs_agent_v01_load.py` for the new RAG-aware slice. Asserts:

  * every YAML in `tasks/pbs-agent-rag-v0.1/` loads cleanly
  * suite name is the expected literal
  * every task references the `kb_query` tool
  * every task has `max_turns >= 2` and a non-zero `tool_budget`
  * every task carries a `success_predicate`
  * at least one task uses `retrieval_recall`
  * at least one task uses `include_faithfulness: true`
  * at least one task uses `include_judge: true`
  * every referenced tool exists in `TOOL_SERVERS`
"""

from __future__ import annotations

from pathlib import Path

from lab.agent.tools import TOOL_SERVERS
from lab.tasks.registry import Task, load_tasks

SUITE_DIR = Path(__file__).resolve().parents[2] / "tasks" / "pbs-agent-rag-v0.1"
EXPECTED_SUITE = "pbs-agent-rag-v0.1"


def _all_tasks() -> list[Task]:
    files = sorted(SUITE_DIR.glob("*.yaml"))
    assert files, f"no YAML files under {SUITE_DIR}"
    out: list[Task] = []
    for f in files:
        out.extend(load_tasks(f))
    return out


def test_suite_directory_exists() -> None:
    assert SUITE_DIR.is_dir()


def test_suite_loads_and_has_minimum_tasks() -> None:
    tasks = _all_tasks()
    # Plan calls for 5-8 tasks; lock the floor at 5 so a hand-deletion is
    # caught, not the ceiling.
    assert len(tasks) >= 5, f"expected >= 5 RAG tasks, got {len(tasks)}"


def test_every_task_uses_expected_suite() -> None:
    for t in _all_tasks():
        assert t.suite == EXPECTED_SUITE, f"{t.slug}: suite={t.suite!r}"


def test_every_task_uses_kb_query_tool() -> None:
    for t in _all_tasks():
        assert t.tools, f"{t.slug}: no tools list"
        names = {spec.get("name") for spec in t.tools if isinstance(spec, dict)}
        assert "kb_query" in names, f"{t.slug}: kb_query missing from tools={names}"


def test_every_task_references_known_tools() -> None:
    for t in _all_tasks():
        for spec in t.tools or []:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            assert name in TOOL_SERVERS, f"{t.slug}: unknown tool {name!r}"


def test_every_task_has_multi_turn_budget() -> None:
    """max_turns must be >= 2 (RAG tasks need at least retrieve + answer).
    tool_budget must be > 0 (a 0 budget would forbid all tool calls).
    """

    for t in _all_tasks():
        assert t.max_turns >= 2, f"{t.slug}: max_turns={t.max_turns} < 2"
        assert t.tool_budget > 0, f"{t.slug}: tool_budget={t.tool_budget} <= 0"


def test_every_task_has_success_predicate() -> None:
    for t in _all_tasks():
        assert isinstance(t.success_predicate, dict), (
            f"{t.slug}: success_predicate is {t.success_predicate!r}"
        )
        assert t.success_predicate.get("type"), f"{t.slug}: success_predicate missing 'type'"


def test_at_least_one_task_uses_retrieval_recall() -> None:
    types = {
        t.success_predicate.get("type")
        for t in _all_tasks()
        if isinstance(t.success_predicate, dict)
    }
    assert "retrieval_recall" in types, "no task uses retrieval_recall predicate"


def test_at_least_one_task_uses_include_faithfulness() -> None:
    flags = [
        bool(t.success_predicate.get("include_faithfulness"))
        for t in _all_tasks()
        if isinstance(t.success_predicate, dict)
    ]
    assert any(flags), "no task opts into faithfulness scoring"


def test_at_least_one_task_uses_include_judge() -> None:
    flags = [
        bool(t.success_predicate.get("include_judge"))
        for t in _all_tasks()
        if isinstance(t.success_predicate, dict)
    ]
    assert any(flags), "no task opts into trajectory_judge scoring"


def test_retrieval_recall_task_has_k_parameter() -> None:
    """retrieval_recall predicates must declare a `k` (recall@k cutoff)."""

    for t in _all_tasks():
        pred = t.success_predicate
        if isinstance(pred, dict) and pred.get("type") == "retrieval_recall":
            assert "k" in pred, f"{t.slug}: retrieval_recall predicate missing 'k'"
            assert int(pred["k"]) > 0, (
                f"{t.slug}: retrieval_recall predicate has non-positive k={pred['k']}"
            )
