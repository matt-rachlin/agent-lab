"""Unit tests for the τ²-bench → lab Task adapter (Stage-1 D4 / task #16).

These tests are vendor-data-light: the task-shape tests run off in-memory
fixtures + tmp dirs, so they pass without the real τ²-bench tree. A single
guarded test exercises the vendored corpus when present on m-box.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.eval.external.tau2 import (
    DEFAULT_DOMAINS,
    SUITE_NAME,
    available_domains,
    domain_tasks_path,
    load_tau2_tasks,
    tau2_task_to_lab_task,
)

_VENDOR_ROOT = Path("/data/lab/vendor/tau2-bench")


def test_suite_name_constant() -> None:
    assert SUITE_NAME == "tau2-bench"


def test_default_domains() -> None:
    assert set(DEFAULT_DOMAINS) == {"airline", "retail", "telecom"}


def _fixture_task(tid: str = "0") -> dict[str, object]:
    return {
        "id": tid,
        "description": {"purpose": "Test refusal on out-of-policy cancellation."},
        "user_scenario": {
            "instructions": {
                "domain": "airline",
                "reason_for_call": "Cancel reservation EHGLP3.",
                "known_info": "You are Emma Kim. user id emma_kim_9957.",
                "task_instructions": "Insist if refused.",
            }
        },
        "evaluation_criteria": {
            "nl_assertions": ["Agent should refuse to proceed with the cancellation."],
            "reward_basis": ["DB", "COMMUNICATE"],
        },
        "initial_state": None,
    }


def test_task_to_lab_task_shape() -> None:
    task = tau2_task_to_lab_task(_fixture_task("0"), domain="airline")
    assert task.suite == "tau2-bench"
    assert task.slug == "airline-0"
    assert task.category == "airline"
    assert task.external_id == "airline/0"
    assert task.system is not None
    assert "airline" in task.system
    # The user scenario is flattened into the agent-visible input.
    assert "EHGLP3" in task.input
    assert "purpose:" in task.input
    # Multi-turn dual-control shape is signalled even though no lane consumes it.
    assert task.max_turns > 1


def test_rubric_carries_tau2_bookkeeping() -> None:
    task = tau2_task_to_lab_task(_fixture_task("7"), domain="airline")
    assert task.rubric is not None
    dumped = task.rubric.model_dump()
    assert dumped["type"] == "custom"
    assert dumped["tau2_domain"] == "airline"
    assert dumped["tau2_id"] == "7"
    # The full evaluation criteria are preserved for the future runner lane.
    assert dumped["evaluation_criteria"]["nl_assertions"]
    assert dumped["user_scenario"]["instructions"]["domain"] == "airline"


def test_load_from_tmp_dir(tmp_path: Path) -> None:
    """load_tau2_tasks reads a vendored-shaped tree from an arbitrary root."""
    dom_dir = tmp_path / "data" / "tau2" / "domains" / "airline"
    dom_dir.mkdir(parents=True)
    (dom_dir / "tasks.json").write_text(
        json.dumps([_fixture_task("0"), _fixture_task("1")]), encoding="utf-8"
    )
    tasks = load_tau2_tasks(["airline"], root=tmp_path)
    assert len(tasks) == 2
    assert {t.slug for t in tasks} == {"airline-0", "airline-1"}
    # limit_per_domain caps.
    assert len(load_tau2_tasks(["airline"], limit_per_domain=1, root=tmp_path)) == 1


def test_missing_vendor_data_raises_with_path_and_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        domain_tasks_path("airline", root=tmp_path)
    msg = str(exc.value)
    assert "airline" in msg
    assert "tasks.json" in msg
    assert "git clone" in msg  # fetch command surfaced


def test_available_domains_empty_when_absent(tmp_path: Path) -> None:
    assert available_domains(root=tmp_path) == []


@pytest.mark.skipif(
    not (_VENDOR_ROOT / "data" / "tau2" / "domains" / "airline" / "tasks.json").exists(),
    reason="τ²-bench vendor data not present at /data/lab/vendor/tau2-bench",
)
def test_load_real_vendor_airline() -> None:
    tasks = load_tau2_tasks(["airline"], limit_per_domain=5, root=_VENDOR_ROOT)
    assert len(tasks) == 5
    assert all(t.suite == "tau2-bench" for t in tasks)
    assert all(t.category == "airline" for t in tasks)
