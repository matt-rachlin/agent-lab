"""Unit tests for the Harbor / Terminal-Bench suite loader (Stage-1 D4 / #16).

Vendor-data-light: task-shape tests build a synthetic task dir under tmp_path;
one guarded test reads the real vendored corpus when present on m-box.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.eval.external.harbor import (
    SUITE_NAME,
    discover_task_dirs,
    harbor_dir_to_task,
    load_harbor_tasks,
    tasks_root,
)

_VENDOR_ROOT = Path("/data/lab/vendor/harbor-datasets/terminal-bench")

_TASK_TOML = """\
version = "1.0"

[metadata]
author_name = "tester"
difficulty = "medium"
category = "scientific-computing"
tags = ["applied-statistics", "simulation"]

[verifier]
timeout_sec = 900.0

[agent]
timeout_sec = 900.0

[environment]
docker_image = "example/img:20251031"
cpus = 1
memory = "2G"
"""


def _make_task_dir(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text(_TASK_TOML, encoding="utf-8")
    (d / "instruction.md").write_text(f"Do the {name} task.", encoding="utf-8")
    return d


def test_suite_name_constant() -> None:
    assert SUITE_NAME == "harbor"


def test_dir_to_task_shape(tmp_path: Path) -> None:
    d = _make_task_dir(tmp_path, "adaptive-rejection-sampler")
    task = harbor_dir_to_task(d)
    assert task.suite == "harbor"
    assert task.slug == "adaptive-rejection-sampler"
    assert task.external_id == "adaptive-rejection-sampler"
    assert task.category == "scientific-computing"
    assert task.difficulty == "medium"
    assert "adaptive-rejection-sampler task" in task.input
    assert task.rubric is not None
    dumped = task.rubric.model_dump()
    assert dumped["type"] == "custom"
    assert dumped["harbor_task_id"] == "adaptive-rejection-sampler"
    assert dumped["docker_image"] == "example/img:20251031"
    assert dumped["agent_timeout_sec"] == 900.0


def test_discover_and_load_from_tmp(tmp_path: Path) -> None:
    _make_task_dir(tmp_path, "task-a")
    _make_task_dir(tmp_path, "task-b")
    (tmp_path / "not-a-task").mkdir()  # no task.toml -> ignored
    dirs = discover_task_dirs(root=tmp_path)
    assert [d.name for d in dirs] == ["task-a", "task-b"]
    tasks = load_harbor_tasks(root=tmp_path)
    assert {t.slug for t in tasks} == {"task-a", "task-b"}
    # limit + task_ids filters.
    assert len(load_harbor_tasks(root=tmp_path, limit=1)) == 1
    only = load_harbor_tasks(root=tmp_path, task_ids=["task-b"])
    assert [t.slug for t in only] == ["task-b"]


def test_missing_vendor_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_HARBOR_DATA_DIR", "/nonexistent/harbor/path/xyz")
    with pytest.raises(FileNotFoundError) as exc:
        tasks_root()
    msg = str(exc.value)
    assert "/nonexistent/harbor/path/xyz" in msg
    assert "git clone" in msg


@pytest.mark.skipif(
    not _VENDOR_ROOT.is_dir(),
    reason="Harbor vendor data not present at /data/lab/vendor/harbor-datasets/terminal-bench",
)
def test_load_real_vendor_corpus() -> None:
    tasks = load_harbor_tasks(limit=5, root=_VENDOR_ROOT)
    assert len(tasks) == 5
    assert all(t.suite == "harbor" for t in tasks)
    assert all(t.slug for t in tasks)
