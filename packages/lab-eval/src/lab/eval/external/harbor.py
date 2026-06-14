"""Harbor / Terminal-Bench suite loader (Stage-1 D4 / task #16).

Makes ``harbor`` referenceable + registerable as a lab task *suite* the
way ``bfcl-v3-ast`` is. Companion to the *agent* side already shipped in
``lab.agent.harbor_adapter`` (``LabReactAgent`` — the ReAct scaffold port
that Harbor drives). This module is the *data/registration* side: it turns
the vendored Terminal-Bench task corpus into ``LabTask`` rows.

Scope of THIS adapter (honest boundary):

  * **Loading / registration — wired.** ``load_harbor_tasks`` walks the
    vendored Terminal-Bench task directories (each a ``task.toml`` +
    ``instruction.md``) and produces ``LabTask`` rows under the suite name
    :data:`SUITE_NAME`. ``lab data add-benchmark harbor`` registers them
    into ``lab.tasks`` (DB); a sweep ``tasks.suite: harbor`` then resolves
    via ``get_tasks`` exactly like BFCL. The rows give the sweep a stable,
    selectable task universe + manifest of what was run.

  * **Execution / scoring — runs OUTSIDE the sweep runner.** Terminal-Bench
    tasks are Docker-image-backed and graded by Harbor's own verifier
    (``harbor run --agent-import-path lab_react_agent:LabReactAgent`` —
    see ``harbor_adapter`` docstring). The lab sweep runner has no Docker
    lane and does NOT execute these cells; ``rubric.type == "custom"`` here
    is a registration/selection marker, not a runner-dispatched scorer. So
    a ``harbor`` sweep config is a *manifest of the cohort*, executed by the
    Harbor CLI, not by ``lab sweep run``. This is called out in the sweep
    config header too.

Vendor data
-----------
Expected on m-box at ``/data/lab/vendor/harbor-datasets/terminal-bench``
(present as of 2026-06; 89 task dirs). Override with
``LAB_HARBOR_DATA_DIR``. If absent, :func:`tasks_root` raises
``FileNotFoundError`` naming the path + fetch command::

    git clone https://github.com/laude-institute/terminal-bench \\
        /data/lab/vendor/harbor-datasets/terminal-bench
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from lab.tasks.registry import Task as LabTask
from lab.tasks.registry import TaskRubric

SUITE_NAME = "harbor"

#: Default vendored corpus root on m-box (overridable via ``LAB_HARBOR_DATA_DIR``).
_DEFAULT_VENDOR_ROOT = Path("/data/lab/vendor/harbor-datasets/terminal-bench")

#: Upstream clone command surfaced in error messages when vendor data is absent.
FETCH_HINT = (
    "git clone https://github.com/laude-institute/terminal-bench "
    "/data/lab/vendor/harbor-datasets/terminal-bench"
)

_Difficulty = Literal["easy", "medium", "hard"]

#: Map Terminal-Bench difficulty strings to the lab Task difficulty Literal.
_DIFFICULTY_MAP: dict[str, _Difficulty] = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}


def tasks_root() -> Path:
    """Where the vendored Terminal-Bench task dirs live. Overridable via env.

    Raises ``FileNotFoundError`` (naming the path + fetch command) when the
    vendor tree is missing.
    """
    override = os.environ.get("LAB_HARBOR_DATA_DIR")
    root = Path(override).expanduser() if override else _DEFAULT_VENDOR_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"Harbor / Terminal-Bench corpus not found at {root}. Vendor it first: {FETCH_HINT}"
        )
    return root


def discover_task_dirs(*, root: Path | None = None) -> list[Path]:
    """Return the task directories (those containing a ``task.toml``), sorted."""
    root = root or tasks_root()
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "task.toml").exists())


def _read_instruction(task_dir: Path) -> str:
    instr = task_dir / "instruction.md"
    if instr.exists():
        return instr.read_text(encoding="utf-8").strip()
    return f"Complete the Terminal-Bench task in {task_dir.name}."


def _read_metadata(task_dir: Path) -> dict[str, Any]:
    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        return {}
    with toml_path.open("rb") as fh:
        data = tomllib.load(fh)
    return data if isinstance(data, dict) else {}


def harbor_dir_to_task(task_dir: Path) -> LabTask:
    """Build one lab ``Task`` row from a Terminal-Bench task directory.

    The rubric is ``type="custom"`` carrying the Harbor bookkeeping a
    selection / manifest consumer needs: ``harbor_task_id`` (the dir name),
    the ``docker_image`` and timeouts from ``task.toml``. Nothing here
    grades — Harbor's own verifier does.
    """
    meta = _read_metadata(task_dir)
    md_meta = meta.get("metadata") or {}
    env = meta.get("environment") or {}
    diff_raw = str(md_meta.get("difficulty", "")).lower()
    difficulty: _Difficulty | None = _DIFFICULTY_MAP.get(diff_raw)
    category = md_meta.get("category")
    task_id = task_dir.name

    rubric_obj = TaskRubric.model_validate(
        {
            "type": "custom",
            "harbor_task_id": task_id,
            "docker_image": env.get("docker_image"),
            "agent_timeout_sec": (meta.get("agent") or {}).get("timeout_sec"),
            "verifier_timeout_sec": (meta.get("verifier") or {}).get("timeout_sec"),
            "tags": md_meta.get("tags"),
        }
    )
    return LabTask(
        suite=SUITE_NAME,
        slug=task_id,
        category=category if isinstance(category, str) else None,
        difficulty=difficulty,
        external_id=task_id,
        description=f"Terminal-Bench task {task_id}",
        input=_read_instruction(task_dir),
        # Harbor drives its own ReAct loop (LabReactAgent); these knobs are
        # informational on the lab row since no sweep lane consumes them.
        max_turns=40,
        tool_budget=60,
        rubric=rubric_obj,
    )


def load_harbor_tasks(
    *,
    limit: int | None = None,
    root: Path | None = None,
    task_ids: Iterable[str] | None = None,
) -> list[LabTask]:
    """Load Terminal-Bench tasks from the vendored corpus as lab Tasks.

    Args:
        limit: optional cap (first N by sorted dir name).
        root: corpus root override (testing).
        task_ids: restrict to these task dir names (others skipped).

    Raises:
        FileNotFoundError: if the corpus root is absent (message names the
            path + the clone command).
    """
    wanted = set(task_ids) if task_ids is not None else None
    dirs = discover_task_dirs(root=root)
    if wanted is not None:
        dirs = [d for d in dirs if d.name in wanted]
    if limit is not None:
        dirs = dirs[:limit]
    return [harbor_dir_to_task(d) for d in dirs]


__all__ = [
    "FETCH_HINT",
    "SUITE_NAME",
    "discover_task_dirs",
    "harbor_dir_to_task",
    "load_harbor_tasks",
    "tasks_root",
]
