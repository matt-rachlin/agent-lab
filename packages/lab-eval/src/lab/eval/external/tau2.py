"""τ²-bench adapter (Stage-1 D4 / task #16).

Turns Sierra/Salesforce τ²-bench domain task files into lab ``Task`` rows
so the suite is *registerable* and *referenceable* from a sweep config the
same way ``bfcl-v3-ast`` is (see ``lab.eval.external.bfcl``).

Scope of THIS adapter (honest boundary):

  * **Loading / registration — wired.** ``load_tau2_tasks`` reads the
    vendored domain ``tasks.json`` files and produces ``LabTask`` rows
    under the suite name :data:`SUITE_NAME`. ``lab data add-benchmark
    tau2-bench`` registers them into ``lab.tasks`` (DB), after which a
    sweep ``tasks.suite: tau2-bench`` resolves via ``get_tasks`` exactly
    like BFCL.

  * **Execution / scoring — NOT wired here (blocked on a runner lane).**
    τ²-bench is a *dual-control* multi-turn benchmark: each task needs a
    user-simulator counterpart-LLM, a stateful domain DB + tool set, and
    an action/DB/NL-assertion reward function. The sweep runner only has
    three dispatch lanes today (``bfcl`` / ``agent`` / ``single_turn``);
    none simulate the τ² user or grade τ² reward. Wiring that lane is a
    runner concern and is deliberately out of scope here. Each task row
    therefore carries a ``rubric.type == "custom"`` with the full τ²
    bookkeeping (domain, user scenario, evaluation criteria) preserved so
    a future ``_execute_tau2_cell`` lane has everything it needs.

Vendor data
-----------
Expected on m-box at ``/data/lab/vendor/tau2-bench`` (present as of
2026-06). Override the root with ``LAB_TAU2_DATA_DIR``. If the vendor
tree is absent, :func:`domain_tasks_path` raises ``FileNotFoundError``
naming the missing path and the fetch command::

    git clone https://github.com/sierra-research/tau2-bench \\
        /data/lab/vendor/tau2-bench
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lab.tasks.registry import Task as LabTask
from lab.tasks.registry import TaskRubric

SUITE_NAME = "tau2-bench"

#: Default vendored data root on m-box (overridable via ``LAB_TAU2_DATA_DIR``).
_DEFAULT_VENDOR_ROOT = Path("/data/lab/vendor/tau2-bench")

#: Upstream clone command surfaced in error messages when vendor data is absent.
FETCH_HINT = "git clone https://github.com/sierra-research/tau2-bench /data/lab/vendor/tau2-bench"

#: τ² domains that ship a ``tasks.json`` (insertion order = registration order).
DEFAULT_DOMAINS: tuple[str, ...] = ("airline", "retail", "telecom")


def dataset_root() -> Path:
    """Where the vendored τ²-bench tree lives. Overridable via env."""
    override = os.environ.get("LAB_TAU2_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_VENDOR_ROOT


def domain_tasks_path(domain: str, *, root: Path | None = None) -> Path:
    """Resolve the ``tasks.json`` for one domain, asserting it exists.

    Raises ``FileNotFoundError`` (naming the path + fetch command) when the
    vendor tree is missing — callers turn this into an operator-facing
    "needs vendor data at X" message rather than fabricating tasks.
    """
    root = root or dataset_root()
    path = root / "data" / "tau2" / "domains" / domain / "tasks.json"
    if not path.exists():
        raise FileNotFoundError(
            f"τ²-bench domain {domain!r} tasks not found at {path}. "
            f"Vendor the dataset first: {FETCH_HINT}"
        )
    return path


def available_domains(*, root: Path | None = None) -> list[str]:
    """List domains that actually have a ``tasks.json`` on disk (may be empty)."""
    root = root or dataset_root()
    domains_dir = root / "data" / "tau2" / "domains"
    if not domains_dir.is_dir():
        return []
    found: list[str] = []
    for child in sorted(domains_dir.iterdir()):
        if (child / "tasks.json").exists():
            found.append(child.name)
    return found


def _user_scenario_text(task: dict[str, Any]) -> str:
    """Flatten the τ² ``user_scenario`` block into the agent-visible prompt.

    τ² hides the scenario behind a user-simulator at run time; for the lab
    Task ``input`` we render the human-readable scenario so the row is
    inspectable and a future runner lane can re-derive the simulator seed.
    """
    scenario = task.get("user_scenario") or {}
    instr = scenario.get("instructions") or {}
    parts: list[str] = []
    for key in ("domain", "reason_for_call", "known_info", "task_instructions"):
        val = instr.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"{key}: {val.strip()}")
    desc = (task.get("description") or {}).get("purpose")
    if isinstance(desc, str) and desc.strip():
        parts.insert(0, f"purpose: {desc.strip()}")
    return "\n\n".join(parts) if parts else json.dumps(task.get("user_scenario") or {})


def _tau2_system_prompt(domain: str) -> str:
    return (
        f"You are a customer-service agent for the {domain} domain. Help the "
        "user accomplish their request using only the provided domain tools "
        "and policies. Follow policy strictly; refuse out-of-policy requests."
    )


def tau2_task_to_lab_task(task: dict[str, Any], *, domain: str) -> LabTask:
    """Build one lab ``Task`` row from a τ² task dict.

    The rubric is ``type="custom"`` (TaskRubric allows extra keys) carrying
    the τ² bookkeeping a future execution lane needs: ``tau2_domain``,
    ``tau2_id``, the raw ``user_scenario``, ``evaluation_criteria`` and
    ``initial_state``. Nothing here grades — that's the deferred lane.
    """
    tid = str(task.get("id", ""))
    slug = f"{domain}-{tid}".replace("/", "-").replace(":", "-")
    rubric_obj = TaskRubric.model_validate(
        {
            "type": "custom",
            "tau2_domain": domain,
            "tau2_id": tid,
            "user_scenario": task.get("user_scenario"),
            "evaluation_criteria": task.get("evaluation_criteria"),
            "initial_state": task.get("initial_state"),
        }
    )
    return LabTask(
        suite=SUITE_NAME,
        slug=slug,
        category=domain,
        external_id=f"{domain}/{tid}",
        description=f"τ²-bench {domain} task {tid}",
        input=_user_scenario_text(task),
        system=_tau2_system_prompt(domain),
        # Multi-turn dual-control: signalled here so the row is honest about
        # its shape, even though no runner lane consumes it yet.
        max_turns=40,
        tool_budget=60,
        rubric=rubric_obj,
    )


def load_tau2_tasks(
    domains: Iterable[str] = DEFAULT_DOMAINS,
    *,
    limit_per_domain: int | None = None,
    root: Path | None = None,
) -> list[LabTask]:
    """Load τ² tasks from the vendored tree and return them as lab Tasks.

    Args:
        domains: which domains to load (default: airline/retail/telecom).
        limit_per_domain: optional cap (first N by file order).
        root: dataset root override (testing).

    Raises:
        FileNotFoundError: if a requested domain's ``tasks.json`` is absent
            (message names the path + the clone command).
    """
    out: list[LabTask] = []
    for domain in domains:
        path = domain_tasks_path(domain, root=root)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(
                f"τ² {domain} tasks.json must be a JSON list, got {type(raw).__name__}"
            )
        rows = raw[:limit_per_domain] if limit_per_domain is not None else raw
        for task in rows:
            if not isinstance(task, dict):
                continue
            out.append(tau2_task_to_lab_task(task, domain=domain))
    return out


__all__ = [
    "DEFAULT_DOMAINS",
    "FETCH_HINT",
    "SUITE_NAME",
    "available_domains",
    "dataset_root",
    "domain_tasks_path",
    "load_tau2_tasks",
    "tau2_task_to_lab_task",
]
