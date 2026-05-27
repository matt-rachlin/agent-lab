"""Verify the Phase 16.4.2 prompt migration: PBS-Agent + PBS-Agent-RAG.

Every PBS-Agent v0.1 and PBS-Agent-RAG v0.1 task should reference a
canonical prompt via ``system_prompt_id`` (rather than inlining a
``system:`` block), and every referenced prompt must exist in the
registry under ``prompts/library/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.eval.prompts import PromptRegistry
from lab.tasks.registry import Task, load_tasks

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_ROOT = REPO_ROOT / "prompts" / "library"
PBS_AGENT_DIR = REPO_ROOT / "tasks" / "pbs-agent-v0.1"
PBS_AGENT_RAG_DIR = REPO_ROOT / "tasks" / "pbs-agent-rag-v0.1"


def _load_all(dir_: Path) -> list[Task]:
    out: list[Task] = []
    for path in sorted(dir_.glob("*.yaml")):
        out.extend(load_tasks(path))
    return out


@pytest.fixture(scope="module")
def registry() -> PromptRegistry:
    return PromptRegistry(root=PROMPTS_ROOT)


def test_pbs_agent_v01_all_tasks_use_prompt_id() -> None:
    tasks = _load_all(PBS_AGENT_DIR)
    assert tasks, "no PBS-Agent v0.1 tasks loaded"
    for t in tasks:
        assert t.system is None, (
            f"{t.suite}/{t.slug} still has inline `system:` — "
            f"should reference system_prompt_id instead"
        )
        assert t.system_prompt_id is not None, f"{t.suite}/{t.slug} missing system_prompt_id"


def test_pbs_agent_rag_v01_all_tasks_use_prompt_id() -> None:
    tasks = _load_all(PBS_AGENT_RAG_DIR)
    assert tasks, "no PBS-Agent-RAG v0.1 tasks loaded"
    for t in tasks:
        assert t.system is None, (
            f"{t.suite}/{t.slug} still has inline `system:` — "
            f"should reference system_prompt_id instead"
        )
        assert t.system_prompt_id is not None, f"{t.suite}/{t.slug} missing system_prompt_id"


def test_every_referenced_prompt_id_exists(registry: PromptRegistry) -> None:
    """Every prompt_id used by PBS-Agent / PBS-Agent-RAG must resolve."""
    referenced: set[str] = set()
    for t in _load_all(PBS_AGENT_DIR) + _load_all(PBS_AGENT_RAG_DIR):
        if t.system_prompt_id:
            referenced.add(t.system_prompt_id)
    assert referenced, "no prompts referenced — suspicious"
    missing = [pid for pid in sorted(referenced) if not registry.has(pid)]
    assert not missing, f"unresolved system_prompt_ids: {missing}"


def test_canonical_prompts_present_in_library(registry: PromptRegistry) -> None:
    """The Phase 16.4.2 canonical prompts must all be in the library."""
    expected = {
        "agent_system_v1",
        "tool_use_system_v1",
        "debug_assistant_v1",
        "rag_grounded_v1",
        "bash_expert_grounded_v1",
        "judge_score_1to5_v1",
    }
    listed = {meta.prompt_id for meta in registry.list()}
    missing = expected - listed
    assert not missing, f"canonical prompts missing from library: {sorted(missing)}"
