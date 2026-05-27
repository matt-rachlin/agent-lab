"""``lab agent run`` plumbs ``hf_cache_mount`` into the Sandbox.

The reranker weights are ~1.5 GB; downloading them per-cell is wasteful.
Phase 7 added ``hf_cache_mount`` to :class:`lab.agent.sandbox.Sandbox`;
this test pins the wiring that fires it iff the task can trigger the
reranker (kb_query in tools AND LAB_RAG_RERANKER != "none").

We invoke :func:`lab.cli.agent_run` directly with stubs for everything
heavy (registry, Sandbox lifecycle, Inspect bridge), capturing the
``Sandbox(...)`` kwargs to assert against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest


class _FakeSandbox:
    last_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        _FakeSandbox.last_kwargs = dict(kwargs)
        self.kwargs = kwargs
        self.container_name = "fake-sandbox"

    def __enter__(self) -> _FakeSandbox:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _fake_inspect_eval(*args: Any, **kwargs: Any) -> list[Any]:
    # agent_run treats an empty log list as a hard failure (Exit(1)); we
    # return one sample with the minimal metadata it reads after.
    class _Score:
        value = "ok"

    class _Sample:
        metadata: ClassVar[dict[str, Any]] = {
            "lab_agent": {
                "actual_turns": 0,
                "tool_call_count": 0,
                "terminated_reason": "stop",
                "total_latency_ms": 1,
                "error": None,
            }
        }
        scores: ClassVar[dict[str, _Score]] = {}

    class _Log:
        samples: ClassVar[list[_Sample]] = [_Sample()]

    return [_Log()]


def _stub_registry(monkeypatch: pytest.MonkeyPatch, tools: list[dict[str, Any]] | None) -> None:
    """Make ``get_tasks`` return one synthetic row carrying the given tools."""

    def _get_tasks(suite: str, slugs: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "suite": suite,
                "slug": slugs[0],
                "task_id": 1,
                "category": "rag",
                "difficulty": None,
                "payload": {
                    "input": "test",
                    "tools": tools,
                    "max_turns": 1,
                    "tool_budget": 0,
                },
            }
        ]

    monkeypatch.setattr("lab.tasks.registry.get_tasks", _get_tasks)


def _stub_inspect_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lab.inspect_bridge.adapter.lab_task_to_inspect", lambda *a, **k: object())
    monkeypatch.setattr("inspect_ai.eval", _fake_inspect_eval)


def _stub_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeSandbox.last_kwargs = {}
    monkeypatch.setattr("lab.agent.sandbox.Sandbox", _FakeSandbox)


def _run_agent_run(
    monkeypatch: pytest.MonkeyPatch,
    tools: list[dict[str, Any]] | None,
    *,
    reranker_env: str | None = None,
) -> None:
    from lab.core.settings import get_settings

    from lab import cli as cli_mod

    # Reset settings cache so monkeypatched env or paths apply cleanly.
    monkeypatch.setattr("lab.core.settings._settings", None, raising=False)
    if reranker_env is None:
        monkeypatch.delenv("LAB_RAG_RERANKER", raising=False)
    else:
        monkeypatch.setenv("LAB_RAG_RERANKER", reranker_env)
    _stub_registry(monkeypatch, tools)
    _stub_inspect_bridge(monkeypatch)
    _stub_sandbox(monkeypatch)
    # Skip Postgres write — we only care about Sandbox kwargs.
    cli_mod.agent_run(
        task="t",
        model="m",
        suite="agent",
        max_turns=None,
        tool_budget=None,
        temperature=0.0,
        max_tokens=16,
        no_persist=True,
    )
    # Force settings to load with the (possibly-monkeypatched) env.
    get_settings()


def test_agent_run_mounts_hf_cache_when_kb_query_and_reranker_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LAB_HF_CACHE_ROOT", str(tmp_path / "hfc"))
    monkeypatch.setattr("lab.core.settings._settings", None, raising=False)
    _run_agent_run(monkeypatch, [{"name": "kb_query"}], reranker_env=None)
    kw = _FakeSandbox.last_kwargs
    assert kw["hf_cache_mount"] == tmp_path / "hfc"
    assert kw["hf_cache_target"] == "/hf-cache"
    # The directory was created on-demand.
    assert (tmp_path / "hfc").is_dir()
    # The env vars steer transformers + HF Hub at the mount target, and
    # force offline mode (the sandbox network blocks huggingface.co — a
    # cache miss should fail loudly, not silently degrade).
    env = kw["env"]
    assert env["HF_HOME"] == "/hf-cache"
    assert env["TRANSFORMERS_CACHE"] == "/hf-cache/transformers"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    # Phase 7.1: when the reranker is enabled, the sandbox must point at
    # the host-side rerank service. Default port 8401, host alias via
    # podman's host.containers.internal.
    assert env["LAB_RAG_RERANKER_URL"] == "http://host.containers.internal:8401"


def test_agent_run_skips_hf_cache_when_reranker_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LAB_HF_CACHE_ROOT", str(tmp_path / "hfc"))
    monkeypatch.setattr("lab.core.settings._settings", None, raising=False)
    _run_agent_run(monkeypatch, [{"name": "kb_query"}], reranker_env="none")
    kw = _FakeSandbox.last_kwargs
    assert kw["hf_cache_mount"] is None
    # Mount target still defaulted (Sandbox ignores it when mount is None).
    assert "HF_HOME" not in kw["env"]
    assert "TRANSFORMERS_CACHE" not in kw["env"]
    # But the disable env propagates into the sandbox so the in-container
    # reranker honours it too — otherwise the tool would try to load
    # weights with no HF cache or network and fail loudly.
    assert kw["env"]["LAB_RAG_RERANKER"] == "none"
    # And there's no point sending the rerank URL when reranking is off.
    assert "LAB_RAG_RERANKER_URL" not in kw["env"]


def test_agent_run_skips_hf_cache_when_no_kb_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LAB_HF_CACHE_ROOT", str(tmp_path / "hfc"))
    monkeypatch.setattr("lab.core.settings._settings", None, raising=False)
    _run_agent_run(monkeypatch, [{"name": "fs_read"}, {"name": "shell_exec"}], reranker_env=None)
    kw = _FakeSandbox.last_kwargs
    assert kw["hf_cache_mount"] is None
