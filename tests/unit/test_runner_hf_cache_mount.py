"""``_execute_agent_cell`` plumbs ``hf_cache_mount`` into the Sandbox.

Twin of ``test_cli_hf_cache_mount``: the sweep runner must mount the HF
cache for any cell whose task could trigger the Phase 7 reranker, so the
~1.5 GB Qwen3-Reranker weights aren't re-downloaded per cell.
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


def _stub_inspect_and_logwriter(monkeypatch: pytest.MonkeyPatch) -> None:
    # Adapter / logwriter / inspect_eval — return the minimum shape the
    # runner's post-eval block can read without exploding.

    class _Sample:
        metadata: ClassVar[dict[str, Any]] = {"lab_agent": {}}
        model_usage: ClassVar[dict[str, Any]] = {}

    class _Log:
        samples: ClassVar[list[_Sample]] = [_Sample()]

    monkeypatch.setattr(
        "lab.inspect_bridge.adapter.lab_task_to_inspect",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr("inspect_ai.eval", lambda *a, **k: [_Log()])
    monkeypatch.setattr(
        "lab.inspect_bridge.logwriter.write_run_from_inspect_log",
        lambda *a, **k: "s3://lab/traj/test",
    )

    # _insert_run touches Postgres — neuter it.
    from lab.sweep import runner as runner_mod

    monkeypatch.setattr(runner_mod, "_insert_run", lambda **k: None)


def _make_cell(tools: list[dict[str, Any]] | None) -> Any:
    from lab.sweep import runner as runner_mod

    return runner_mod.Cell(
        run_id="run-1",
        experiment_id=1,
        experiment_slug="EXP",
        model_id=2,
        model_litellm_id="m",
        model_backend="cloud",  # avoid gpu_lease path
        task_id=3,
        task_slug="t",
        task_payload={
            "input": "hi",
            "max_turns": 3,
            "tool_budget": 2,
            "tools": tools,
            "sandbox": {"network": "none"},
        },
        config=runner_mod.RunConfig(name="c"),
        config_hash="h",
        seed=0,
    )


def _run(monkeypatch: pytest.MonkeyPatch, cell: Any) -> None:
    from lab.sweep import runner as runner_mod

    _FakeSandbox.last_kwargs = {}
    monkeypatch.setattr("lab.agent.sandbox.Sandbox", _FakeSandbox)
    _stub_inspect_and_logwriter(monkeypatch)
    monkeypatch.setattr("lab.core.settings._settings", None, raising=False)
    runner_mod._execute_agent_cell(cell=cell, manifest_sha="deadbeef", timeout=10)


def test_runner_mounts_hf_cache_when_kb_query_and_reranker_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LAB_HF_CACHE_ROOT", str(tmp_path / "hfc"))
    monkeypatch.delenv("LAB_RAG_RERANKER", raising=False)
    cell = _make_cell([{"name": "kb_query"}])
    _run(monkeypatch, cell)
    kw = _FakeSandbox.last_kwargs
    assert kw["hf_cache_mount"] == tmp_path / "hfc"
    assert kw["hf_cache_target"] == "/hf-cache"
    assert (tmp_path / "hfc").is_dir()
    assert kw["env"]["HF_HOME"] == "/hf-cache"
    assert kw["env"]["TRANSFORMERS_CACHE"] == "/hf-cache/transformers"
    assert kw["env"]["HF_HUB_OFFLINE"] == "1"
    assert kw["env"]["TRANSFORMERS_OFFLINE"] == "1"
    # Phase 7.1: route in-sandbox reranks to the host-side service.
    assert kw["env"]["LAB_RAG_RERANKER_URL"] == "http://host.containers.internal:8401"


def test_runner_skips_hf_cache_when_reranker_disabled_via_sandbox_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runner consults env.LAB_RAG_RERANKER (the value plumbed into the
    sandbox), not the host's environment — sweep configs disable the
    reranker via task.sandbox.env."""

    monkeypatch.setenv("LAB_HF_CACHE_ROOT", str(tmp_path / "hfc"))
    # Make the host enable the reranker but the task disable it.
    monkeypatch.setenv("LAB_RAG_RERANKER", "Qwen/Qwen3-Reranker-0.6B")

    from lab.sweep import runner as runner_mod

    cell = runner_mod.Cell(
        run_id="run-1",
        experiment_id=1,
        experiment_slug="EXP",
        model_id=2,
        model_litellm_id="m",
        model_backend="cloud",
        task_id=3,
        task_slug="t",
        task_payload={
            "input": "hi",
            "max_turns": 3,
            "tool_budget": 2,
            "tools": [{"name": "kb_query"}],
            "sandbox": {"network": "none", "env": {"LAB_RAG_RERANKER": "none"}},
        },
        config=runner_mod.RunConfig(name="c"),
        config_hash="h",
        seed=0,
    )
    _run(monkeypatch, cell)
    assert _FakeSandbox.last_kwargs["hf_cache_mount"] is None


def test_runner_skips_hf_cache_when_no_kb_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LAB_HF_CACHE_ROOT", str(tmp_path / "hfc"))
    cell = _make_cell([{"name": "fs_read"}])
    _run(monkeypatch, cell)
    assert _FakeSandbox.last_kwargs["hf_cache_mount"] is None
