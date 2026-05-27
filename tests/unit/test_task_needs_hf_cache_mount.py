"""Unit tests for the ``task_needs_hf_cache_mount`` heuristic.

Mirrors the kb_mount heuristic but adds the LAB_RAG_RERANKER guard: the
hf-cache bind mount is only meaningful when the cross-encoder reranker is
actually going to load. With ``LAB_RAG_RERANKER=none`` the rerank pass is
skipped entirely, so we drop the mount to keep the sandbox surface lean.
"""

from __future__ import annotations

import pytest
from lab.agent.tools import TOOLS_NEEDING_HF_CACHE, task_needs_hf_cache_mount


def test_no_tools_returns_false() -> None:
    assert task_needs_hf_cache_mount(None) is False
    assert task_needs_hf_cache_mount([]) is False


def test_only_unrelated_tools_returns_false() -> None:
    assert (
        task_needs_hf_cache_mount(
            [{"name": "fs_read"}, {"name": "shell_exec"}],
            reranker_env="",
        )
        is False
    )


def test_kb_query_with_reranker_default_returns_true() -> None:
    # Unset env (passed as None) means default model — reranker enabled.
    assert (
        task_needs_hf_cache_mount(
            [{"name": "kb_query"}],
            reranker_env="",
        )
        is True
    )


def test_kb_query_with_reranker_disabled_returns_false() -> None:
    assert (
        task_needs_hf_cache_mount(
            [{"name": "kb_query"}],
            reranker_env="none",
        )
        is False
    )
    # Case-insensitive sentinel; whitespace is stripped.
    assert (
        task_needs_hf_cache_mount(
            [{"name": "kb_query"}],
            reranker_env="  NONE  ",
        )
        is False
    )


def test_kb_query_with_explicit_model_returns_true() -> None:
    assert (
        task_needs_hf_cache_mount(
            [{"name": "kb_query"}],
            reranker_env="BAAI/bge-reranker-v2-m3",
        )
        is True
    )


def test_plain_string_tool_names_supported() -> None:
    """The heuristic mirrors task_needs_kb_mount and tolerates bare strings."""
    assert task_needs_hf_cache_mount(["kb_query"], reranker_env="") is True
    assert task_needs_hf_cache_mount(["fs_read"], reranker_env="") is False


def test_env_var_is_read_when_override_not_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAB_RAG_RERANKER", "none")
    assert task_needs_hf_cache_mount([{"name": "kb_query"}]) is False
    monkeypatch.setenv("LAB_RAG_RERANKER", "Qwen/Qwen3-Reranker-0.6B")
    assert task_needs_hf_cache_mount([{"name": "kb_query"}]) is True
    monkeypatch.delenv("LAB_RAG_RERANKER", raising=False)
    assert task_needs_hf_cache_mount([{"name": "kb_query"}]) is True


def test_tools_needing_hf_cache_is_subset_of_kb_mount() -> None:
    """Anything that may trigger the reranker also reads the KB."""
    from lab.agent.tools import TOOLS_NEEDING_KB_MOUNT

    assert TOOLS_NEEDING_HF_CACHE <= TOOLS_NEEDING_KB_MOUNT
