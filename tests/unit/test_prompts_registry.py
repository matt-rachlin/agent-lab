"""Tests for lab.eval.prompts — prompt registry loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.eval.prompts import PromptNotFoundError, PromptRegistry


def _write_prompt(
    root: Path,
    *,
    doc_id: str,
    title: str,
    body: str,
    tags: list[str] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    # Derive filename from doc_id (drop the leading 'prompt-')
    name = doc_id[len("prompt-") :] if doc_id.startswith("prompt-") else doc_id
    path = root / f"{name}.md"
    tags_line = ""
    if tags:
        tags_line = f"tags: {tags}\n"
    text = (
        "---\n"
        f"doc_id: {doc_id}\n"
        f"title: {title}\n"
        "zone: lab\n"
        "kind: prompt\n"
        "status: active\n"
        "owner: m\n"
        "created: 2026-05-27\n"
        "last_updated: 2026-05-27\n"
        f"{tags_line}"
        "---\n\n"
        f"{body}\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def test_empty_registry_lists_nothing(tmp_path: Path) -> None:
    reg = PromptRegistry(root=tmp_path / "empty")
    assert reg.list() == []
    assert not reg.has("anything")


def test_load_and_get_single_prompt(tmp_path: Path) -> None:
    _write_prompt(
        tmp_path,
        doc_id="prompt-agent-system-v1",
        title="Agent system",
        body="You are a careful research assistant.",
        tags=["lab", "prompt", "agent"],
    )
    reg = PromptRegistry(root=tmp_path)
    assert reg.has("agent_system_v1")
    body = reg.get("agent_system_v1")
    assert body.startswith("You are a careful research assistant")
    assert "research assistant" in body
    meta = reg.get_meta("agent_system_v1")
    assert meta.version == 1
    assert meta.title == "Agent system"
    assert "lab" in meta.tags


def test_versioning_picks_highest_by_default(tmp_path: Path) -> None:
    _write_prompt(
        tmp_path,
        doc_id="prompt-judge-v1",
        title="Judge v1",
        body="rubric v1",
    )
    _write_prompt(
        tmp_path,
        doc_id="prompt-judge-v2",
        title="Judge v2",
        body="rubric v2",
    )
    reg = PromptRegistry(root=tmp_path)
    # Base-id lookup returns latest.
    assert reg.get("judge").strip() == "rubric v2"
    assert reg.get("judge", version=1).strip() == "rubric v1"
    assert reg.get("judge", version=2).strip() == "rubric v2"
    # Versioned-id lookup pins automatically.
    assert reg.get("judge_v1").strip() == "rubric v1"
    assert reg.get("judge_v2").strip() == "rubric v2"


def test_has_supports_both_base_and_versioned_ids(tmp_path: Path) -> None:
    _write_prompt(tmp_path, doc_id="prompt-judge-v1", title="J", body="x")
    reg = PromptRegistry(root=tmp_path)
    assert reg.has("judge") is True
    assert reg.has("judge_v1") is True
    assert reg.has("judge_v2") is False  # version doesn't exist
    assert reg.has("missing") is False


def test_frontmatter_version_mismatch_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "judge-v1.md"
    tmp_path.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        "---\n"
        "doc_id: prompt-judge-v1\n"
        "title: J\n"
        "zone: lab\n"
        "kind: prompt\n"
        "status: active\n"
        "owner: m\n"
        "created: 2026-05-27\n"
        "last_updated: 2026-05-27\n"
        "version: 99\n"  # disagrees with -v1
        "---\n\nbody",
        encoding="utf-8",
    )
    reg = PromptRegistry(root=tmp_path)
    with pytest.raises(ValueError, match="version"):
        reg.list()


def test_get_missing_prompt_raises(tmp_path: Path) -> None:
    reg = PromptRegistry(root=tmp_path)
    with pytest.raises(PromptNotFoundError):
        reg.get("nonexistent")


def test_get_missing_version_raises(tmp_path: Path) -> None:
    _write_prompt(
        tmp_path,
        doc_id="prompt-judge-v1",
        title="J",
        body="x",
    )
    reg = PromptRegistry(root=tmp_path)
    with pytest.raises(PromptNotFoundError):
        reg.get("judge", version=99)


def test_list_sorted_by_id_then_version(tmp_path: Path) -> None:
    _write_prompt(tmp_path, doc_id="prompt-b-tool-v1", title="B", body="b1")
    _write_prompt(tmp_path, doc_id="prompt-a-sys-v2", title="A2", body="a2")
    _write_prompt(tmp_path, doc_id="prompt-a-sys-v1", title="A1", body="a1")
    reg = PromptRegistry(root=tmp_path)
    listing = reg.list()
    assert [m.base_id for m in listing] == [
        "a_sys",
        "a_sys",
        "b_tool",
    ]
    assert [m.prompt_id for m in listing] == [
        "a_sys_v1",
        "a_sys_v2",
        "b_tool_v1",
    ]
    assert [m.version for m in listing] == [1, 2, 1]


def test_bad_doc_id_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "weird.md"
    tmp_path.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        "---\n"
        "doc_id: not-a-prompt-id\n"
        "title: x\n"
        "zone: lab\n"
        "kind: prompt\n"
        "status: active\n"
        "owner: m\n"
        "created: 2026-05-27\n"
        "last_updated: 2026-05-27\n"
        "---\n\nbody",
        encoding="utf-8",
    )
    reg = PromptRegistry(root=tmp_path)
    with pytest.raises(ValueError, match="doc_id"):
        reg.list()


def test_wrong_kind_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "spec_v1.md"
    tmp_path.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        "---\n"
        "doc_id: prompt-spec-v1\n"
        "title: x\n"
        "zone: lab\n"
        "kind: spec\n"
        "status: active\n"
        "owner: m\n"
        "created: 2026-05-27\n"
        "last_updated: 2026-05-27\n"
        "---\n\nbody",
        encoding="utf-8",
    )
    reg = PromptRegistry(root=tmp_path)
    with pytest.raises(ValueError, match="kind"):
        reg.list()


def test_reload_picks_up_disk_changes(tmp_path: Path) -> None:
    _write_prompt(tmp_path, doc_id="prompt-x-v1", title="X1", body="old")
    reg = PromptRegistry(root=tmp_path)
    assert reg.get("x").strip() == "old"
    # Mutate the file and reload.
    (tmp_path / "x-v1.md").write_text(
        "---\n"
        "doc_id: prompt-x-v1\n"
        "title: X1\n"
        "zone: lab\n"
        "kind: prompt\n"
        "status: active\n"
        "owner: m\n"
        "created: 2026-05-27\n"
        "last_updated: 2026-05-27\n"
        "---\n\nnew\n",
        encoding="utf-8",
    )
    # Without reload, cache still holds 'old'.
    assert reg.get("x").strip() == "old"
    reg.reload()
    assert reg.get("x").strip() == "new"
