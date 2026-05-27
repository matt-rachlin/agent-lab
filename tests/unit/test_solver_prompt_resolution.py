"""Tests for system-prompt resolution in the solver / adapter pipeline.

These tests exercise the *resolution* surface that the adapter (and, for
the single-turn path, the sweep runner) use to turn ``task.system_prompt_id``
into a real system message. They do not run a full Inspect eval — that's
covered by the integration suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lab.eval.prompts import PromptNotFoundError, PromptRegistry
from lab.inspect_bridge.adapter import _resolve_system_prompt
from lab.sweep.runner import _build_messages
from lab.tasks.registry import Task


def _write_prompt(root: Path, *, doc_id: str, body: str) -> Path:
    """Stage a minimal doc-meta-compliant prompt file."""
    root.mkdir(parents=True, exist_ok=True)
    name = doc_id[len("prompt-") :] if doc_id.startswith("prompt-") else doc_id
    path = root / f"{name}.md"
    path.write_text(
        "---\n"
        f"doc_id: {doc_id}\n"
        "title: T\n"
        "zone: lab\n"
        "kind: prompt\n"
        "status: active\n"
        "owner: m\n"
        "created: 2026-05-27\n"
        "last_updated: 2026-05-27\n"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def _task(**overrides: Any) -> Task:
    base: dict[str, Any] = {
        "suite": "t",
        "slug": "s",
        "input": "hi",
    }
    base.update(overrides)
    return Task.model_validate(base)


# ---------------------------------------------------------------------------
# _resolve_system_prompt
# ---------------------------------------------------------------------------


def test_resolve_uses_registry_when_id_set(tmp_path: Path) -> None:
    _write_prompt(tmp_path, doc_id="prompt-foo-v1", body="be helpful")
    reg = PromptRegistry(root=tmp_path)
    body, used = _resolve_system_prompt(_task(system_prompt_id="foo_v1"), registry=reg)
    # PromptRegistry preserves the trailing newline of the body — assert
    # on a substring to stay robust to that detail.
    assert body is not None
    assert "be helpful" in body
    assert used == "foo_v1"


def test_resolve_falls_back_to_inline_system(tmp_path: Path) -> None:
    reg = PromptRegistry(root=tmp_path)
    body, used = _resolve_system_prompt(_task(system="legacy inline"), registry=reg)
    assert body == "legacy inline"
    assert used is None


def test_resolve_returns_none_when_neither_set(tmp_path: Path) -> None:
    reg = PromptRegistry(root=tmp_path)
    body, used = _resolve_system_prompt(_task(), registry=reg)
    assert body is None
    assert used is None


def test_resolve_raises_on_missing_prompt(tmp_path: Path) -> None:
    """system_prompt_id pointing to a missing prompt → loud error at build time."""
    reg = PromptRegistry(root=tmp_path)  # empty registry
    with pytest.raises(PromptNotFoundError) as excinfo:
        _resolve_system_prompt(_task(system_prompt_id="nope_v1"), registry=reg)
    msg = str(excinfo.value)
    assert "nope_v1" in msg
    assert "t/s" in msg  # suite/slug in the error for triage


def test_resolve_rejects_both_set_defense_in_depth(tmp_path: Path) -> None:
    """Even if the Task validator was bypassed, _resolve refuses both fields.

    We can't construct a `Task` with both set (the validator blocks it),
    so we use a lightweight stand-in object with the same attributes —
    the resolver should treat ``both set`` as an error case regardless.
    """

    class _FauxTask:
        suite = "t"
        slug = "s"
        system = "inline"
        system_prompt_id = "foo_v1"

    reg = PromptRegistry(root=tmp_path)
    with pytest.raises(ValueError, match="both 'system' and 'system_prompt_id'"):
        _resolve_system_prompt(_FauxTask(), registry=reg)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _build_messages (single-turn sweep path)
# ---------------------------------------------------------------------------


def test_build_messages_resolves_system_prompt_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_build_messages goes through PromptRegistry when system_prompt_id is set."""
    _write_prompt(tmp_path, doc_id="prompt-foo-v1", body="be helpful")

    # Monkey-patch the registry so it scans our tmp dir instead of the
    # canonical prompts/library — the sweep runner imports it lazily.
    from lab.eval import prompts as prompts_mod

    monkeypatch.setattr(prompts_mod, "DEFAULT_PROMPTS_ROOT", tmp_path)

    payload = {"input": "hi", "system_prompt_id": "foo_v1"}
    msgs = _build_messages(payload)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "be helpful" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_build_messages_inline_system_wins_over_prompt_id(tmp_path: Path) -> None:
    """If `system` is set, we never touch the registry (defensive)."""
    payload = {
        "input": "hi",
        "system": "inline-wins",
        # If the resolver fired this would error (no registry rooted at tmp).
        "system_prompt_id": "nope_v1",
    }
    msgs = _build_messages(payload)
    assert msgs[0] == {"role": "system", "content": "inline-wins"}
