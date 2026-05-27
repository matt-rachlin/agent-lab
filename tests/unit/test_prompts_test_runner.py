"""Tests for lab.eval.prompt_tests — golden test parsing + mocked runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lab.eval.prompt_tests import (
    PromptTest,
    PromptTestFile,
    load_prompt_test_dir,
    load_prompt_test_file,
    run_prompt_test,
    run_prompt_test_file,
)
from lab.eval.prompts import PromptRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _write_test_file(
    path: Path,
    *,
    prompt_id: str,
    tests_yaml: str,
    title: str = "T",
) -> Path:
    """Write a test file using the body-YAML format.

    Frontmatter stays strictly doc-meta-compliant; the test payload
    lives in a fenced ```yaml block in the body.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""---
doc_id: prompt-test-{prompt_id.replace("_", "-")}
title: {title}
zone: lab
kind: card
status: active
owner: m
created: 2026-05-27
last_updated: 2026-05-27
---

# Test

```yaml
prompt_id: {prompt_id}
tests:
{tests_yaml}
```
""",
        encoding="utf-8",
    )
    return path


def test_loader_parses_basic_file(tmp_path: Path) -> None:
    p = _write_test_file(
        tmp_path / "agent.test.md",
        prompt_id="agent_system_v1",
        tests_yaml=(
            '  - name: "tool_use_intent"\n'
            '    input: "Read /workspace/note.txt"\n'
            '    expected_tool_calls: ["fs_read"]\n'
            '    expected_response_substring: "the secret is"\n'
            '  - name: "no_tool_when_unneeded"\n'
            '    input: "What is 2+2?"\n'
            "    expected_tool_calls: []\n"
        ),
    )
    f = load_prompt_test_file(p)
    assert f.prompt_id == "agent_system_v1"
    assert len(f.tests) == 2
    assert f.tests[0].name == "tool_use_intent"
    assert f.tests[0].expected_tool_calls == ["fs_read"]
    assert f.tests[0].expected_response_substring == "the secret is"
    assert f.tests[1].expected_tool_calls == []
    assert f.tests[1].expected_response_substring is None


def test_loader_rejects_missing_prompt_id(tmp_path: Path) -> None:
    """A test file whose yaml block lacks prompt_id is rejected."""
    bad = tmp_path / "bad.test.md"
    bad.write_text(
        "---\n"
        "doc_id: prompt-test-x\n"
        "title: x\n"
        "zone: lab\n"
        "kind: card\n"
        "status: active\n"
        "owner: m\n"
        "created: 2026-05-27\n"
        "last_updated: 2026-05-27\n"
        "---\n\n"
        "```yaml\n"
        "tests:\n"
        "  - name: x\n"
        '    input: "y"\n'
        "```\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prompt_id"):
        load_prompt_test_file(bad)


def test_loader_rejects_missing_yaml_block(tmp_path: Path) -> None:
    """A test file without any fenced yaml block is rejected."""
    bad = tmp_path / "bad.test.md"
    bad.write_text(
        "---\n"
        "doc_id: prompt-test-x\n"
        "title: x\n"
        "zone: lab\n"
        "kind: card\n"
        "status: active\n"
        "owner: m\n"
        "created: 2026-05-27\n"
        "last_updated: 2026-05-27\n"
        "---\n\n"
        "no yaml here, just text\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="yaml"):
        load_prompt_test_file(bad)


def test_loader_rejects_test_missing_name(tmp_path: Path) -> None:
    p = _write_test_file(
        tmp_path / "x.test.md",
        prompt_id="x_v1",
        tests_yaml='  - input: "missing name"\n',
    )
    with pytest.raises(ValueError, match="name"):
        load_prompt_test_file(p)


def test_loader_rejects_test_missing_input(tmp_path: Path) -> None:
    p = _write_test_file(
        tmp_path / "x.test.md",
        prompt_id="x_v1",
        tests_yaml='  - name: "missing input"\n',
    )
    with pytest.raises(ValueError, match="input"):
        load_prompt_test_file(p)


def test_loader_handles_empty_tests_list(tmp_path: Path) -> None:
    """tests: [] in the body yaml block is a valid (zero-case) file."""
    p = tmp_path / "x.test.md"
    p.write_text(
        "---\n"
        "doc_id: prompt-test-x\n"
        "title: x\n"
        "zone: lab\n"
        "kind: card\n"
        "status: active\n"
        "owner: m\n"
        "created: 2026-05-27\n"
        "last_updated: 2026-05-27\n"
        "---\n\n"
        "```yaml\n"
        "prompt_id: x_v1\n"
        "tests: []\n"
        "```\n",
        encoding="utf-8",
    )
    f = load_prompt_test_file(p)
    assert f.tests == []
    assert f.prompt_id == "x_v1"


def test_load_dir_iterates_test_files(tmp_path: Path) -> None:
    _write_test_file(
        tmp_path / "a.test.md",
        prompt_id="a_v1",
        tests_yaml='  - name: "t"\n    input: "i"\n',
    )
    _write_test_file(
        tmp_path / "b.test.md",
        prompt_id="b_v1",
        tests_yaml='  - name: "t"\n    input: "i"\n',
    )
    files = load_prompt_test_dir(tmp_path)
    assert {f.prompt_id for f in files} == {"a_v1", "b_v1"}


def test_load_dir_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_prompt_test_dir(tmp_path / "no_such_dir") == []


# ---------------------------------------------------------------------------
# Mocked runner
# ---------------------------------------------------------------------------


def _mock_caller(response: str = "ok", tool_calls: list[str] | None = None) -> Any:
    """Build a ModelCaller-compatible callable returning fixed values."""

    def _call(*, system: str, user: str) -> dict[str, Any]:
        # `system` and `user` are unused in the mock but match the protocol.
        return {
            "response_text": response,
            "tool_calls": list(tool_calls or []),
        }

    return _call


def test_runner_passes_when_tool_calls_match() -> None:
    t = PromptTest(
        name="x",
        input="read foo",
        expected_tool_calls=["fs_read"],
    )
    res = run_prompt_test(
        t,
        prompt_body="be a tool-user",
        caller=_mock_caller(response="done", tool_calls=["fs_read"]),
    )
    assert res.passed is True
    assert res.failures == []


def test_runner_fails_when_tool_calls_differ() -> None:
    t = PromptTest(
        name="x",
        input="read foo",
        expected_tool_calls=["fs_read"],
    )
    res = run_prompt_test(
        t,
        prompt_body="be a tool-user",
        caller=_mock_caller(response="done", tool_calls=["fs_write"]),
    )
    assert res.passed is False
    assert any("tool-call mismatch" in f for f in res.failures)


def test_runner_checks_response_substring() -> None:
    t = PromptTest(
        name="x",
        input="hello",
        expected_tool_calls=[],
        expected_response_substring="the secret is",
    )
    # Substring present → pass
    res_good = run_prompt_test(
        t,
        prompt_body="...",
        caller=_mock_caller(response="ok the secret is 42", tool_calls=[]),
    )
    assert res_good.passed is True
    # Substring missing → fail
    res_bad = run_prompt_test(
        t,
        prompt_body="...",
        caller=_mock_caller(response="ok 42", tool_calls=[]),
    )
    assert res_bad.passed is False
    assert any("substring" in f for f in res_bad.failures)


def test_runner_loads_prompt_body_via_registry(tmp_path: Path) -> None:
    """run_prompt_test_file pulls the body from the registry by prompt_id."""
    # Build a prompts library fixture.
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "x-v1.md").write_text(
        "---\n"
        "doc_id: prompt-x-v1\n"
        "title: X\n"
        "zone: lab\n"
        "kind: prompt\n"
        "status: active\n"
        "owner: m\n"
        "created: 2026-05-27\n"
        "last_updated: 2026-05-27\n"
        "---\n\nbe excellent\n",
        encoding="utf-8",
    )
    reg = PromptRegistry(root=lib)

    captured: dict[str, Any] = {}

    def _capture_caller(*, system: str, user: str) -> dict[str, Any]:
        captured["system"] = system
        captured["user"] = user
        return {"response_text": "ok", "tool_calls": []}

    pt = PromptTestFile(
        prompt_id="x_v1",
        path=Path("/dev/null"),
        title="t",
        tests=[PromptTest(name="t", input="hello", expected_tool_calls=[])],
    )
    results = run_prompt_test_file(pt, caller=_capture_caller, registry=reg)
    assert len(results) == 1
    assert results[0].passed is True
    # The runner must pass the prompt body in as `system`.
    assert "be excellent" in captured["system"]
    assert captured["user"] == "hello"


# ---------------------------------------------------------------------------
# Repo-wide sanity: the three Phase 16.4.3 test files load
# ---------------------------------------------------------------------------


def test_repo_prompt_test_files_load() -> None:
    """The three first-batch prompt test files exist and parse."""
    files = load_prompt_test_dir(REPO_ROOT / "prompts" / "tests")
    prompt_ids = {f.prompt_id for f in files}
    expected = {"agent_system_v1", "tool_use_system_v1", "rag_grounded_v1"}
    missing = expected - prompt_ids
    assert not missing, f"missing prompt test files for: {sorted(missing)}"
    # Each file must have at least one test case.
    for f in files:
        assert f.tests, f"{f.path} has no test cases"
