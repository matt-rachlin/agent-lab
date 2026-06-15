"""Tests for `lab eval prompts {list,validate,test}` CLI commands.

These tests run the typer app against an isolated prompts/tests layout
in ``tmp_path``; they mock the LiteLLM caller so no network round-trips
fire. The aim is the wiring (argument parsing, exit codes, registry
resolution) — the heavy lifting is unit-tested elsewhere in
``test_prompts_registry.py`` and ``test_prompts_test_runner.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from lab.eval_cli import prompts_app


def _write_prompt(root: Path, *, doc_id: str, body: str) -> Path:
    """Write a minimal doc-meta-compliant prompt file."""
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


def _write_test_file(root: Path, *, prompt_id: str, tests_yaml: str) -> Path:
    """Write a minimal doc-meta-compliant test file."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{prompt_id.replace('_', '-')}.test.md"
    path.write_text(
        f"""---
doc_id: prompt-test-{prompt_id.replace("_", "-")}
title: T
zone: lab
kind: card
status: active
owner: m
created: 2026-05-27
last_updated: 2026-05-27
---

```yaml
prompt_id: {prompt_id}
tests:
{tests_yaml}
```
""",
        encoding="utf-8",
    )
    return path


def test_prompts_list_shows_registered_prompts(tmp_path: Path) -> None:
    """`list --root <tmp>` walks the dir and prints one row per prompt."""
    root = tmp_path / "library"
    _write_prompt(root, doc_id="prompt-foo-v1", body="hello")
    _write_prompt(root, doc_id="prompt-bar-v2", body="world")

    runner = CliRunner()
    result = runner.invoke(prompts_app, ["list", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "foo_v1" in result.output
    assert "bar_v2" in result.output


def test_prompts_list_empty_dir_is_not_an_error(tmp_path: Path) -> None:
    """An empty/non-existent root prints a warning but exits 0."""
    runner = CliRunner()
    result = runner.invoke(prompts_app, ["list", "--root", str(tmp_path / "missing")])
    assert result.exit_code == 0, result.output
    assert "no prompts found" in result.output


def test_prompts_validate_reports_errors(tmp_path: Path) -> None:
    """`validate` exits non-zero when a prompt has bad frontmatter."""
    root = tmp_path / "library"
    _write_prompt(root, doc_id="prompt-good-v1", body="hi")
    # A file with `kind: card` (not `prompt`) — _load_one rejects it.
    bad = root / "bad-v1.md"
    bad.write_text(
        "---\n"
        "doc_id: prompt-bad-v1\n"
        "title: bad\n"
        "zone: lab\n"
        "kind: card\n"
        "status: active\n"
        "owner: m\n"
        "created: 2026-05-27\n"
        "last_updated: 2026-05-27\n"
        "---\n\nbody\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(prompts_app, ["validate", "--root", str(root)])
    assert result.exit_code == 1, result.output
    assert "invalid" in result.output


def test_prompts_validate_clean_root(tmp_path: Path) -> None:
    """All-good root → exit 0 and friendly success line."""
    root = tmp_path / "library"
    _write_prompt(root, doc_id="prompt-a-v1", body="hi")
    runner = CliRunner()
    result = runner.invoke(prompts_app, ["validate", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output


def test_prompts_test_runs_and_succeeds(tmp_path: Path) -> None:
    """`test <prompt_id>` resolves the test file, calls the model, prints results."""
    prompts_root = tmp_path / "library"
    tests_root = tmp_path / "tests"
    _write_prompt(prompts_root, doc_id="prompt-foo-v1", body="be helpful")
    _write_test_file(
        tests_root,
        prompt_id="foo_v1",
        tests_yaml=(
            '  - name: "happy"\n'
            '    input: "ping"\n'
            "    expected_tool_calls: []\n"
            '    expected_response_substring: "pong"\n'
        ),
    )

    def fake_caller(*, system: str, user: str) -> dict[str, Any]:
        # The system prompt should be the body we wrote above.
        assert "be helpful" in system
        return {"response_text": "pong from model", "tool_calls": []}

    with patch("lab.eval_cli._make_litellm_caller", return_value=fake_caller):
        runner = CliRunner()
        result = runner.invoke(
            prompts_app,
            [
                "test",
                "foo_v1",
                "--tests-root",
                str(tests_root),
                "--prompts-root",
                str(prompts_root),
                "--n",
                "1",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "happy" in result.output
    assert "passed=1" in result.output


def test_prompts_test_reports_failures_with_nonzero_exit(tmp_path: Path) -> None:
    """A failing assertion → table shows it AND the command exits non-zero."""
    prompts_root = tmp_path / "library"
    tests_root = tmp_path / "tests"
    _write_prompt(prompts_root, doc_id="prompt-foo-v1", body="be helpful")
    _write_test_file(
        tests_root,
        prompt_id="foo_v1",
        tests_yaml=(
            '  - name: "expects_pong"\n'
            '    input: "ping"\n'
            "    expected_tool_calls: []\n"
            '    expected_response_substring: "pong"\n'
        ),
    )

    def fake_caller(*, system: str, user: str) -> dict[str, Any]:
        return {"response_text": "no match here", "tool_calls": []}

    with patch("lab.eval_cli._make_litellm_caller", return_value=fake_caller):
        runner = CliRunner()
        result = runner.invoke(
            prompts_app,
            [
                "test",
                "foo_v1",
                "--tests-root",
                str(tests_root),
                "--prompts-root",
                str(prompts_root),
            ],
        )
    assert result.exit_code == 1, result.output
    assert "expects_pong" in result.output
    assert "failed=1" in result.output


def test_prompts_test_missing_file_exits_2(tmp_path: Path) -> None:
    """Asking for an unknown prompt_id → exit 2 (usage error)."""
    runner = CliRunner()
    result = runner.invoke(
        prompts_app,
        [
            "test",
            "nope_v1",
            "--tests-root",
            str(tmp_path / "missing-tests"),
            "--prompts-root",
            str(tmp_path / "missing-lib"),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "no test file" in result.output


def test_prompts_test_supports_n_repetitions(tmp_path: Path) -> None:
    """`--n 3` fires each test case three times."""
    prompts_root = tmp_path / "library"
    tests_root = tmp_path / "tests"
    _write_prompt(prompts_root, doc_id="prompt-foo-v1", body="be helpful")
    _write_test_file(
        tests_root,
        prompt_id="foo_v1",
        tests_yaml=('  - name: "t1"\n    input: "ping"\n    expected_tool_calls: []\n'),
    )

    call_count = {"n": 0}

    def fake_caller(*, system: str, user: str) -> dict[str, Any]:
        call_count["n"] += 1
        return {"response_text": "ok", "tool_calls": []}

    with patch("lab.eval_cli._make_litellm_caller", return_value=fake_caller):
        runner = CliRunner()
        result = runner.invoke(
            prompts_app,
            [
                "test",
                "foo_v1",
                "--tests-root",
                str(tests_root),
                "--prompts-root",
                str(prompts_root),
                "--n",
                "3",
            ],
        )
    assert result.exit_code == 0, result.output
    assert call_count["n"] == 3
    assert "passed=3" in result.output
