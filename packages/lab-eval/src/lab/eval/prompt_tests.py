"""Prompt golden-test runner.

A per-prompt golden test file lives at ``prompts/tests/<prompt_id>.test.md``.
Its frontmatter is strict doc-meta (``kind: card``); the test data lives
in the body inside a fenced ``yaml`` code block::

    ```yaml
    prompt_id: agent_system_v1
    tests:
      - name: tool_use_intent
        input: "Read /workspace/note.txt"
        expected_tool_calls: ["fs_read"]
        expected_response_substring: "the secret is"
      - name: no_tool_when_unneeded
        input: "What is 2+2?"
        expected_tool_calls: []
    ```

Each test entry has:

* ``name``        — display name
* ``input``       — user message to send
* ``expected_tool_calls`` — list of tool names expected (in order);
  empty list means "no tool calls"
* ``expected_response_substring`` — optional, must appear in the final
  response text

The runner loads the prompt body via :class:`PromptRegistry`, builds the
messages, dispatches to a model-call callable (real or mocked), then
checks the assertions.

Why a body YAML block rather than extra frontmatter keys? The doc-meta
schema (``m_cli.docs.schema``) is strict-mode and rejects unknown
fields. Keeping the frontmatter compliant means ``m docs scan`` and
``m docs lint`` accept these as ordinary cards. The test payload
travels in a fenced YAML block in the body so it stays machine-readable
without bending the global schema.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from lab.eval.prompts import PromptRegistry

__all__ = [
    "DEFAULT_TESTS_ROOT",
    "ModelCaller",
    "PromptTest",
    "PromptTestFile",
    "PromptTestResult",
    "load_prompt_test_dir",
    "load_prompt_test_file",
    "run_prompt_test",
    "run_prompt_test_file",
]


DEFAULT_TESTS_ROOT = Path("prompts/tests")

_FRONTMATTER_RE = re.compile(
    r"\A---\r?\n(?P<body>.*?)\r?\n---\r?\n?",
    re.DOTALL,
)

# Fenced YAML code block in the body that holds the test payload.
_YAML_FENCE_RE = re.compile(
    r"```ya?ml\s*\n(?P<payload>.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class PromptTest:
    """One golden test case parsed out of a ``.test.md`` file."""

    name: str
    input: str
    expected_tool_calls: list[str] = field(default_factory=list)
    expected_response_substring: str | None = None


@dataclass(frozen=True)
class PromptTestFile:
    """A whole ``.test.md`` file — metadata + ordered test cases."""

    prompt_id: str
    path: Path
    title: str
    tests: list[PromptTest]


@dataclass(frozen=True)
class PromptTestResult:
    """Outcome of running one :class:`PromptTest`."""

    test_name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    response_text: str = ""
    tool_calls: list[str] = field(default_factory=list)


class ModelCaller(Protocol):
    """Pluggable model-call interface used by the runner.

    A caller takes a system + user message pair and returns a dict with
    ``response_text`` (str) and ``tool_calls`` (list of str tool names).
    Real implementations talk to LiteLLM; unit tests pass a mock.
    """

    def __call__(
        self,
        *,
        system: str,
        user: str,
    ) -> dict[str, Any]: ...


def _parse_tests_list(raw: Any, path: Path) -> list[PromptTest]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: 'tests' must be a list, got {type(raw).__name__}")
    out: list[PromptTest] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: tests[{i}] must be a mapping, got {type(entry).__name__}")
        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"{path}: tests[{i}] missing 'name'")
        inp = entry.get("input")
        if not inp or not isinstance(inp, str):
            raise ValueError(f"{path}: tests[{i}] missing 'input'")
        tools_raw = entry.get("expected_tool_calls", [])
        if tools_raw is None:
            tools_raw = []
        if not isinstance(tools_raw, list):
            raise ValueError(f"{path}: tests[{i}] expected_tool_calls must be a list")
        expected_tools = [str(t) for t in tools_raw]
        substring = entry.get("expected_response_substring")
        if substring is not None and not isinstance(substring, str):
            raise ValueError(f"{path}: tests[{i}] expected_response_substring must be a string")
        out.append(
            PromptTest(
                name=name,
                input=inp,
                expected_tool_calls=expected_tools,
                expected_response_substring=substring,
            )
        )
    return out


def load_prompt_test_file(path: Path) -> PromptTestFile:
    """Parse one ``.test.md`` file.

    The frontmatter supplies the document title (and is strict doc-meta).
    The first fenced ``yaml`` block in the body holds the test payload
    (``prompt_id`` + ``tests`` list).
    """
    text = path.read_text(encoding="utf-8")
    fm = _FRONTMATTER_RE.match(text)
    if not fm:
        raise ValueError(f"{path}: missing YAML frontmatter")
    meta = yaml.safe_load(fm.group("body")) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: frontmatter is not a mapping")
    title = str(meta.get("title") or "")
    body = text[fm.end() :]

    yaml_match = _YAML_FENCE_RE.search(body)
    if not yaml_match:
        raise ValueError(f"{path}: missing fenced ```yaml block with prompt_id + tests")
    payload = yaml.safe_load(yaml_match.group("payload")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: yaml block must be a mapping")
    prompt_id = payload.get("prompt_id")
    if not prompt_id or not isinstance(prompt_id, str):
        raise ValueError(f"{path}: yaml block missing 'prompt_id'")
    if not title:
        title = prompt_id
    tests = _parse_tests_list(payload.get("tests"), path)
    return PromptTestFile(prompt_id=prompt_id, path=path, title=title, tests=tests)


def load_prompt_test_dir(root: Path | None = None) -> list[PromptTestFile]:
    """Load every ``*.test.md`` under ``root`` (default ``prompts/tests/``)."""
    base = root if root is not None else DEFAULT_TESTS_ROOT
    if not base.exists():
        return []
    files: list[PromptTestFile] = []
    for path in sorted(base.glob("*.test.md")):
        files.append(load_prompt_test_file(path))
    return files


def _check_tool_calls(expected: Sequence[str], actual: Sequence[str]) -> list[str]:
    """Return failure messages comparing expected vs actual tool-call names."""
    if list(expected) == list(actual):
        return []
    return [f"tool-call mismatch: expected={list(expected)!r}, actual={list(actual)!r}"]


def run_prompt_test(
    test: PromptTest,
    *,
    prompt_body: str,
    caller: ModelCaller,
) -> PromptTestResult:
    """Run a single test case against ``caller`` and return the outcome."""
    out = caller(system=prompt_body, user=test.input)
    response = str(out.get("response_text") or "")
    tool_calls = [str(t) for t in (out.get("tool_calls") or [])]
    failures: list[str] = []
    failures.extend(_check_tool_calls(test.expected_tool_calls, tool_calls))
    if (
        test.expected_response_substring is not None
        and test.expected_response_substring not in response
    ):
        failures.append(f"response missing substring {test.expected_response_substring!r}")
    return PromptTestResult(
        test_name=test.name,
        passed=not failures,
        failures=failures,
        response_text=response,
        tool_calls=tool_calls,
    )


def run_prompt_test_file(
    file: PromptTestFile,
    *,
    caller: ModelCaller,
    registry: PromptRegistry | None = None,
) -> list[PromptTestResult]:
    """Run every test in ``file`` against ``caller``.

    ``registry`` defaults to ``PromptRegistry()`` (canonical location).
    Tests resolve the prompt body via ``registry.get(file.prompt_id)``.
    """
    reg = registry if registry is not None else PromptRegistry()
    body = reg.get(file.prompt_id)
    return [run_prompt_test(t, prompt_body=body, caller=caller) for t in file.tests]
