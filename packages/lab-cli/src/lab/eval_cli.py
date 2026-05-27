"""`lab eval prompts` subcommand group.

Phase 16.4 follow-up: wire the prompt registry + prompt-test runner into
the CLI. The actual implementations live in
:mod:`lab.eval.prompts` and :mod:`lab.eval.prompt_tests`; this module is
the typer surface.

Subcommands:

* ``lab eval prompts list``              — list prompts under ``prompts/library/``.
* ``lab eval prompts test <prompt_id>``  — run ``prompts/tests/<prompt_id>.test.md`` against a model.
* ``lab eval prompts validate``          — validate every prompt file's frontmatter.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from lab.eval.prompt_tests import (
    PromptTestResult,
    load_prompt_test_file,
    run_prompt_test_file,
)
from lab.eval.prompts import (
    DEFAULT_PROMPTS_ROOT,
    PromptNotFoundError,
    PromptRegistry,
)

__all__ = ["prompts_app"]

prompts_app = typer.Typer(
    help="Canonical prompt registry + golden tests (prompts/library, prompts/tests)",
    no_args_is_help=True,
)

_console = Console()

# Default model for `prompts test`: glm-5.1-cloud is the most reliable
# prompt-following model in the local LiteLLM proxy line-up — it stays
# closer to the assertion shape than the qwen3 variants and it's free at
# Phase 16. Override with --model on the command line when needed.
_DEFAULT_TEST_MODEL = "glm-5.1-cloud"

# Mirror the prompt_tests default; importing the constant avoids drift
# if the upstream default ever moves.
from lab.eval.prompt_tests import DEFAULT_TESTS_ROOT as _DEFAULT_TESTS_ROOT  # noqa: E402


def _resolve_test_file(prompt_id: str, root: Path) -> Path:
    """Find the canonical ``<prompt_id>.test.md`` file for ``prompt_id``.

    The on-disk convention uses hyphenated filenames (``rag-grounded-v1.test.md``)
    while prompt ids are snake_case (``rag_grounded_v1``). We try both
    forms so callers can use either convention.
    """
    candidates = [
        root / f"{prompt_id}.test.md",
        root / f"{prompt_id.replace('_', '-')}.test.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no test file for prompt {prompt_id!r}; searched: "
        f"{', '.join(str(c) for c in candidates)}"
    )


def _make_litellm_caller(model: str) -> Any:
    """Build a :class:`ModelCaller` that hits the LiteLLM proxy.

    Lazy-imports lab.core to keep CLI startup fast; the import only fires
    when a user actually runs ``prompts test``.
    """
    from lab.core.llm import call_litellm_chat
    from lab.core.settings import get_settings

    settings = get_settings()
    key = settings.litellm_key or ""
    if not key:
        candidate = Path("/data/lab/services/litellm-master-key")
        if candidate.exists():
            key = candidate.read_text().strip()

    def _call(*, system: str, user: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        resp, _latency = call_litellm_chat(
            settings=settings,
            litellm_key=key,
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )
        choice = (resp.get("choices") or [{}])[0]
        msg = (choice or {}).get("message") or {}
        text = msg.get("content") or ""
        # Tool calls (if any) — surface their names for the matcher.
        tool_calls_raw = msg.get("tool_calls") or []
        tool_calls: list[str] = []
        for tc in tool_calls_raw:
            fn = (tc or {}).get("function") or {}
            name = fn.get("name")
            if name:
                tool_calls.append(str(name))
        return {"response_text": str(text), "tool_calls": tool_calls}

    return _call


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@prompts_app.command("list")
def prompts_list(
    root: Path = typer.Option(
        DEFAULT_PROMPTS_ROOT,
        "--root",
        help="Override the prompts/library directory to scan",
    ),
) -> None:
    """List every prompt registered under ``prompts/library/``."""
    registry = PromptRegistry(root=root)
    metas = registry.list()
    if not metas:
        _console.print(f"[yellow]no prompts found under[/] {root}")
        return
    table = Table("prompt_id", "version", "title", "tags", "path")
    for meta in metas:
        rel = str(meta.path)
        with contextlib.suppress(ValueError):
            rel = str(meta.path.relative_to(Path.cwd()))
        tags = ", ".join(meta.tags) if meta.tags else "-"
        table.add_row(meta.prompt_id, str(meta.version), meta.title, tags, rel)
    _console.print(table)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@prompts_app.command("validate")
def prompts_validate(
    root: Path = typer.Option(
        DEFAULT_PROMPTS_ROOT,
        "--root",
        help="Override the prompts/library directory to validate",
    ),
) -> None:
    """Validate every prompt file's frontmatter, surfacing parse errors.

    Walks ``root`` directly (rather than using PromptRegistry.list) so a
    single bad file doesn't mask the rest — each file's failure is
    reported independently, and the command exits non-zero on any failure.
    """
    from lab.eval.prompts import _load_one

    if not root.exists():
        _console.print(f"[red]prompts root does not exist[/]: {root}")
        raise typer.Exit(code=2)
    failures: list[tuple[Path, str]] = []
    n_ok = 0
    for path in sorted(root.glob("*.md")):
        try:
            _load_one(path)
        except Exception as exc:
            failures.append((path, f"{type(exc).__name__}: {exc}"))
        else:
            n_ok += 1
    if failures:
        table = Table("path", "error")
        for path, err in failures:
            table.add_row(str(path), err)
        _console.print(table)
        _console.print(f"[red]{len(failures)} invalid prompt(s)[/]; {n_ok} ok")
        raise typer.Exit(code=1)
    _console.print(f"[green]all {n_ok} prompt(s) valid[/]")


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------


def _summarise_results(results: list[PromptTestResult]) -> tuple[int, int]:
    """Return ``(n_passed, n_failed)`` for one file's results."""
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    return passed, failed


@prompts_app.command("test")
def prompts_test(
    prompt_id: str = typer.Argument(..., help="Prompt id (e.g. 'agent_system_v1')"),
    model: str = typer.Option(
        _DEFAULT_TEST_MODEL,
        "--model",
        help="LiteLLM model id for the run (default: glm-5.1-cloud)",
    ),
    n: int = typer.Option(
        1,
        "--n",
        help="Run each test N times (default 1). Useful for surfacing flakiness.",
    ),
    tests_root: Path = typer.Option(
        _DEFAULT_TESTS_ROOT,
        "--tests-root",
        help="Override the prompts/tests directory to load from",
    ),
    prompts_root: Path = typer.Option(
        DEFAULT_PROMPTS_ROOT,
        "--prompts-root",
        help="Override the prompts/library directory to resolve from",
    ),
) -> None:
    """Run ``prompts/tests/<prompt_id>.test.md`` against ``model``.

    Each test case fires ``n`` times; we report pass/fail per (test, attempt)
    pair and a per-test aggregate at the end. The exit code is non-zero
    if any (test, attempt) failed.
    """
    if n < 1:
        _console.print(f"[red]--n must be >= 1; got {n}")
        raise typer.Exit(code=2)
    try:
        path = _resolve_test_file(prompt_id, tests_root)
    except FileNotFoundError as exc:
        _console.print(f"[red]{exc}")
        raise typer.Exit(code=2) from None
    test_file = load_prompt_test_file(path)

    registry = PromptRegistry(root=prompts_root)
    try:
        registry.get(test_file.prompt_id)
    except PromptNotFoundError as exc:
        _console.print(f"[red]prompt registry error[/]: {exc}")
        raise typer.Exit(code=2) from None

    caller = _make_litellm_caller(model)

    _console.print(
        f"[bold]prompts test[/] file={path} prompt={test_file.prompt_id} "
        f"model={model} n={n} cases={len(test_file.tests)}"
    )

    table = Table("test", "attempt", "passed", "failures")
    total_pass = 0
    total_fail = 0
    for attempt in range(1, n + 1):
        results = run_prompt_test_file(test_file, caller=caller, registry=registry)
        p, f = _summarise_results(results)
        total_pass += p
        total_fail += f
        for r in results:
            failures = "; ".join(r.failures) if r.failures else "-"
            table.add_row(
                r.test_name,
                str(attempt),
                "[green]yes[/]" if r.passed else "[red]no[/]",
                failures[:120],
            )
    _console.print(table)
    _console.print(f"passed={total_pass} failed={total_fail}")
    if total_fail:
        raise typer.Exit(code=1)
