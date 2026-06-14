"""NS-2 Code Maintainer v0 (charter NS-2, ADR-012 / ADR-013) — the lab's first
WRITE-path Lab Agent Runtime caller.

A thin caller of the Lab Agent Runtime (lab.core.agent_runtime.run_agent),
mirroring the analyst (lab.analyst) and synthesizer (lab.synthesizer) thin-caller
pattern: it supplies a system prompt, a tool list, an authorizer, and a return
shape, and delegates the bounded tool-use loop, the audit, and the ADR-013
side-effect gate to run_agent.

Unlike NS-1/NS-3 (strictly read-only), the maintainer MUTATES code: it reads the
failing source, edits files, runs the tests, and iterates until they pass. It is
therefore the FIRST real consumer of the ADR-013 WRITE path.

WRITE-TOOL BACKEND (v0)
-----------------------
In-process Tool ABI impls (NOT the sandboxed FastMCP `lab.agent.tools`
MCP-subprocess bridge — that podman/MCP backend is the ADR-012 "#13 future"
seam). Every tool is confined to a single configurable WORKSPACE directory that
the caller guarantees is a git-tracked scratch repo, so every mutation is a
diff-able, revertible change. This is exactly ADR-013's definition of
`write_local`: "reversible local mutation in a git-tracked workspace". The tools
reject any path that escapes the workspace, and `mtn_run` only executes a fixed
allowlist of commands (pytest / ruff / python), never arbitrary shell.

AUTHORIZATION (ADR-013)
-----------------------
Because the maintainer operates autonomously inside a reversible git-tracked
workspace, `maintain` builds an `AuthzPolicy` that AUTO-GRANTS `write_local` for
actor="maintainer" (the `grants={("maintainer", "write_local")}` set — exercising
authz's explicit operator-grant path, the deny-by-default override that lets an
earnable class resolve to `allow` without going through the ratchet). `read`
auto-allows by the ADR-013 default. `irreversible` is NEVER granted in v0 (the
policy refuses it even if present in `grants`), so a send/publish/delete attempt
is denied and audited — the structural property NS-2 depends on.

EVAL SIGNAL (objective, no judge)
---------------------------------
Success is GROUND TRUTH, not a model's self-report and not an LLM judge: the run
SUCCEEDS iff the workspace's test command exits 0 (returncode == 0) on its final
invocation during the run. `maintain` reads this back from run_agent's
tool_results — the last `mtn_run` result's returncode — and reports it as
`passed`. tests-pass = the objective signal (charter NS-2 / ADR-013 §1: writes
are reversible and the success criterion is the suite going green).
"""

from __future__ import annotations

from typing import Any

from lab.core.agent_runtime import run_agent
from lab.core.authz import AuthzPolicy
from lab.core.settings import get_settings
from lab.maintainer_tools import (
    MAINTAINER_ACTOR,
    build_tools,
    mtn_run,
)


#: The maintainer's auto-grant: write_local is granted for the maintainer actor
#: (ADR-013 explicit operator-grant). irreversible is intentionally absent — the
#: policy never auto-grants it in v0 regardless.
def _maintainer_authorizer() -> AuthzPolicy:
    """ADR-013 policy that auto-grants `write_local` to the maintainer actor and
    nothing else. `read` auto-allows by default; `irreversible` stays
    require_approval (and with the default fail-closed approver, effectively
    denied) — never auto-granted in v0."""
    return AuthzPolicy(grants={(MAINTAINER_ACTOR, "write_local")})


_SYSTEM = """You are the lab's code maintainer. You are given ONE scoped change
in a git-tracked scratch workspace. Make the workspace's tests pass.

Workflow:
1. Read the failing code and the failing test with mtn_read.
2. Edit the source with mtn_write to fix the failure. Write the WHOLE file
   contents each time (mtn_write overwrites).
3. Run the tests with mtn_run (e.g. mtn_run("pytest -q")). Read the output.
4. If the tests still fail, iterate: read, edit, re-run. If they pass, STOP with
   a short summary of what you changed. Do not keep editing once green.

You may ONLY touch files inside the workspace; path escapes are rejected. You may
ONLY run pytest / ruff / python via mtn_run; nothing else. You cannot send,
publish, delete, or spend — those are not your tools and would be refused."""


def _passed_from_results(results: list[dict[str, Any]]) -> bool:
    """OBJECTIVE eval signal: the run succeeded iff the LAST mtn_run during the
    run reported returncode == 0. This is ground truth (the suite actually went
    green), not a judge and not the model's self-report."""
    for r in reversed(results):
        if r.get("name") != "mtn_run":
            continue
        res = r.get("result")
        if isinstance(res, dict):
            return res.get("returncode") == 0
        return False
    return False


def _tool_calls_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A compact, audit-friendly view of every tool call the run made: tool name
    plus its path/cmd argument (not the file bodies)."""
    out: list[dict[str, Any]] = []
    for r in results:
        args = r.get("args") or {}
        out.append(
            {
                "name": r.get("name"),
                "path": args.get("path"),
                "cmd": args.get("cmd"),
            }
        )
    return out


def _diff_summary(*, workspace: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    """A summary of the mutations the run made: the set of files SUCCESSFULLY written
    (from the mtn_write calls that actually landed) and whether the final test run passed. Reversibility is
    guaranteed by the workspace being git-tracked (ADR-013 write_local)."""
    paths: set[str] = set()
    for r in results:
        if r.get("name") != "mtn_write":
            continue
        # only count writes that actually landed: a rejected write (path escape,
        # etc.) returns {"error": ...} with no "bytes_written" and must NOT
        # appear in the diff summary.
        res = r.get("result")
        if not isinstance(res, dict) or "bytes_written" not in res:
            continue
        path = (r.get("args") or {}).get("path")
        if isinstance(path, str) and path:
            paths.add(path)
    written = sorted(paths)
    return {
        "workspace": workspace,
        "files_written": written,
        "n_writes": len(written),
        "final_tests_pass": _passed_from_results(results),
    }


def maintain(
    *,
    task: str,
    workspace: str,
    test_cmd: str = "pytest -q",
    model: str = "qwen3-4b-ft-toolcall-q4-latest",
    max_tool_calls: int = 24,
    timeout: int = 120,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Drive the maintainer LAR over one scoped change and return the outcome.

    The maintainer is wired as actor="maintainer" with an `AuthzPolicy` that
    auto-grants `write_local` in this workspace (ADR-013 explicit grant), so
    mtn_write / mtn_run execute without per-call approval while `irreversible`
    stays denied. The tools are path-confined to `workspace` and `mtn_run` is
    allowlisted to pytest/ruff/python.

    OBJECTIVE EVAL (cited in module docstring): `passed` is ground truth — the
    last `mtn_run` returncode during the run — i.e. the workspace's tests
    actually went green. No LLM judge.

    Returns {task, passed, tool_calls, stop, diff_summary}.
    """
    settings = get_settings()
    res = run_agent(
        settings=settings,
        litellm_key=settings.litellm_key,
        model=model,
        system=_SYSTEM,
        user=(
            f"Workspace: {workspace}\nScoped change: {task}\n\n"
            f"Make the tests pass. Verify with mtn_run({test_cmd!r})."
        ),
        tools=build_tools(workspace),
        actor=MAINTAINER_ACTOR,
        authorizer=_maintainer_authorizer(),
        max_turns=max_tool_calls,
        max_tool_calls=max_tool_calls,
        timeout=timeout,
        num_ctx=num_ctx,
    )
    return {
        "task": task,
        "passed": _passed_from_results(res.tool_results),
        "tool_calls": res.tool_calls,
        "stop": res.stop_reason,
        "diff_summary": _diff_summary(workspace=workspace, results=res.tool_results),
    }


__all__ = ["build_tools", "maintain", "mtn_run"]
