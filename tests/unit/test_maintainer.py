"""NS-2 Code Maintainer v0 (charter NS-2, ADR-012 / ADR-013) — mocked unit tests.

No GPU and no live LLM. The strategy mirrors test_analyst:

  * A real tmp git-tracked workspace is created with a TRIVIALLY FAILING test
    (a module with a bug + a pytest that asserts the fixed behavior).
  * run_agent is MOCKED to a SCRIPTED trajectory (mtn_read -> mtn_write the fix
    -> mtn_run pytest -> done) that drives the REAL in-process maintainer tools
    against the tmp workspace — exercising the actual write + run path, not the
    LLM.

What we prove (the wiring, not the model):
  * mtn_write actually wrote the fixed file and the suite goes green
    (returncode 0 -> the OBJECTIVE eval signal -> maintain(...)["passed"] True).
  * The ADR-013 gate ALLOWS write_local for actor="maintainer" via the granting
    AuthzPolicy, while leaving the default fail-closed approver in place.
  * Path-escape is rejected by the tools (no filesystem mutation outside root).
  * An IRREVERSIBLE attempt is DENIED by the same policy (never auto in v0).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from lab.core.authz import AuthzPolicy
from lab.maintainer import maintain
from lab.maintainer_tools import (
    MAINTAINER_ACTOR,
    PathEscape,
    build_tools,
    make_read,
    make_run,
    make_write,
)

# --------------------------------------------------------------------------- #
# tmp git workspace with a trivially failing test                              #
# --------------------------------------------------------------------------- #

_BUGGY_SRC = "def add(a, b):\n    return a - b  # BUG: should be +\n"
_FIXED_SRC = "def add(a, b):\n    return a + b\n"
_TEST_SRC = "from calc import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n"


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "scratch"
    ws.mkdir()
    (ws / "calc.py").write_text(_BUGGY_SRC, encoding="utf-8")
    (ws / "test_calc.py").write_text(_TEST_SRC, encoding="utf-8")
    # git-tracked scratch repo -> write_local mutations are reversible (ADR-013)
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=ws,
        check=True,
    )
    return ws


# --------------------------------------------------------------------------- #
# Tool-level unit tests (real impls, no agent)                                 #
# --------------------------------------------------------------------------- #


def test_build_tools_side_effects_and_confinement() -> None:
    tools = {t.name: t for t in build_tools("/tmp/ws")}
    assert set(tools) == {"mtn_read", "mtn_write", "mtn_run"}
    assert tools["mtn_read"].side_effect == "read"
    assert tools["mtn_write"].side_effect == "write_local"
    assert tools["mtn_run"].side_effect == "write_local"


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    read = make_read(ws)
    write = make_write(ws)
    # absolute + traversal escapes are refused; nothing is written outside root
    assert "error" in write("../evil.py", "x")
    assert "error" in read("../calc.py")
    assert "error" in write("/etc/evil", "x")
    assert not (tmp_path / "evil.py").exists()
    # the low-level resolver raises for an escape, returns for an in-root path
    import pytest

    from lab.maintainer_tools import _resolve

    with pytest.raises(PathEscape):
        _resolve(ws, "../../escape")


def test_mtn_run_command_allowlist(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    run = make_run(ws)
    bad = run("rm -rf /")
    assert "error" in bad
    assert "not allowed" in bad["error"]
    # buggy suite fails first (objective signal works in both directions)
    failing = run("python -m pytest -q")
    assert failing["returncode"] != 0


def test_mtn_write_then_tests_go_green(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    write = make_write(ws)
    run = make_run(ws)
    write("calc.py", _FIXED_SRC)
    assert (ws / "calc.py").read_text() == _FIXED_SRC
    passed = run("python -m pytest -q")
    assert passed["returncode"] == 0


# --------------------------------------------------------------------------- #
# Authz wiring (ADR-013): the granting policy allows write_local for the        #
# maintainer, denies irreversible, and the default approver stays fail-closed.  #
# --------------------------------------------------------------------------- #


def test_authz_grants_write_local_denies_irreversible() -> None:
    policy = AuthzPolicy(grants={(MAINTAINER_ACTOR, "write_local")})
    # write_local AUTO-ALLOWED for the maintainer (the grant path)
    assert policy.decide(MAINTAINER_ACTOR, "mtn_write", "write_local", "fs_write") == "allow"
    assert policy.decide(MAINTAINER_ACTOR, "mtn_run", "write_local", "run_cmd") == "allow"
    # read is allowed by ADR-013 default
    assert policy.decide(MAINTAINER_ACTOR, "mtn_read", "read", "fs_read") == "allow"
    # irreversible is NEVER auto in v0, even if granted -> require_approval
    leaky = AuthzPolicy(grants={(MAINTAINER_ACTOR, "irreversible")})
    assert (
        leaky.decide(MAINTAINER_ACTOR, "send_email", "irreversible", "send") == "require_approval"
    )
    # a DIFFERENT actor gets no write grant -> require_approval (deny-by-default)
    assert policy.decide("stranger", "mtn_write", "write_local", "fs_write") == "require_approval"


# --------------------------------------------------------------------------- #
# Golden eval: maintain() with run_agent MOCKED to a scripted trajectory that   #
# runs the REAL tools through the REAL ADR-013 gate against the tmp workspace.   #
# --------------------------------------------------------------------------- #


class _FakeAgentResult:
    def __init__(self, messages: list[dict[str, Any]], tool_results: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.tool_calls = len(tool_results)
        self.tool_results = tool_results
        self.stop_reason = "stop"


def _dispatch_through_gate(
    *, tool: Any, args: dict[str, Any], actor: str, authorizer: AuthzPolicy
) -> dict[str, Any]:
    """Re-create run_agent's enforcement for ONE call: consult the authorizer,
    execute only on `allow`, otherwise return the gate's refusal. Proves the
    tool result is gated, not called blind."""
    decision = authorizer.decide(actor, tool.name, tool.side_effect, tool.capability)
    if decision != "allow":
        return {"error": f"gate:{decision}"}
    impl_result: dict[str, Any] = tool.impl(**args)
    return impl_result


def test_maintain_golden_writes_fix_and_goes_green(tmp_path: Path, monkeypatch: Any) -> None:
    """Mock run_agent to a scripted mtn_read -> mtn_write -> mtn_run trajectory
    that invokes the REAL tools THROUGH the REAL granting authorizer against the
    tmp git workspace. Assert: the file was written (gate ALLOWED write_local),
    the tests went green (objective pass signal), and a path-escape write is
    rejected by the tool. No LLM, no GPU."""
    ws = _make_workspace(tmp_path)

    def fake_run_agent(**kwargs: Any) -> _FakeAgentResult:
        # wiring assertions: maintainer actor + a granting authorizer is passed
        assert kwargs["actor"] == MAINTAINER_ACTOR
        authorizer = kwargs["authorizer"]
        assert isinstance(authorizer, AuthzPolicy)
        assert (MAINTAINER_ACTOR, "write_local") in authorizer.grants
        tools = {t.name: t for t in kwargs["tools"]}

        results: list[dict[str, Any]] = []

        # 1. read the buggy source (read -> allow)
        a1 = {"path": "calc.py"}
        r1 = _dispatch_through_gate(
            tool=tools["mtn_read"], args=a1, actor=MAINTAINER_ACTOR, authorizer=authorizer
        )
        results.append({"name": "mtn_read", "args": a1, "result": r1})
        assert "BUG" in r1["content"]

        # path-escape attempt is rejected by the tool itself (still gate-allowed,
        # but the impl refuses to leave the workspace)
        a_esc = {"path": "../evil.py", "content": "x"}
        r_esc = _dispatch_through_gate(
            tool=tools["mtn_write"], args=a_esc, actor=MAINTAINER_ACTOR, authorizer=authorizer
        )
        results.append({"name": "mtn_write", "args": a_esc, "result": r_esc})
        assert "error" in r_esc

        # 2. write the fix (write_local -> gate ALLOWS via the grant)
        a2 = {"path": "calc.py", "content": _FIXED_SRC}
        r2 = _dispatch_through_gate(
            tool=tools["mtn_write"], args=a2, actor=MAINTAINER_ACTOR, authorizer=authorizer
        )
        results.append({"name": "mtn_write", "args": a2, "result": r2})
        assert "bytes_written" in r2  # not a gate refusal

        # 3. run the tests (write_local -> allow); objective green signal
        a3 = {"cmd": "python -m pytest -q"}
        r3 = _dispatch_through_gate(
            tool=tools["mtn_run"], args=a3, actor=MAINTAINER_ACTOR, authorizer=authorizer
        )
        results.append({"name": "mtn_run", "args": a3, "result": r3})

        messages = [
            {"role": "system", "content": kwargs["system"]},
            {"role": "user", "content": kwargs["user"]},
            {"role": "assistant", "content": "Fixed add() to use +; tests pass."},
        ]
        return _FakeAgentResult(messages, results)

    monkeypatch.setattr("lab.maintainer.run_agent", fake_run_agent)

    out = maintain(task="fix add() so test_add passes", workspace=str(ws))

    # the fix actually landed on disk (gate allowed the write)
    assert (ws / "calc.py").read_text() == _FIXED_SRC
    # objective eval signal: tests went green
    assert out["passed"] is True
    assert out["stop"] == "stop"
    assert out["tool_calls"] == 4
    # diff summary reflects the single real write (the escaped write never landed)
    assert out["diff_summary"]["files_written"] == ["calc.py"]
    assert out["diff_summary"]["final_tests_pass"] is True


def test_maintain_reports_failure_when_suite_stays_red(tmp_path: Path, monkeypatch: Any) -> None:
    """If the run never makes the suite green, the objective signal reports
    passed=False — the maintainer cannot fake success (no judge, tests are
    ground truth)."""
    ws = _make_workspace(tmp_path)

    def fake_run_agent(**kwargs: Any) -> _FakeAgentResult:
        authorizer = kwargs["authorizer"]
        tools = {t.name: t for t in kwargs["tools"]}
        # run the tests WITHOUT fixing the bug -> still red
        a = {"cmd": "python -m pytest -q"}
        r = _dispatch_through_gate(
            tool=tools["mtn_run"], args=a, actor=MAINTAINER_ACTOR, authorizer=authorizer
        )
        results = [{"name": "mtn_run", "args": a, "result": r}]
        messages = [{"role": "assistant", "content": "could not fix it"}]
        return _FakeAgentResult(messages, results)

    monkeypatch.setattr("lab.maintainer.run_agent", fake_run_agent)

    out = maintain(task="try (and fail) to fix", workspace=str(ws))
    assert out["passed"] is False
    assert out["diff_summary"]["files_written"] == []
