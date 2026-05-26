"""6f follow-up to 6e: bridge uses sys.executable, not a bare 'python' literal.

6e flagged that `tools.py` shelled out to `python -m <module>` which breaks
the schema-discovery path when `lab agent run` is invoked outside an
activated venv. 6f swaps that for `sys.executable` (with fallback).
"""

from __future__ import annotations

import sys

from lab.inspect_bridge.tools import _host_python


def test_host_python_returns_sys_executable() -> None:
    assert _host_python() == sys.executable


def test_host_python_falls_back_when_sys_executable_empty(monkeypatch) -> None:
    """When sys.executable is unset (rare), we resolve `python3` from PATH."""
    monkeypatch.setattr(sys, "executable", "")
    result = _host_python()
    # The fallback must yield SOMETHING runnable — either a resolved path
    # or the literal string for the last-ditch attempt.
    assert result in {"python3", "python"} or result.endswith(("python3", "python"))
