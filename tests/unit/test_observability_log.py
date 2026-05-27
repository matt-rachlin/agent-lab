"""Unit tests for lab.observability.log.

Covers:
* JSON mode renders one valid JSON object per line.
* TTY/console mode does not emit JSON.
* run-context bound via ``bind_run_context`` shows up on subsequent log
  calls, including across an async boundary.
* configure_logging is idempotent (second call doesn't re-wire).
* get_logger works without an explicit configure (lazy default).
* level filtering: DEBUG suppressed under INFO.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any

import pytest

import lab.observability.log as log_module


@pytest.fixture(autouse=True)
def _reset_log_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a fresh wiring per test (configure_logging is idempotent)."""

    monkeypatch.setattr(log_module, "_GLOBAL_CONFIGURED", False)
    # Drop any leftover contextvars from a prior test.
    log_module.clear_run_context()
    yield
    log_module.clear_run_context()


def _capture_stderr(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Redirect sys.stderr to a StringIO and return it."""

    buf = io.StringIO()
    monkeypatch.setattr("sys.stderr", buf)
    return buf


def test_json_mode_emits_valid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = _capture_stderr(monkeypatch)
    log_module.configure_logging(json_mode=True, level="INFO")
    log = log_module.get_logger("t.json")
    log.info("hello", k="v", n=42)
    line = buf.getvalue().splitlines()[-1]
    obj: Any = json.loads(line)
    assert obj["event"] == "hello"
    assert obj["k"] == "v"
    assert obj["n"] == 42
    assert obj["level"] == "info"
    assert "timestamp" in obj


def test_console_mode_does_not_emit_json(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = _capture_stderr(monkeypatch)
    log_module.configure_logging(json_mode=False, level="INFO")
    log = log_module.get_logger("t.console")
    log.info("hello", k="v")
    output = buf.getvalue()
    # Console renderer is human-formatted; ``hello`` appears but the
    # output is not a JSON document.
    assert "hello" in output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output.splitlines()[-1])


def test_bind_run_context_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = _capture_stderr(monkeypatch)
    log_module.configure_logging(json_mode=True, level="INFO")
    log = log_module.get_logger("t.bind")
    log_module.bind_run_context(run_id="r-123", experiment_slug="exp-x", model="ollama/qwen3:8b")
    log.info("cell_started")
    log.info("cell_done")
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("{")]
    assert lines, "expected JSON log lines"
    for ln in lines:
        obj = json.loads(ln)
        assert obj["run_id"] == "r-123"
        assert obj["experiment_slug"] == "exp-x"
        assert obj["model"] == "ollama/qwen3:8b"


def test_clear_run_context_drops_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = _capture_stderr(monkeypatch)
    log_module.configure_logging(json_mode=True, level="INFO")
    log = log_module.get_logger("t.clear")
    log_module.bind_run_context(run_id="r-a")
    log.info("first")
    log_module.clear_run_context()
    log.info("second")
    lines = [json.loads(ln) for ln in buf.getvalue().splitlines() if ln.startswith("{")]
    assert lines[-2]["run_id"] == "r-a"
    assert "run_id" not in lines[-1]


def test_bind_survives_async_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = _capture_stderr(monkeypatch)
    log_module.configure_logging(json_mode=True, level="INFO")
    log = log_module.get_logger("t.async")

    async def inner() -> None:
        log.info("inner")

    async def main() -> None:
        log_module.bind_run_context(run_id="r-async", experiment_slug="async-exp")
        await inner()

    asyncio.run(main())
    lines = [json.loads(ln) for ln in buf.getvalue().splitlines() if ln.startswith("{")]
    found = [ln for ln in lines if ln.get("event") == "inner"]
    assert found, "did not capture inner async log"
    assert found[0]["run_id"] == "r-async"
    assert found[0]["experiment_slug"] == "async-exp"


def test_configure_logging_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_stderr(monkeypatch)
    log_module.configure_logging(json_mode=True, level="INFO")
    assert log_module.is_configured()

    # Second call attempts WARNING level — should NOT take effect because
    # the configure is a no-op once wired.
    log_module.configure_logging(json_mode=True, level="WARNING")
    # The stdlib root remains at INFO from the first call.
    assert logging.getLogger().level == logging.INFO


def test_level_filtering_drops_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = _capture_stderr(monkeypatch)
    log_module.configure_logging(json_mode=True, level="INFO")
    log = log_module.get_logger("t.level")
    log.debug("ignored")
    log.info("kept")
    output = buf.getvalue()
    assert "kept" in output
    assert "ignored" not in output


def test_lazy_default_configures_on_first_get_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_stderr(monkeypatch)
    assert not log_module.is_configured()
    log_module.get_logger("t.lazy")
    assert log_module.is_configured()
