"""Structured logging wrapper around structlog.

Phase 16.1 module. Exposes a tiny stable surface so the rest of the lab can
adopt structured logging incrementally:

    from lab.observability.log import configure_logging, get_logger, bind_run_context

    configure_logging()
    log = get_logger(__name__)
    bind_run_context(run_id="...", experiment_slug="exp-001")
    log.info("cell_started", model="ollama/qwen3:8b", task="kb-rag-q1")

Design notes:

* ``configure_logging`` is idempotent. The first call wires structlog +
  the stdlib root logger; subsequent calls are no-ops. This lets every
  entry point (CLI, sweep main, unit tests) safely call it without fear
  of double-wiring.

* JSON vs console rendering: auto-detected from ``sys.stderr.isatty()``.
  In a TTY we use the colourised console renderer (good for local dev);
  off a TTY we emit JSON lines (good for CI, ``journalctl``, joining
  with Prometheus). Override with ``json_mode=True/False``.

* Run context is held in ``contextvars`` so it survives async boundaries
  inside sweep cells. ``bind_run_context`` / ``clear_run_context`` are
  the only way to attach / detach the bindings.

* User-facing CLI output (rich tables, summary panels) stays on
  ``rich.console.Console`` — structlog is for diagnostics, not for the
  human-facing UI.
"""

from __future__ import annotations

import contextvars
import logging
import sys
from typing import Any

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
)

__all__ = [
    "bind_run_context",
    "clear_run_context",
    "configure_logging",
    "get_logger",
    "is_configured",
]


# Module-level guard so ``configure_logging`` is idempotent.
_configured: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_lab_logging_configured", default=False
)

# Global flag — contextvars don't propagate across configure → get_logger
# in different threads. We need a plain module-level flag for that.
_GLOBAL_CONFIGURED = False


def is_configured() -> bool:
    """Return True iff configure_logging has already wired structlog."""

    return _GLOBAL_CONFIGURED


def configure_logging(
    *,
    level: str = "INFO",
    json_mode: bool | None = None,
) -> None:
    """Configure structlog once at app start.

    Args:
        level: stdlib log level name (DEBUG/INFO/WARNING/ERROR).
        json_mode: ``None`` (default) → JSON when stderr is not a TTY,
            colourised console otherwise. ``True`` / ``False`` overrides.

    Idempotent: a second call is a no-op. We never re-wire structlog
    once it's been configured — callers that need to change the level
    mid-run should use ``logging.getLogger().setLevel(...)`` directly.
    """

    global _GLOBAL_CONFIGURED  # noqa: PLW0603 - module-level config flag by design
    if _GLOBAL_CONFIGURED:
        return

    if json_mode is None:
        json_mode = not sys.stderr.isatty()

    # Stdlib root: send everything to stderr at the requested level. structlog
    # ultimately writes via stdlib logging.
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        stream=sys.stderr,
        force=True,
    )

    shared_processors: list[Any] = [
        merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any
    if json_mode:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    _GLOBAL_CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog BoundLogger.

    If ``configure_logging`` has not been called, we configure with
    defaults first so a bare ``get_logger(__name__).info(...)`` still
    works — useful for ad-hoc scripts and tests.
    """

    if not _GLOBAL_CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_run_context(
    run_id: str,
    experiment_slug: str | None = None,
    **extra: Any,
) -> None:
    """Bind run-level context for subsequent log calls.

    Stored in ``contextvars`` so it survives async boundaries and stays
    bound for the lifetime of the current async task / thread.

    Args:
        run_id: cell-level identifier.
        experiment_slug: experiment slug (optional).
        **extra: any additional bindings (model, task, seed, ...).
    """

    bindings: dict[str, Any] = {"run_id": run_id}
    if experiment_slug is not None:
        bindings["experiment_slug"] = experiment_slug
    bindings.update(extra)
    bind_contextvars(**bindings)


def clear_run_context() -> None:
    """Drop all context bindings made via ``bind_run_context``."""

    clear_contextvars()
