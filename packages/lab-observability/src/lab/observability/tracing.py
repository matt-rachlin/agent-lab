"""OpenTelemetry span helpers for the lab.

Phase 16.2 module. We expose three things:

* ``configure_tracing(...)`` — wires the OTel SDK once at app start. The
  default exporter is OTLP gRPC at ``http://localhost:4317`` (Tempo's
  default port). Idempotent.

* ``span(name, **attrs)`` — context manager that opens a span and tags
  it with ``attrs``. Attributes are coerced to the allowed OTel set
  (str / int / float / bool); other values are stringified.

* ``current_span_attrs(**attrs)`` — bolt extra attributes onto the
  currently-active span. No-op if no span is active.

If tracing is never configured, ``span`` still works — it falls back to
the OTel ``NoOpTracer`` so the call-sites stay one shape. Tests get a
deterministic in-memory exporter by calling ``_install_test_exporter()``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.sampling import (
    ParentBased,
    Sampler,
    TraceIdRatioBased,
)
from opentelemetry.trace import Span, Status, StatusCode

__all__ = [
    "configure_tracing",
    "current_span_attrs",
    "get_tracer",
    "is_configured",
    "span",
]


_GLOBAL_CONFIGURED = False
_TRACER_NAME = "lab"

# Module-level TracerProvider override for tests. When set, get_tracer
# pulls its tracer from this provider instead of the global SDK
# TracerProvider (which OTel only lets you set once per process).
_TEST_PROVIDER: TracerProvider | None = None


def is_configured() -> bool:
    """Return True iff configure_tracing has wired the SDK."""

    return _GLOBAL_CONFIGURED


def _resolve_sample_ratio() -> float:
    raw = os.environ.get("LAB_OTEL_SAMPLE_RATIO", "1.0")
    try:
        ratio = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return ratio


def configure_tracing(
    *,
    exporter_url: str | None = None,
    service_name: str = "lab",
    sampler: Sampler | None = None,
) -> None:
    """Configure the OpenTelemetry SDK once at app start.

    Args:
        exporter_url: OTLP gRPC endpoint. Defaults to ``http://localhost:4317``
            (Tempo). Set to the literal string ``"none"`` to skip wiring
            an exporter (useful in unit tests where the in-memory exporter
            is installed via ``_install_test_exporter``).
        service_name: ``service.name`` resource attribute.
        sampler: optional custom sampler. Defaults to
            ``ParentBased(TraceIdRatioBased(LAB_OTEL_SAMPLE_RATIO))``.

    Idempotent: the second call is a no-op.
    """

    global _GLOBAL_CONFIGURED  # noqa: PLW0603 - module-level config flag by design
    if _GLOBAL_CONFIGURED:
        return

    if sampler is None:
        sampler = ParentBased(TraceIdRatioBased(_resolve_sample_ratio()))

    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": "lab",
        }
    )

    provider = TracerProvider(resource=resource, sampler=sampler)

    endpoint = exporter_url
    if endpoint is None:
        endpoint = os.environ.get("LAB_OTEL_EXPORTER_URL", "http://localhost:4317")

    if endpoint and endpoint.lower() != "none":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            # The OTLP exporter is optional at runtime; if missing we keep
            # the provider wired but with no exporter so spans are still
            # created (and visible to test exporters) but never shipped.
            pass

    trace.set_tracer_provider(provider)
    _GLOBAL_CONFIGURED = True


def _install_test_exporter() -> Any:
    """Wire an InMemorySpanExporter for unit tests.

    Returns the exporter so tests can inspect finished spans. Installs a
    local TracerProvider that overrides the lab's get_tracer lookup — we
    never touch the OTel global provider here because that is process-
    wide and only settable once.
    """

    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    global _TEST_PROVIDER, _GLOBAL_CONFIGURED  # noqa: PLW0603 - test-only module flags
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "lab-test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _TEST_PROVIDER = provider
    _GLOBAL_CONFIGURED = True
    return exporter


def _reset_for_tests() -> None:
    """Reset module-level state. Tests only."""

    global _GLOBAL_CONFIGURED, _TEST_PROVIDER  # noqa: PLW0603 - test-only module flags
    if _TEST_PROVIDER is not None:
        _TEST_PROVIDER.shutdown()
    _TEST_PROVIDER = None
    _GLOBAL_CONFIGURED = False


def get_tracer() -> trace.Tracer:
    """Return the lab's tracer (configures with defaults if unwired)."""

    if _TEST_PROVIDER is not None:
        return _TEST_PROVIDER.get_tracer(_TRACER_NAME)
    if not _GLOBAL_CONFIGURED:
        # Don't auto-export at import time — we don't want a stray import
        # in a unit test to spin up a real OTLP connection. We install a
        # provider with no exporter; spans are created but discarded.
        configure_tracing(exporter_url="none")
    return trace.get_tracer(_TRACER_NAME)


def _coerce_attr(value: Any) -> str | int | float | bool:
    """Coerce a value to an OTel-allowed attribute type."""

    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    return str(value)


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """Open a span named ``name`` with ``attrs`` as attributes.

    Errors raised inside the ``with`` block are recorded on the span as
    ``ERROR`` status with ``error.message`` / ``error.type`` attributes,
    then re-raised. Skipped attributes (``None`` values) are omitted so
    cardinality stays bounded.
    """

    tracer = get_tracer()
    coerced = {k: _coerce_attr(v) for k, v in attrs.items() if v is not None}
    with tracer.start_as_current_span(name, attributes=coerced) as s:
        try:
            yield s
        except BaseException as exc:
            s.set_status(Status(StatusCode.ERROR, str(exc)))
            s.set_attribute("error.type", type(exc).__name__)
            s.set_attribute("error.message", str(exc))
            raise


def current_span_attrs(**attrs: Any) -> None:
    """Attach extra attributes to the currently-active span.

    No-op if no span is active or the active span is the OTel no-op
    sentinel.
    """

    s = trace.get_current_span()
    if not s or not s.is_recording():
        return
    for key, value in attrs.items():
        if value is None:
            continue
        s.set_attribute(key, _coerce_attr(value))
