"""Unit tests for lab.observability.tracing.

We use the OTel ``InMemorySpanExporter`` to capture spans deterministically.
Covers:
* Spans are created with the expected name + attributes.
* Span nesting produces a parent → child relationship.
* Error inside a ``span`` block tags the span with error.type / error.message
  and ERROR status, and re-raises.
* ``current_span_attrs`` attaches to the active span.
* ``current_span_attrs`` outside any span is a no-op.
* ``span`` with no exporter configured does not raise (falls back to
  the lab's default no-exporter provider).
* Sampling ratio env var is parsed and clamped.
"""

from __future__ import annotations

import pytest
from opentelemetry.trace import StatusCode

import lab.observability.tracing as tracing_module


@pytest.fixture
def exporter() -> object:
    """Install the in-memory exporter for one test, then reset."""

    tracing_module._reset_for_tests()
    ex = tracing_module._install_test_exporter()
    yield ex
    tracing_module._reset_for_tests()


def test_span_creates_span_with_attributes(exporter: object) -> None:
    with tracing_module.span("cell", run_id="r-1", model="qwen3:8b", seed=42):
        pass
    spans = exporter.get_finished_spans()  # type: ignore[attr-defined]
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "cell"
    assert s.attributes["run_id"] == "r-1"
    assert s.attributes["model"] == "qwen3:8b"
    assert s.attributes["seed"] == 42


def test_span_nesting_parent_child(exporter: object) -> None:
    with tracing_module.span("outer"), tracing_module.span("inner"):
        pass
    spans = exporter.get_finished_spans()  # type: ignore[attr-defined]
    by_name = {s.name: s for s in spans}
    inner = by_name["inner"]
    outer = by_name["outer"]
    # inner's parent span id matches outer's span id
    assert inner.parent is not None
    assert inner.parent.span_id == outer.context.span_id


def test_error_in_span_tags_and_reraises(exporter: object) -> None:
    with pytest.raises(ValueError, match="boom"), tracing_module.span("cell"):
        raise ValueError("boom")
    spans = exporter.get_finished_spans()  # type: ignore[attr-defined]
    assert len(spans) == 1
    s = spans[0]
    assert s.status.status_code == StatusCode.ERROR
    assert s.attributes["error.type"] == "ValueError"
    assert s.attributes["error.message"] == "boom"


def test_current_span_attrs_attaches_to_active(exporter: object) -> None:
    with tracing_module.span("cell"):
        tracing_module.current_span_attrs(extra_label="abc", count=7)
    spans = exporter.get_finished_spans()  # type: ignore[attr-defined]
    attrs = spans[0].attributes
    assert attrs["extra_label"] == "abc"
    assert attrs["count"] == 7


def test_current_span_attrs_no_active_is_noop(exporter: object) -> None:
    # Should not raise even with no active span.
    tracing_module.current_span_attrs(foo="bar")
    assert exporter.get_finished_spans() == ()  # type: ignore[attr-defined]


def test_span_without_configure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling span() with no configured exporter falls back cleanly."""

    tracing_module._reset_for_tests()
    # With no test exporter installed, span() should still work — the
    # implementation lazy-configures with no exporter on first use.
    with tracing_module.span("x"):
        pass
    # We cannot inspect spans (no exporter); the test just verifies no
    # exception is raised.
    tracing_module._reset_for_tests()


def test_sample_ratio_env_var_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_OTEL_SAMPLE_RATIO", "2.5")
    assert tracing_module._resolve_sample_ratio() == 1.0
    monkeypatch.setenv("LAB_OTEL_SAMPLE_RATIO", "-0.5")
    assert tracing_module._resolve_sample_ratio() == 0.0
    monkeypatch.setenv("LAB_OTEL_SAMPLE_RATIO", "0.25")
    assert tracing_module._resolve_sample_ratio() == 0.25
    monkeypatch.setenv("LAB_OTEL_SAMPLE_RATIO", "bogus")
    assert tracing_module._resolve_sample_ratio() == 1.0


def test_attributes_with_none_value_are_dropped(exporter: object) -> None:
    with tracing_module.span("cell", run_id="r", maybe=None):
        pass
    spans = exporter.get_finished_spans()  # type: ignore[attr-defined]
    attrs = spans[0].attributes
    assert "maybe" not in attrs
    assert attrs["run_id"] == "r"
