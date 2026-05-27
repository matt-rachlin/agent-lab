"""Tests for lab.eval.golden — frozen golden output comparator."""

from __future__ import annotations

from pathlib import Path

from lab.eval.golden import (
    GoldenComparison,
    GoldenOutput,
    compare_to_golden,
    golden_path,
    load_golden,
    save_golden,
)


def _make_golden(tmp_path: Path, **overrides: object) -> Path:
    """Save a default golden under tmp_path and return the root."""
    base: dict[str, object] = {
        "task_slug": "arith-01",
        "model": "qwen3-14b-q4",
        "suite": "pbs-v0.1",
        "config_hash": "abc123",
        "captured_at": "2026-05-27T10:00:00Z",
        "response_text": "148",
        "tool_calls": [],
        "scorer_outcomes": {"exact_match": 1.0},
    }
    base.update(overrides)
    golden = GoldenOutput(**base)  # type: ignore[arg-type]
    save_golden(golden, root=tmp_path)
    return tmp_path


def test_golden_path_layout(tmp_path: Path) -> None:
    p = golden_path("arith-01", "qwen3-14b-q4", suite="pbs-v0.1", root=tmp_path)
    assert p == tmp_path / "pbs-v0.1" / "arith-01" / "qwen3-14b-q4.json"


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    golden = GoldenOutput(
        task_slug="arith-01",
        model="qwen3-14b-q4",
        suite="pbs-v0.1",
        config_hash="abc",
        captured_at="2026-05-27T10:00:00Z",
        response_text="148",
        tool_calls=[{"tool": "calc", "args": {"x": 1}}],
        scorer_outcomes={"exact_match": 1.0, "tool_correctness": 0.5},
    )
    p = save_golden(golden, root=tmp_path)
    assert p.exists()
    loaded = load_golden("arith-01", "qwen3-14b-q4", suite="pbs-v0.1", root=tmp_path)
    assert loaded == golden


def test_load_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_golden("nope", "no-model", suite="pbs-v0.1", root=tmp_path) is None


def test_compare_exact_match(tmp_path: Path) -> None:
    root = _make_golden(tmp_path)
    actual = {
        "response_text": "148",
        "tool_calls": [],
        "scorer_outcomes": {"exact_match": 1.0},
    }
    cmp = compare_to_golden("arith-01", "qwen3-14b-q4", actual, suite="pbs-v0.1", root=root)
    assert cmp.found is True
    assert cmp.same_response is True
    assert cmp.same_tool_calls is True
    assert cmp.scorer_drift == {}
    assert cmp.is_match is True


def test_compare_response_drift_detected(tmp_path: Path) -> None:
    root = _make_golden(tmp_path)
    actual = {
        "response_text": "wrong",
        "tool_calls": [],
        "scorer_outcomes": {"exact_match": 1.0},
    }
    cmp = compare_to_golden("arith-01", "qwen3-14b-q4", actual, suite="pbs-v0.1", root=root)
    assert cmp.found is True
    assert cmp.same_response is False
    assert cmp.is_match is False
    assert cmp.response_diff is not None
    assert "148" in cmp.response_diff


def test_compare_tool_call_drift_detected(tmp_path: Path) -> None:
    root = _make_golden(tmp_path, tool_calls=[{"tool": "calc", "args": {"x": 1}}])
    actual = {
        "response_text": "148",
        "tool_calls": [{"tool": "calc", "args": {"x": 2}}],  # different args
        "scorer_outcomes": {"exact_match": 1.0},
    }
    cmp = compare_to_golden("arith-01", "qwen3-14b-q4", actual, suite="pbs-v0.1", root=root)
    assert cmp.found is True
    assert cmp.same_tool_calls is False
    assert cmp.tool_call_diff is not None


def test_compare_scorer_drift_with_tolerance(tmp_path: Path) -> None:
    root = _make_golden(tmp_path, scorer_outcomes={"score": 0.85})
    # Within tolerance — no drift.
    actual_within = {
        "response_text": "148",
        "tool_calls": [],
        "scorer_outcomes": {"score": 0.86},
    }
    cmp = compare_to_golden(
        "arith-01",
        "qwen3-14b-q4",
        actual_within,
        suite="pbs-v0.1",
        tolerance=0.05,
        root=root,
    )
    assert cmp.scorer_drift == {}

    # Outside tolerance — drift recorded.
    actual_drift = {
        "response_text": "148",
        "tool_calls": [],
        "scorer_outcomes": {"score": 0.50},
    }
    cmp2 = compare_to_golden(
        "arith-01",
        "qwen3-14b-q4",
        actual_drift,
        suite="pbs-v0.1",
        tolerance=0.05,
        root=root,
    )
    assert "score" in cmp2.scorer_drift
    assert cmp2.scorer_drift["score"] == 0.35
    assert cmp2.is_match is False


def test_compare_scorer_missing_on_either_side_records_max_drift(
    tmp_path: Path,
) -> None:
    root = _make_golden(tmp_path, scorer_outcomes={"a": 1.0, "b": 0.5})
    actual = {
        "response_text": "148",
        "tool_calls": [],
        "scorer_outcomes": {"a": 1.0},  # 'b' missing
    }
    cmp = compare_to_golden("arith-01", "qwen3-14b-q4", actual, suite="pbs-v0.1", root=root)
    assert "b" in cmp.scorer_drift
    assert cmp.scorer_drift["b"] == 1.0


def test_compare_missing_golden_returns_not_found(tmp_path: Path) -> None:
    cmp = compare_to_golden(
        "missing",
        "no-model",
        {"response_text": "x", "tool_calls": [], "scorer_outcomes": {}},
        suite="pbs-v0.1",
        root=tmp_path,
    )
    assert cmp.found is False
    assert cmp.is_match is False
    assert isinstance(cmp, GoldenComparison)


def test_normalisation_of_tool_calls_accepts_name_or_arguments_aliases(
    tmp_path: Path,
) -> None:
    # Golden uses canonical {tool, args}; actual uses {name, arguments}.
    root = _make_golden(tmp_path, tool_calls=[{"tool": "calc", "args": {"x": 1}}])
    actual = {
        "response_text": "148",
        "tool_calls": [{"name": "calc", "arguments": {"x": 1}}],
        "scorer_outcomes": {"exact_match": 1.0},
    }
    cmp = compare_to_golden("arith-01", "qwen3-14b-q4", actual, suite="pbs-v0.1", root=root)
    assert cmp.same_tool_calls is True


def test_golden_output_from_dict_handles_missing_optional_fields() -> None:
    raw = {
        "task_slug": "x",
        "model": "m",
        # everything else missing
    }
    g = GoldenOutput.from_dict(raw)
    assert g.task_slug == "x"
    assert g.model == "m"
    assert g.suite == ""
    assert g.tool_calls == []
    assert g.scorer_outcomes == {}
