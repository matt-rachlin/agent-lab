"""Tests for tools/sync_golden_outputs.py — real-capture path.

These tests exercise the capture path with the model client mocked. The
matrix-expansion / dry-run behaviour is also covered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from tools import sync_golden_outputs as sgo

# ---------------------------------------------------------------------------
# Rubric scoring (single-turn fast path)
# ---------------------------------------------------------------------------


def test_score_single_turn_exact_match_passes() -> None:
    payload = {
        "gold_answer": "148",
        "rubric": {"type": "exact_match"},
    }
    out = sgo._score_single_turn(payload, "The answer is 148.")
    assert out == {"exact_match": 1.0}


def test_score_single_turn_exact_match_fails() -> None:
    payload = {
        "gold_answer": "148",
        "rubric": {"type": "exact_match"},
    }
    out = sgo._score_single_turn(payload, "I think it is 149.")
    assert out == {"exact_match": 0.0}


def test_score_single_turn_regex_match() -> None:
    payload = {
        "rubric": {"type": "regex", "pattern": r"5\s*/\s*12"},
    }
    out = sgo._score_single_turn(payload, "Answer: 5 / 12.")
    assert out == {"regex_match": 1.0}


def test_score_single_turn_unknown_rubric_returns_empty() -> None:
    payload = {"rubric": {"type": "json_schema"}}
    out = sgo._score_single_turn(payload, "anything")
    assert out == {}


# ---------------------------------------------------------------------------
# Matrix expansion
# ---------------------------------------------------------------------------


def test_build_matrix_filters_by_suite_and_model(tmp_path: Path) -> None:
    """`--suite X --model Y` only emits triples for that combo."""
    args = mock.Mock()
    args.suite = "pbs-v0.1"
    args.model = "qwen3-14b-q4"
    args.task = None

    triples = sgo._build_matrix(args)
    # All triples should be in pbs-v0.1 with the picked model.
    assert triples, "matrix is empty — repo layout shifted?"
    suites = {s for s, _, _ in triples}
    models = {m for _, _, m in triples}
    assert suites == {"pbs-v0.1"}
    assert models == {"qwen3-14b-q4"}


# ---------------------------------------------------------------------------
# Capture single-turn (model client mocked)
# ---------------------------------------------------------------------------


def _fake_call_litellm_chat(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], int]:
    return (
        {
            "choices": [{"message": {"role": "assistant", "content": "148"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        37,
    )


def test_capture_single_turn_writes_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_capture` (single-turn) returns a payload with the schema fields."""

    fake_task_rows = [
        {
            "task_id": 1,
            "suite": "PBS-v0.1",
            "slug": "math-001",
            "category": "math",
            "difficulty": "easy",
            "payload": {
                "input": "Compute (47 * 8) - (12 * 19).",
                "gold_answer": "148",
                "rubric": {"type": "exact_match"},
            },
        }
    ]

    monkeypatch.setattr("lab.tasks.registry.get_tasks", lambda *a, **kw: fake_task_rows)
    monkeypatch.setattr("lab.core.llm.call_litellm_chat", _fake_call_litellm_chat)
    monkeypatch.setattr(sgo, "_read_litellm_key", lambda: "test-key")

    payload = sgo._capture(
        suite_db="PBS-v0.1",
        suite_label="pbs-v0.1",
        task_slug="math-001",
        model="gpt-oss-120b-cloud",  # cloud → no GPU lease
        backend="ollama-cloud",
        timeout=30,
        force_lease=True,
    )

    assert payload["task_slug"] == "math-001"
    assert payload["model"] == "gpt-oss-120b-cloud"
    assert payload["suite"] == "pbs-v0.1"
    assert payload["response_text"] == "148"
    assert payload["tool_calls"] == []
    assert payload["scorer_outcomes"] == {"exact_match": 1.0}
    assert payload["trajectory_summary"]["actual_turns"] == 1
    assert payload["trajectory_summary"]["tool_call_count"] == 0
    assert payload["config_hash"] == sgo._config_hash("pbs-v0.1", "gpt-oss-120b-cloud")
    assert payload["captured_at"].endswith("Z")


def test_capture_single_turn_acquires_gpu_lease_for_local_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local backend → must enter gpu_lease context manager."""

    fake_task_rows = [
        {
            "task_id": 1,
            "suite": "PBS-v0.1",
            "slug": "math-001",
            "category": "math",
            "difficulty": "easy",
            "payload": {
                "input": "x",
                "gold_answer": "148",
                "rubric": {"type": "exact_match"},
            },
        }
    ]
    monkeypatch.setattr("lab.tasks.registry.get_tasks", lambda *a, **kw: fake_task_rows)
    monkeypatch.setattr("lab.core.llm.call_litellm_chat", _fake_call_litellm_chat)
    monkeypatch.setattr(sgo, "_read_litellm_key", lambda: "test-key")

    entered = {"count": 0}

    from contextlib import contextmanager

    @contextmanager
    def _fake_lease(*args: Any, **kwargs: Any) -> Any:
        entered["count"] += 1
        yield "tag"

    monkeypatch.setattr("lab.core.gpu_lease.gpu_lease", _fake_lease)

    sgo._capture(
        suite_db="PBS-v0.1",
        suite_label="pbs-v0.1",
        task_slug="math-001",
        model="qwen3-14b-q4",
        backend="ollama-local",
        timeout=30,
        force_lease=False,
    )
    assert entered["count"] == 1


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_existing_golden_matches_when_hash_present(tmp_path: Path) -> None:
    target = tmp_path / "x.json"
    target.write_text(json.dumps({"config_hash": "deadbeef00000000", "task_slug": "t"}))
    assert sgo._existing_golden_matches(target, "deadbeef00000000")
    assert not sgo._existing_golden_matches(target, "different00000000")
    assert not sgo._existing_golden_matches(tmp_path / "missing.json", "x")


def test_existing_golden_matches_handles_corrupt_file(tmp_path: Path) -> None:
    target = tmp_path / "bad.json"
    target.write_text("not valid json{")
    assert not sgo._existing_golden_matches(target, "anything")


# ---------------------------------------------------------------------------
# Main entry: capture written file is readable by compare_to_golden
# ---------------------------------------------------------------------------


def test_main_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rc = sgo.main(
        [
            "--suite",
            "pbs-v0.1",
            "--model",
            "qwen3-14b-q4",
            "--root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "would-write" in out
    # No file should have been touched in tmp_path.
    assert list(tmp_path.rglob("*.json")) == []


def test_captured_payload_loadable_via_compare_to_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: write a payload, then read it back via the comparator."""

    fake_task_rows = [
        {
            "task_id": 1,
            "suite": "PBS-v0.1",
            "slug": "math-001",
            "category": "math",
            "difficulty": "easy",
            "payload": {
                "input": "Compute.",
                "gold_answer": "148",
                "rubric": {"type": "exact_match"},
            },
        }
    ]
    monkeypatch.setattr("lab.tasks.registry.get_tasks", lambda *a, **kw: fake_task_rows)
    monkeypatch.setattr("lab.core.llm.call_litellm_chat", _fake_call_litellm_chat)
    monkeypatch.setattr(sgo, "_read_litellm_key", lambda: "test-key")

    payload = sgo._capture(
        suite_db="PBS-v0.1",
        suite_label="pbs-v0.1",
        task_slug="math-001",
        model="gpt-oss-120b-cloud",
        backend="ollama-cloud",
        timeout=30,
        force_lease=True,
    )

    target = tmp_path / "pbs-v0.1" / "math-001" / "gpt-oss-120b-cloud.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    from lab.eval.golden import compare_to_golden

    cmp = compare_to_golden(
        "math-001",
        "gpt-oss-120b-cloud",
        {
            "response_text": "148",
            "tool_calls": [],
            "scorer_outcomes": {"exact_match": 1.0},
        },
        suite="pbs-v0.1",
        root=tmp_path,
    )
    assert cmp.found is True
    assert cmp.same_response is True
    assert cmp.is_match is True
