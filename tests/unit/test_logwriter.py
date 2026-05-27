"""Unit tests for `lab.inspect_bridge.logwriter`.

We don't talk to MinIO or Postgres here — we replace `upload_bytes` and
the inner upsert helpers with capture fixtures and validate that the right
records would be written.
"""

from __future__ import annotations

from typing import Any

import pytest

from lab.inspect_bridge import logwriter as lw_mod
from lab.inspect_bridge.logwriter import (
    SweepContext,
    _aggregate_tokens,
    _compact_turns,
    _trajectory_bytes,
    write_run_from_inspect_log,
)


def _ctx(run_id: str = "run-1") -> SweepContext:
    return SweepContext(
        run_id=run_id,
        experiment_id=42,
        experiment_slug="EXP-X",
        model_id=7,
        model_litellm_id="qwen3-14b-q4",
        task_id=99,
        task_slug="fs-read-001",
        config_hash="abc",
        config={"temperature": 0.0},
        seed=1,
        manifest_sha="deadbeef",
    )


class _FakeSample:
    def __init__(self, lab_agent: dict[str, Any]) -> None:
        self.metadata = {"lab_agent": lab_agent}
        self.scores = {}
        self.messages = []
        self.model_usage = {}
        self.total_time = 1.0
        self.error = None


class _FakeLog:
    def __init__(self, lab_agent: dict[str, Any]) -> None:
        self.samples = [_FakeSample(lab_agent)]
        self.error = None


def _lab_agent() -> dict[str, Any]:
    return {
        "actual_turns": 3,
        "tool_call_count": 2,
        "terminated_reason": "model_finished",
        "total_latency_ms": 1234,
        "error": None,
        "turns": [
            {
                "turn": 0,
                "latency_ms": 100,
                "tokens_in": 10,
                "tokens_out": 5,
                "tool_calls_requested": 1,
                "tool_calls": [
                    {
                        "tool": "fs_read",
                        "args": {"path": "x"},
                        "result": {"content": "y"},
                        "latency_ms": 30,
                        "error": None,
                    }
                ],
            },
            {
                "turn": 1,
                "latency_ms": 80,
                "tokens_in": 12,
                "tokens_out": 6,
                "tool_calls_requested": 0,
                "content_preview": "done",
            },
        ],
    }


def test_compact_turns_keeps_essential_fields() -> None:
    out = _compact_turns({"turns": _lab_agent()["turns"]})
    assert out[0]["turn"] == 0
    assert out[0]["latency_ms"] == 100
    assert out[0]["tools"] == [{"tool": "fs_read", "latency_ms": 30, "error": None}]
    # No args/result/preview in compact form.
    assert "args" not in out[0]
    assert "content_preview" not in out[0]


def test_trajectory_bytes_emits_jsonl_with_header_messages_turns_footer() -> None:
    ctx = _ctx()
    extracted = {
        "lab_agent": _lab_agent(),
        "messages": [{"role": "user", "content": "hi"}],
        "model_usage": {},
        "score": None,
    }
    raw = _trajectory_bytes(ctx=ctx, extracted=extracted)
    lines = raw.decode().rstrip("\n").split("\n")
    import json as _json

    assert _json.loads(lines[0])["type"] == "header"
    assert _json.loads(lines[1])["type"] == "messages"
    turn_types = [_json.loads(line)["type"] for line in lines[2:-1]]
    assert turn_types == ["turn", "turn"]
    footer = _json.loads(lines[-1])
    assert footer["type"] == "footer"
    assert footer["actual_turns"] == 3
    assert footer["tool_call_count"] == 2
    assert footer["terminated_reason"] == "model_finished"


def test_write_run_uploads_and_upserts(monkeypatch: pytest.MonkeyPatch) -> None:
    uploads: list[tuple[str, bytes]] = []
    upserts: list[dict[str, Any]] = []
    agent_logs: list[dict[str, Any]] = []

    def fake_upload(*, key: str, data: bytes, content_type: str) -> str:
        uploads.append((key, data))
        return f"s3://bucket/{key}"

    def fake_run_upsert(*, ctx: SweepContext, extracted: Any, trajectory_key: str) -> None:
        upserts.append(
            {
                "run_id": ctx.run_id,
                "trajectory_key": trajectory_key,
                "lab_agent": extracted["lab_agent"],
            }
        )

    def fake_agent_log_upsert(
        *,
        run_id_: str,
        trajectory_key: str,
        compact_turns: list[dict[str, Any]],
        score_breakdown: dict[str, Any] | None = None,
    ) -> None:
        agent_logs.append(
            {
                "run_id": run_id_,
                "trajectory_key": trajectory_key,
                "turns": compact_turns,
                "score_breakdown": score_breakdown,
            }
        )

    monkeypatch.setattr(lw_mod, "upload_bytes", fake_upload)
    monkeypatch.setattr(lw_mod, "_upsert_experiment_run", fake_run_upsert)
    monkeypatch.setattr(lw_mod, "_upsert_agent_log", fake_agent_log_upsert)

    ctx = _ctx()
    log = _FakeLog(_lab_agent())
    out = write_run_from_inspect_log(log, ctx)
    assert out.startswith("s3://bucket/")
    # Exactly one upload, one runs row, one agent_logs row.
    assert len(uploads) == 1
    assert len(upserts) == 1
    assert len(agent_logs) == 1
    assert agent_logs[0]["run_id"] == ctx.run_id
    # Compact turns are 2 (matches lab_agent above).
    assert len(agent_logs[0]["turns"]) == 2


def test_aggregate_tokens_sums_per_turn_when_model_usage_empty() -> None:
    """Issue #70 regression: bypass-solver runs have `model_usage={}` but
    populated `lab_agent.turns[].tokens_in/out`; aggregation must use the
    turns."""
    lab_agent = _lab_agent()
    ti, to = _aggregate_tokens(lab_agent=lab_agent, model_usage={})
    assert ti == 10 + 12
    assert to == 5 + 6


def test_aggregate_tokens_falls_back_to_model_usage_when_no_turns() -> None:
    """If `lab_agent.turns` lacks token counts (or is empty), the inspect
    native `model_usage` field is used as fallback."""
    ti, to = _aggregate_tokens(
        lab_agent={"turns": []},
        model_usage={
            "qwen3-14b-q4": {"input_tokens": 100, "output_tokens": 25},
            "qwen3-8b": {"input_tokens": 50, "output_tokens": 10},
        },
    )
    assert ti == 150
    assert to == 35


def test_aggregate_tokens_returns_none_when_no_counts_available() -> None:
    ti, to = _aggregate_tokens(lab_agent={}, model_usage=None)
    assert ti is None
    assert to is None


def test_aggregate_tokens_prefers_turns_over_model_usage() -> None:
    """When both are populated, lab_agent.turns wins (it's the bypass
    solver's authoritative record)."""
    ti, to = _aggregate_tokens(
        lab_agent={"turns": [{"tokens_in": 7, "tokens_out": 3}]},
        model_usage={"x": {"input_tokens": 999, "output_tokens": 999}},
    )
    assert ti == 7
    assert to == 3


def test_write_run_handles_empty_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lw_mod, "upload_bytes", lambda **kw: "s3://bucket/x")
    monkeypatch.setattr(lw_mod, "_upsert_experiment_run", lambda **kw: None)
    monkeypatch.setattr(lw_mod, "_upsert_agent_log", lambda **kw: None)

    class _EmptyLog:
        samples = None
        error = "catastrophic"

    out = write_run_from_inspect_log(_EmptyLog(), _ctx())
    assert out == "s3://bucket/x"
