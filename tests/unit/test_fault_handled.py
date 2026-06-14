"""Lab task #18 — fault recovery evaluator (reliability signal, no veto).

Exercises the pure trace parser + mapping with synthesised trajectory JSONL
rows (no DB, no MinIO, no GPU). A fault that fired + an end_state pass =>
recovered; a fault + task failure => not recovered; no fault => skip (no
recovery signal). Also covers the per-turn `fault_injected` fallback when the
footer summary is empty.
"""

import json

from lab.eval.builtin.fault_handled import (
    parse_trace,
    result_from_trace,
)


def _trace(turns, footer):
    lines = [json.dumps({"type": "header"})]
    for t in turns:
        lines.append(json.dumps({"type": "turn", **t}))
    lines.append(json.dumps({"type": "footer", **footer}))
    return ("\n".join(lines) + "\n").encode("utf-8")


# --- pure mapping ----------------------------------------------------------


def test_recovered_passes():
    r = result_from_trace(2, 1.0)
    assert r.passed is True
    assert r.score == 1.0
    assert r.metadata["faults_fired"] == 2


def test_fault_but_failed_not_recovered():
    r = result_from_trace(1, 0.0)
    assert r.passed is False
    assert r.score == 0.0
    assert r.metadata["faults_fired"] == 1


def test_no_fault_skips():
    r = result_from_trace(0, 1.0)
    assert r.skipped is True


def test_missing_score_unconfirmed_fails_but_noted():
    r = result_from_trace(1, None)
    assert r.passed is False
    assert r.score == 0.0
    assert r.metadata["score_seen"] is None


# --- trace parser ----------------------------------------------------------


def test_parse_footer_faults_and_score():
    blob = _trace(
        turns=[{"turn": 1, "tool_calls": [{"tool": "fs_read", "args": {}}]}],
        footer={
            "faults_fired": [
                {"mode": "error", "tool": "fs_read", "call_index": 1},
                {"mode": "timeout", "tool": "python_eval", "call_index": 1},
            ],
            "score": 1.0,
        },
    )
    n, score = parse_trace(blob)
    assert n == 2
    assert score == 1.0


def test_parse_falls_back_to_turn_markers_when_footer_empty():
    blob = _trace(
        turns=[
            {
                "turn": 1,
                "tool_calls": [
                    {
                        "tool": "fs_read",
                        "args": {},
                        "fault_injected": {"mode": "truncate", "tool": "fs_read"},
                    }
                ],
            }
        ],
        footer={"faults_fired": [], "score": 1.0},
    )
    n, score = parse_trace(blob)
    assert n == 1
    assert score == 1.0


def test_parse_empty_blob():
    assert parse_trace(None) == (0, None)
    assert parse_trace(b"") == (0, None)


def test_end_to_end_recovered_via_parser():
    blob = _trace(
        turns=[{"turn": 1, "tool_calls": [{"tool": "fs_read", "args": {}}]}],
        footer={
            "faults_fired": [{"mode": "error", "tool": "fs_read", "call_index": 1}],
            "score": 1.0,
        },
    )
    n, score = parse_trace(blob)
    r = result_from_trace(n, score)
    assert r.passed is True
