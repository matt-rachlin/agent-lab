"""Judge response parser tests — no network."""

from __future__ import annotations

from lab.eval.judge import parse_judge_response


def test_clean_json() -> None:
    score, reason = parse_judge_response('{"score": 0.8, "reasoning": "addresses task"}')
    assert score == 0.8
    assert reason == "addresses task"


def test_fenced_json() -> None:
    score, _ = parse_judge_response('```json\n{"score": 0.5}\n```')
    assert score == 0.5


def test_score_colon_pattern() -> None:
    score, _ = parse_judge_response("Score: 0.73 — looks ok")
    assert score == 0.73


def test_leading_number() -> None:
    score, _ = parse_judge_response("0.42 because reasons")
    assert score == 0.42


def test_clamping() -> None:
    score, _ = parse_judge_response('{"score": 1.5}')
    assert score == 1.0
    score, _ = parse_judge_response('{"score": -0.3}')
    assert score == 0.0


def test_unparseable() -> None:
    score, reason = parse_judge_response("haha this is just text")
    assert score == 0.0
    assert reason is not None
    assert "unparseable" in reason


def test_empty() -> None:
    score, reason = parse_judge_response("")
    assert score == 0.0
    assert reason is not None
