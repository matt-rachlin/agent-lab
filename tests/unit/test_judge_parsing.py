"""Judge response parser tests — no network."""

from __future__ import annotations

from typing import Any

import pytest

from lab.eval import judge as judge_mod
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


def _fake_response(payload: dict[str, Any]) -> Any:
    class _Resp:
        def raise_for_status(self) -> None:  # pragma: no cover - trivial
            return None

        def json(self) -> dict[str, Any]:
            return payload

    return _Resp()


def test_call_litellm_reasoning_content_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When content is empty but reasoning_content has JSON, parse it.

    Regression for Phase 17.6 — gpt-oss-120b-cloud burns max_tokens on
    reasoning_content, leaving message.content="". The judge call site
    now falls back to reasoning_content so the parser can still recover
    the embedded JSON object.
    """

    captured: dict[str, Any] = {}

    def fake_post(url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: int) -> Any:
        captured["max_tokens"] = json["max_tokens"]
        return _fake_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": (
                                "Let me think. The response is faithful. "
                                'Return: {"score": 0.8, "reasoning": "grounded"}.'
                            ),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 80},
            }
        )

    monkeypatch.setattr(judge_mod.httpx, "post", fake_post)
    content, usage = judge_mod._call_litellm(model="gpt-oss-120b-cloud", system="sys", user="usr")
    score, _ = parse_judge_response(content)
    assert score == 0.8
    assert usage["prompt_tokens"] == 100


def test_call_litellm_default_max_tokens_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default max_tokens must be >= 1024 so reasoning judges can answer."""

    captured: dict[str, Any] = {}

    def fake_post(url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: int) -> Any:
        captured["max_tokens"] = json["max_tokens"]
        return _fake_response(
            {
                "choices": [{"message": {"content": '{"score": 0.5}'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

    monkeypatch.setattr(judge_mod.httpx, "post", fake_post)
    judge_mod._call_litellm(model="any", system="sys", user="usr")
    assert captured["max_tokens"] >= 1024
