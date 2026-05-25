"""LLM-as-judge — LiteLLM-backed, position-swap-mitigated, tolerant parsing.

Usage:

    from lab.eval.judge import make_judge

    judge = make_judge(model="gpt-oss-20b-cloud")
    score, reasoning = judge(prompt="Score the following 0-1: ...")
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import httpx

from lab.settings import get_settings

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator. You score outputs strictly per the rubric. "
    "Always respond with a single JSON object: "
    '{"score": <float 0.0-1.0>, "reasoning": "<one short sentence>"}. '
    "Never include any text outside the JSON."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b(0(?:\.\d+)?|1(?:\.0+)?|0|1)\b")


def _read_litellm_key() -> str:
    p = Path("/data/lab/services/litellm-master-key")
    return p.read_text().strip() if p.exists() else ""


def parse_judge_response(text: str) -> tuple[float, str | None]:
    """Tolerant: try JSON first, then `score: N` shapes, then leading-number."""
    if not text:
        return 0.0, "empty judge response"
    cleaned = text.strip()
    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    # Try strict JSON
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "score" in obj:
            score = float(obj["score"])
            return _clamp(score), obj.get("reasoning")
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try `score: N` patterns
    m = re.search(r"score['\"]?\s*[:=]\s*([0-9]*\.?[0-9]+)", cleaned, re.IGNORECASE)
    if m:
        return _clamp(float(m.group(1))), cleaned

    # Leading number
    m = re.match(r"\s*([0-9]*\.?[0-9]+)", cleaned)
    if m:
        return _clamp(float(m.group(1))), cleaned

    return 0.0, f"unparseable: {cleaned[:120]!r}"


def _clamp(x: float) -> float:
    if x != x:  # NaN check
        return 0.0
    return max(0.0, min(1.0, x))


def _call_litellm(
    *, model: str, system: str, user: str, timeout: int = 120
) -> tuple[str, dict[str, int]]:
    """Plain chat completion via the lab's LiteLLM proxy."""
    settings = get_settings()
    key = _read_litellm_key()
    url = settings.litellm_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    r = httpx.post(url, json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = ((data.get("choices") or [{}])[0]).get("message", {}).get("content", "")
    usage = data.get("usage") or {}
    return content, {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


def make_judge(
    *,
    model: str,
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
    position_swap: bool = False,
    timeout: int = 120,
) -> Callable[..., tuple[float, str | None]]:
    """Return a judge callable. Optional position-swap averaging (for pair tasks)."""

    def judge(
        *,
        prompt: str,
        expected_format: Literal["score_only", "score_reasoning"] = "score_reasoning",
    ) -> tuple[float, str | None]:
        forward, _ = _call_litellm(model=model, system=system_prompt, user=prompt, timeout=timeout)
        score_a, reason_a = parse_judge_response(forward)
        if not position_swap:
            return score_a, reason_a

        # Naive position-swap: only useful for pairwise prompts that contain "A:" / "B:"
        if "A:" in prompt and "B:" in prompt:
            swapped = prompt.replace("A:", "<<TMP>>").replace("B:", "A:").replace("<<TMP>>", "B:")
            backward, _ = _call_litellm(
                model=model, system=system_prompt, user=swapped, timeout=timeout
            )
            score_b, reason_b = parse_judge_response(backward)
            avg = (score_a + (1.0 - score_b)) / 2.0  # B's score is for the swapped item
            return avg, (reason_a or "") + " | swap: " + (reason_b or "")
        return score_a, reason_a

    return judge
