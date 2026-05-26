"""Shared LiteLLM proxy client.

Single source of truth for the request shape we send to the LiteLLM proxy
(OpenAI-compatible `/v1/chat/completions`). The single-turn sweep runner and
the multi-turn agent solver both go through this so we keep their request
shapes from drifting apart.

The function returns the raw response JSON plus the wall-clock latency in
milliseconds. Callers parse the response themselves — agent loops need the
`tool_calls` field that single-turn callers ignore.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import httpx


class _SettingsLike(Protocol):
    litellm_url: str


def call_litellm_chat(
    *,
    settings: _SettingsLike,
    litellm_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    timeout: int = 600,
) -> tuple[dict[str, Any], int]:
    """POST to `{litellm_url}/v1/chat/completions`.

    Returns `(response_json, latency_ms)`. Raises `httpx.HTTPStatusError` on
    non-2xx. `extra` is forwarded into the body verbatim — useful for backend
    knobs like Ollama's `think: false` — but `system_prompt` is dropped
    because it is consumed locally by the message-building precedence rules.
    """

    url = settings.litellm_url.rstrip("/") + "/v1/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if tools:
        body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    if extra:
        for k, v in extra.items():
            if k == "system_prompt":
                continue
            body[k] = v
    headers = {"Authorization": f"Bearer {litellm_key}", "Content-Type": "application/json"}
    t0 = time.monotonic()
    resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
    latency_ms = int((time.monotonic() - t0) * 1000)
    resp.raise_for_status()
    return resp.json(), latency_ms


__all__ = ["call_litellm_chat"]
