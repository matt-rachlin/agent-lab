"""Verify Ollama Cloud auth + minimal completion."""

from __future__ import annotations

import os
import sys

import httpx


def main() -> int:
    key = os.environ.get("OLLAMA_API_KEY")
    if not key:
        print("OLLAMA_API_KEY not set", file=sys.stderr)
        return 1

    # Hit Ollama Cloud /api/chat with the cheapest cloud model
    url = "https://ollama.com/api/chat"
    payload = {
        "model": "gpt-oss:20b-cloud",
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {key}"}

    try:
        r = httpx.post(url, json=payload, headers=headers, timeout=60)
    except httpx.HTTPError as exc:
        print(f"http error: {exc}", file=sys.stderr)
        return 2

    if r.status_code != 200:
        print(f"status {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return 3

    data = r.json()
    msg = (data.get("message") or {}).get("content", "")
    eval_count = data.get("eval_count", 0)
    print(f"cloud OK — gpt-oss:20b-cloud responded ({eval_count} tokens):")
    print(f"  {msg.strip()!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
