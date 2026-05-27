"""Multi-query expansion (Phase 12, 2026-05-26).

At query time, ask a small local LLM to produce N alternate phrasings of
the user's question. Run the hybrid retriever once per phrasing
(including the original), RRF-fuse the result lists. Helps the long-tail
of ambiguous queries where the user's wording doesn't match the
documentation's wording.

Cost: one LLM call + N additional retrievals per top-level question. The retrievals
hit the tier-1 embedding cache for repeats, so successive queries that
expand to the same phrasings replay cheaply.

Module mirrors :mod:`lab.rag.hype` in shape — same parser tolerances,
same fallback strategy (return ``[question]`` on irrecoverable failure
so the caller degrades to single-query behaviour silently).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from ollama import Client

logger = logging.getLogger(__name__)

#: Default phrasing-generator model. Local Ollama, qwen3:8b.
DEFAULT_EXPAND_MODEL = "qwen3:8b"

#: Default number of *alternate* phrasings (caller receives ``n+1`` strings,
#: including the original).
DEFAULT_N_ALTERNATIVES = 3

#: Hard cap on phrasing length (chars). Anything over this is dropped.
MAX_PHRASING_CHARS = 200

#: Minimum length for a phrasing to be kept.
MIN_PHRASING_CHARS = 4

_SYSTEM_PROMPT = (
    "You generate alternate phrasings of a user's search query that mean "
    "the same thing but use different vocabulary. Output ONLY the phrasings, "
    "one per line. No numbering. No commentary. No markdown. No quoting. "
    "Each phrasing must be answerable from the same underlying answer as the "
    "original."
)

_SYSTEM_PROMPT_TIGHT = (
    "Output exactly the requested number of phrasings. One per line. "
    "No numbering. No bullets. No commentary. No markdown. No quoting."
)


_LEADING_NOISE_RE = re.compile(
    r"^\s*(?:[\(\[]?\s*\d+\s*[\)\].:-]\s*|[-*•]\s+)+",
)
_WRAP_QUOTES_RE = re.compile(r'^["\'`](.*)["\'`]$')


def _clean_phrasing(raw: str) -> str | None:
    """Best-effort clean of a single LLM-emitted line.

    Mirrors :func:`lab.rag.hype._clean_question` but does NOT force a
    trailing ``?`` — many user queries are statements ("find me X").
    """
    s = raw.strip()
    if not s:
        return None
    s = _LEADING_NOISE_RE.sub("", s).strip()
    m = _WRAP_QUOTES_RE.match(s)
    if m:
        s = m.group(1).strip()
    if not re.search(r"[A-Za-z]", s):
        return None
    # Squash internal whitespace runs.
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    if len(s) < MIN_PHRASING_CHARS:
        return None
    if len(s) > MAX_PHRASING_CHARS:
        return None
    return s


def _parse_phrasings(raw_text: str, *, n_max: int) -> list[str]:
    """Parse LLM output into a deduped (case-insensitive) list of phrasings.

    Returns ``[]`` when nothing usable came back. Caps at ``n_max``.
    """
    if not raw_text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in raw_text.splitlines():
        cleaned = _clean_phrasing(line)
        if cleaned is None:
            continue
        norm = cleaned.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(cleaned)
        if len(out) >= n_max:
            break
    return out


def _chat_once(
    *,
    client: Client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "num_ctx": 4096,
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    )
    msg = resp.get("message") if isinstance(resp, dict) else None
    if not isinstance(msg, dict):
        return ""
    text = msg.get("content")
    return text if isinstance(text, str) else ""


def _build_user_prompt(question: str, *, n: int) -> str:
    return (
        f"Original query:\n{question}\n\n"
        f"Generate {n} alternate phrasings of the query, one per line."
    )


def _make_client() -> Client:
    return Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))


def _dedupe_keep_original(original: str, alternates: list[str]) -> list[str]:
    """Return ``[original] + dedup(alternates)`` with case-insensitive
    deduplication that drops alternates that match the original.
    """
    seen: set[str] = {original.strip().lower()}
    out: list[str] = [original]
    for a in alternates:
        norm = a.strip().lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(a)
    return out


def multi_query(
    question: str,
    *,
    n: int = DEFAULT_N_ALTERNATIVES,
    model: str = DEFAULT_EXPAND_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 96,
    client: Any | None = None,
) -> list[str]:
    """Generate N alternate phrasings of ``question`` via local Ollama.

    Returns a list where element 0 is the original question and elements
    1..N are LLM-generated phrasings (deduped, case-insensitive). On
    irrecoverable LLM failure we return ``[question]`` so callers can
    degrade gracefully to single-query behaviour without exception
    handling at the call site.

    The returned list may contain fewer than N+1 entries when the LLM
    produced fewer usable phrasings than asked; the only guarantee is
    that ``out[0] == question``.
    """
    if not question or not question.strip():
        return []
    if n <= 0:
        return [question]

    cli = client if client is not None else _make_client()
    user_prompt = _build_user_prompt(question, n=n)

    try:
        raw = _chat_once(
            client=cli,
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.warning("multi_query: first chat round failed: %s", exc)
        raw = ""

    alternates = _parse_phrasings(raw, n_max=n)
    if not alternates:
        try:
            raw2 = _chat_once(
                client=cli,
                model=model,
                system_prompt=_SYSTEM_PROMPT_TIGHT,
                user_prompt=user_prompt,
                temperature=max(temperature - 0.1, 0.0),
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.warning("multi_query: retry chat round failed: %s", exc)
            return [question]
        alternates = _parse_phrasings(raw2, n_max=n)
    if not alternates:
        # Degrade gracefully: caller still gets the original.
        return [question]
    return _dedupe_keep_original(question, alternates)


__all__ = [
    "DEFAULT_EXPAND_MODEL",
    "DEFAULT_N_ALTERNATIVES",
    "MAX_PHRASING_CHARS",
    "MIN_PHRASING_CHARS",
    "multi_query",
]
