"""HyPE — Hypothetical Prompt Embeddings (Phase 11, 2026-05-26).

At index time, generate a small number of *hypothetical* questions a user
might ask to find each chunk, embed those question strings alongside the
chunk's body, and store both in the LanceDB row. At query time the user's
question matches stored questions (which look like questions) rather than
raw doc text (which doesn't), giving the dense head a much more on-shape
target.

The cost of generating questions is paid once at index time. The query
path remains cheap — one extra vector compare per stored question per
chunk on the candidates the dense head already returned.

Reference: "HyPE: Hypothetical Prompt Embeddings" (the +42pp precision
result reported in the paper applies to a particular benchmark; lab
confidence is medium — single-paper provenance, but the mechanism is
sound and degrades to no-op when the data is absent).

Defaults match the rest of the lab.rag stack: local Ollama, ``qwen3:8b``
for question generation, temperature 0.3 (not zero — a tiny bit of
diversity helps the N questions cover different phrasings).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from ollama import Client

logger = logging.getLogger(__name__)

#: Default question-generator model. Local Ollama, qwen3:8b — same model
#: used by :mod:`lab.rag.eval_retrieval`.
DEFAULT_HYPE_MODEL = "qwen3:8b"

#: Default number of hypothetical questions per chunk. The HyPE paper
#: uses 3 in most experiments; we follow suit.
DEFAULT_N_QUESTIONS = 3

#: Hard cap on individual question length (in chars). Anything over this
#: is dropped as malformed — real user questions don't run this long.
MAX_QUESTION_CHARS = 150

#: Minimum length for a question to be kept. Anything shorter is junk
#: (LLM emitted a bullet, a single word, etc.).
MIN_QUESTION_CHARS = 10

_SYSTEM_PROMPT = (
    "You generate hypothetical user questions that a real user might type into a "
    "search box to find the passage below. Output ONLY the questions, one per "
    "line, no numbering, no commentary, no markdown, no quoting. Each question "
    "should stand alone and be answerable from the passage."
)

_SYSTEM_PROMPT_TIGHT = (
    "Output exactly the requested number of questions. One question per line. "
    "No numbering. No bullets. No commentary. No markdown. No quoting. "
    "Each line must end with a question mark."
)


# Leading-noise pattern: numbered ("1.", "1)", "(1)"), dashed/bulleted ("- ",
# "* ", "• "), or both, possibly with whitespace.
_LEADING_NOISE_RE = re.compile(
    r"^\s*(?:[\(\[]?\s*\d+\s*[\)\].:-]\s*|[-*•]\s+)+",
)

# Strip surrounding quotes if the LLM wrapped a question in them.
_WRAP_QUOTES_RE = re.compile(r'^["\'`](.*)["\'`]$')


def _clean_question(raw: str) -> str | None:
    """Best-effort clean of a single LLM-emitted line.

    Strips numbering, bullets, surrounding quotes; lowercases; drops trailing
    punctuation runs. Returns None when the line is unusable (empty, too short,
    too long, no letters).
    """
    s = raw.strip()
    if not s:
        return None
    # Drop common leading noise (numbering, bullets).
    s = _LEADING_NOISE_RE.sub("", s).strip()
    # Unwrap surrounding quotes.
    m = _WRAP_QUOTES_RE.match(s)
    if m:
        s = m.group(1).strip()
    # Sanity guard: must contain at least one ASCII letter.
    if not re.search(r"[A-Za-z]", s):
        return None
    # Strip trailing punctuation runs of ?!. so we don't end up with "x??".
    s = s.rstrip("?!.")
    if not s:
        return None
    # Reattach a single trailing question mark; we explicitly want each
    # stored string to read like a user question.
    s = s + "?"
    s = s.lower()
    if len(s) < MIN_QUESTION_CHARS:
        return None
    if len(s) > MAX_QUESTION_CHARS:
        return None
    return s


def _parse_questions(raw_text: str, *, n_max: int) -> list[str]:
    """Parse the LLM's response into a deduped, lowercased list of questions.

    Returns ``[]`` when nothing usable came back. Caps the result at
    ``n_max`` even if the LLM emitted more.
    """
    if not raw_text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in raw_text.splitlines():
        cleaned = _clean_question(line)
        if cleaned is None:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
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
    """Single Ollama chat round-trip; returns the assistant message text."""
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


def _build_user_prompt(
    chunk_text: str,
    *,
    n_questions: int,
    section_path: list[str] | None,
) -> str:
    sec = " / ".join(section_path) if section_path else "(none)"
    # Cap the chunk we feed the LLM. Anything over ~1500 chars is wasted
    # context — the question-gen task is dominated by the first paragraph
    # or two.
    body = chunk_text[:1500]
    return (
        f"Section: {sec}\n"
        f"Passage:\n---\n{body}\n---\n"
        f"Generate {n_questions} distinct hypothetical questions, one per line."
    )


def _make_client() -> Client:
    """Construct an Ollama client honouring ``OLLAMA_HOST``."""
    return Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))


def generate_hype_questions(
    chunk_text: str,
    *,
    section_path: list[str] | None = None,
    n_questions: int = DEFAULT_N_QUESTIONS,
    model: str = DEFAULT_HYPE_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 64,
    client: Any | None = None,
) -> list[str]:
    """LLM-generate N hypothetical questions a user might ask to find ``chunk_text``.

    Uses the local Ollama chat endpoint. Parsing is tolerant of common
    formatting quirks (numbering, bullets, surrounding quotes). Returns a
    list of *up to* ``n_questions`` lowercased question strings; if the
    LLM yields fewer usable questions than asked, we return what we have
    (callers must accept partial output).

    On parser failure (zero usable questions) we retry once with a
    tightened system prompt — the second call is short and we only do it
    if the first produced nothing.

    Returns ``[]`` on irrecoverable failure (Ollama unreachable, model
    refused twice, etc). The index path treats this as "leave hype
    columns null for this chunk" — never a hard error.
    """
    if not chunk_text or not chunk_text.strip():
        return []
    if n_questions <= 0:
        return []

    cli = client if client is not None else _make_client()
    user_prompt = _build_user_prompt(
        chunk_text,
        n_questions=n_questions,
        section_path=section_path,
    )

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
        logger.warning("hype: first chat round failed: %s", exc)
        raw = ""

    questions = _parse_questions(raw, n_max=n_questions)
    if questions:
        return questions

    # Retry once with a tightened prompt. We bump the model toward the
    # exact format we want.
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
        logger.warning("hype: retry chat round failed: %s", exc)
        return []
    return _parse_questions(raw2, n_max=n_questions)


__all__ = [
    "DEFAULT_HYPE_MODEL",
    "DEFAULT_N_QUESTIONS",
    "MAX_QUESTION_CHARS",
    "MIN_QUESTION_CHARS",
    "generate_hype_questions",
]
