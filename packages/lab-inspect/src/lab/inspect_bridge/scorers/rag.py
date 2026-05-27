"""Inspect Scorers for RAG agent runs (Phase 6h-c).

Five scorer factories that grade an agent's behaviour on retrieval-augmented
tasks. Each is decorated with `@scorer` and follows the same shape as the
6e set in :mod:`lab.inspect_bridge.scorer`:

  * `recall_at_k(k)` — fraction of expected chunks the agent retrieved
    across all `kb_query` calls, taking the first-`k` hits per call.
  * `mrr()` — Mean Reciprocal Rank over expected chunks; for each
    expected chunk we take the BEST rank seen across any `kb_query` call.
  * `ndcg(k)` — nDCG@k against the first `kb_query` call's hit list,
    using ``relevance_grades`` from the predicate (default 1.0 per
    expected chunk).
  * `faithfulness(judge_model)` — LLM-as-judge over the final assistant
    message vs the union of retrieved chunk text. Opt-in via
    ``success_predicate.include_faithfulness`` so we don't pay the judge
    cost on every retrieval cell.
  * `attribution()` — cheap heuristic: does the final assistant message
    cite any of the retrieved `source_url`s? 1.0 if yes, 0.5 if it
    references a chunk_id / section, 0.0 otherwise.

All scorers read the trajectory from
``state.metadata["lab_agent"]`` and the task spec from
``state.metadata["lab_task"]``. They return ``Score(value=NOANSWER, ...)``
when their preconditions aren't met (e.g. a non-retrieval task, or no
`kb_query` calls in the trajectory), so they can be added to a default
scorer list without polluting unrelated runs.
"""

from __future__ import annotations

import math
import re
from typing import Any

from inspect_ai.scorer import NOANSWER, Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState
from lab.eval.judge import make_judge

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

#: Tool name the harness exposes for KB retrieval. Hard-coded here because
#: the scorers don't import the tool module (which would drag mcp / lancedb
#: into the scorer's import path).
_KB_TOOL = "kb_query"


def _get_lab_task(state: TaskState) -> Any:
    """Pull the lab Task off ``state.metadata``. Returns None if missing."""

    md = state.metadata or {}
    return md.get("lab_task")


def _get_lab_agent(state: TaskState) -> dict[str, Any]:
    """Pull the trajectory dict off ``state.metadata``."""

    md = state.metadata or {}
    return md.get("lab_agent") or {}


def _retrieval_predicate(state: TaskState) -> dict[str, Any] | None:
    """Return the ``retrieval_recall`` predicate dict, or None."""

    task = _get_lab_task(state)
    if task is None:
        return None
    pred = getattr(task, "success_predicate", None)
    if not isinstance(pred, dict):
        return None
    if pred.get("type") != "retrieval_recall":
        return None
    return pred


def _kb_query_calls(lab_agent: dict[str, Any]) -> list[dict[str, Any]]:
    """All `kb_query` tool calls flattened across turns, in order.

    Each returned dict carries the call's full record (``tool``, ``args``,
    ``result`` …). We don't filter on whether the call succeeded — a call
    that returned ``{"hits": []}`` is still a call, and the scorers want
    to count it.
    """

    out: list[dict[str, Any]] = []
    for turn in lab_agent.get("turns") or []:
        for tc in turn.get("tool_calls") or []:
            if tc.get("tool") == _KB_TOOL:
                out.append(tc)
    return out


def _hits_from_call(call: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``result["hits"]`` list for a `kb_query` call, or [].

    Defensive against missing / malformed payloads — a tool call that
    blew up on the sandbox side still leaves a ``result`` slot, possibly
    a string or None.
    """

    result = call.get("result")
    if isinstance(result, dict):
        hits = result.get("hits")
        if isinstance(hits, list):
            return [h for h in hits if isinstance(h, dict)]
    return []


def _final_assistant_message(state: TaskState) -> str:
    """Best-effort: pull the agent's final assistant message text."""

    for msg in reversed(state.messages or []):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role != "assistant":
            continue
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        # Handle list-of-parts content shape (Inspect sometimes ships it as
        # a list of ContentText objects).
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                else:
                    txt = getattr(part, "text", None)
                    if isinstance(txt, str):
                        parts.append(txt)
            joined = "\n".join(p for p in parts if p)
            if joined.strip():
                return joined
    return ""


# ---------------------------------------------------------------------------
# 1. recall@k
# ---------------------------------------------------------------------------


@scorer(metrics=[mean()], name="recall_at_k")
def recall_at_k(k: int = 5) -> Scorer:
    """Fraction of expected chunks the agent retrieved (top-k per call).

    Walks every `kb_query` call in the trajectory, takes the first ``k``
    hits per call, and unions their ``chunk_id`` values. Recall is
    ``|expected ∩ retrieved| / |expected|``.

    Returns NOANSWER for tasks whose ``success_predicate.type`` is not
    ``retrieval_recall``.
    """

    async def score(state: TaskState, target: Target) -> Score:
        pred = _retrieval_predicate(state)
        if pred is None:
            return Score(
                value=NOANSWER,
                explanation="no retrieval_recall predicate on this task",
            )
        expected_raw = pred.get("expected_chunks") or []
        expected = {str(c) for c in expected_raw if c is not None}
        if not expected:
            return Score(
                value=NOANSWER,
                explanation="retrieval_recall predicate has no expected_chunks",
            )
        # Predicate can override the default k; explicit arg wins for tests.
        effective_k = int(pred.get("k", k))
        if effective_k <= 0:
            effective_k = k

        lab_agent = _get_lab_agent(state)
        retrieved: set[str] = set()
        for call in _kb_query_calls(lab_agent):
            for hit in _hits_from_call(call)[:effective_k]:
                cid = hit.get("chunk_id")
                if cid is not None:
                    retrieved.add(str(cid))

        hit_set = expected & retrieved
        recall = len(hit_set) / len(expected)
        return Score(
            value=recall,
            explanation=(
                f"retrieved {len(hit_set)}/{len(expected)} expected chunks "
                f"(k={effective_k}, |retrieved|={len(retrieved)})"
            ),
            metadata={
                "expected_count": len(expected),
                "retrieved_count": len(retrieved),
                "matched_count": len(hit_set),
                "k": effective_k,
            },
        )

    return score


# ---------------------------------------------------------------------------
# 2. MRR
# ---------------------------------------------------------------------------


@scorer(metrics=[mean()], name="mrr")
def mrr() -> Scorer:
    """Mean Reciprocal Rank averaged over expected chunks.

    For each expected chunk, we find its BEST rank across all `kb_query`
    calls (rank = 1-indexed position in that call's hit list). RR is
    ``1/rank`` for found chunks, ``0`` for misses. The score is the mean
    over expected chunks.
    """

    async def score(state: TaskState, target: Target) -> Score:
        pred = _retrieval_predicate(state)
        if pred is None:
            return Score(
                value=NOANSWER,
                explanation="no retrieval_recall predicate on this task",
            )
        expected_raw = pred.get("expected_chunks") or []
        expected = [str(c) for c in expected_raw if c is not None]
        if not expected:
            return Score(
                value=NOANSWER,
                explanation="retrieval_recall predicate has no expected_chunks",
            )

        lab_agent = _get_lab_agent(state)
        calls = _kb_query_calls(lab_agent)
        if not calls:
            # Every expected chunk gets rank "infinity" → RR=0 → mean=0.
            return Score(
                value=0.0,
                explanation=f"no {_KB_TOOL} calls; MRR=0",
            )

        rrs: list[float] = []
        for cid in expected:
            best_rr = 0.0
            for call in calls:
                for idx, hit in enumerate(_hits_from_call(call)):
                    if str(hit.get("chunk_id")) == cid:
                        rr = 1.0 / (idx + 1)
                        if rr > best_rr:
                            best_rr = rr
                        break  # rank within this call is fixed
            rrs.append(best_rr)
        value = sum(rrs) / len(rrs)
        found = sum(1 for r in rrs if r > 0)
        return Score(
            value=value,
            explanation=(
                f"MRR={value:.3f} over {len(expected)} expected chunks "
                f"({found} found across {len(calls)} {_KB_TOOL} call(s))"
            ),
            metadata={"per_chunk_rr": dict(zip(expected, rrs, strict=True))},
        )

    return score


# ---------------------------------------------------------------------------
# 3. nDCG
# ---------------------------------------------------------------------------


def _dcg(relevances: list[float]) -> float:
    """Standard DCG: sum_i rel_i / log2(i+2) with i 0-indexed."""

    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


@scorer(metrics=[mean()], name="ndcg")
def ndcg(k: int = 10) -> Scorer:
    """nDCG@k against the agent's first `kb_query` call.

    The predicate may carry ``relevance_grades: {chunk_id: float}``;
    chunks not listed are assumed unsupervised (relevance 0). When
    ``relevance_grades`` is absent we default every ``expected_chunks``
    entry to 1.0 — graded relevance is opt-in.

    nDCG = DCG(agent top-k) / IDCG (perfectly sorted relevances).
    """

    async def score(state: TaskState, target: Target) -> Score:
        pred = _retrieval_predicate(state)
        if pred is None:
            return Score(
                value=NOANSWER,
                explanation="no retrieval_recall predicate on this task",
            )
        expected_raw = pred.get("expected_chunks") or []
        expected = [str(c) for c in expected_raw if c is not None]
        if not expected:
            return Score(
                value=NOANSWER,
                explanation="retrieval_recall predicate has no expected_chunks",
            )
        # Build the relevance map.
        grades_raw = pred.get("relevance_grades") or {}
        if not isinstance(grades_raw, dict):
            grades_raw = {}
        # Default: every expected chunk has relevance 1.0.
        grades: dict[str, float] = dict.fromkeys(expected, 1.0)
        for cid, val in grades_raw.items():
            try:
                grades[str(cid)] = float(val)
            except (TypeError, ValueError):
                continue

        effective_k = int(pred.get("k", k))
        if effective_k <= 0:
            effective_k = k

        lab_agent = _get_lab_agent(state)
        calls = _kb_query_calls(lab_agent)
        if not calls:
            return Score(value=0.0, explanation=f"no {_KB_TOOL} calls; nDCG=0")

        first_hits = _hits_from_call(calls[0])[:effective_k]
        agent_rels = [grades.get(str(h.get("chunk_id")), 0.0) for h in first_hits]
        dcg = _dcg(agent_rels)

        ideal_rels_full = sorted(grades.values(), reverse=True)
        ideal_rels = ideal_rels_full[:effective_k]
        idcg = _dcg(ideal_rels)
        if idcg <= 0:
            return Score(
                value=0.0,
                explanation="ideal DCG is zero (all relevance grades 0)",
            )
        value = dcg / idcg
        return Score(
            value=value,
            explanation=(
                f"nDCG@{effective_k}={value:.3f} "
                f"(DCG={dcg:.3f}, IDCG={idcg:.3f}, agent top-{effective_k} returned "
                f"{len(first_hits)} hits)"
            ),
            metadata={"dcg": dcg, "idcg": idcg, "k": effective_k},
        )

    return score


# ---------------------------------------------------------------------------
# 4. faithfulness
# ---------------------------------------------------------------------------


_FAITHFULNESS_RUBRIC = (
    "You are evaluating whether an AI assistant's response is FAITHFUL to a "
    "set of retrieved knowledge-base passages.\n"
    "A response is faithful when EVERY factual claim is either:\n"
    "  * directly supported by one of the retrieved passages, OR\n"
    "  * a clearly-marked admission that the information is not in the passages.\n"
    "Score the response on a 1-5 integer scale:\n"
    "  5 — every claim is supported; no hallucinations\n"
    "  4 — almost all claims supported; one minor unsupported aside\n"
    "  3 — mostly supported, some unsupported claims\n"
    "  2 — significant unsupported claims, but partially grounded\n"
    "  1 — response is largely hallucinated or contradicts the passages\n"
    'Reply with a JSON object: {"score": <int 1-5>, "reasoning": "<one short sentence>"}.\n'
    "Output JSON only — no preamble or commentary outside the JSON."
)


def _normalise_1_to_5(raw: float, reasoning: str | None) -> float:
    """Same trick as `lab.inspect_bridge.scorer._normalise_1_to_5`.

    The judge parser clamps anything > 1 to 1.0, so a 1-5 reply loses
    granularity. Sniff the reasoning for an explicit ``"score": N`` and
    rescale. This avoids importing the private helper from sibling
    module (keeps the import graph clean) at the cost of a tiny dup.
    """

    if reasoning:
        m = re.search(r'"score"\s*:\s*([1-5])\b', reasoning)
        if m:
            return int(m.group(1)) / 5.0
        m = re.search(r"\bscore\s*[:=]\s*([1-5])\b", reasoning, re.IGNORECASE)
        if m:
            return int(m.group(1)) / 5.0
    return max(0.0, min(1.0, float(raw)))


@scorer(metrics=[mean()], name="faithfulness")
def faithfulness(judge_model: str = "gpt-oss-120b-cloud") -> Scorer:
    """LLM-as-judge: is the agent's final response grounded in retrieved chunks?

    Opt-in. Even with the scorer wired up, the adapter only adds it when
    ``success_predicate.include_faithfulness`` is true AND the task has
    `kb_query` in its tool list.

    Edge cases:
      * No `kb_query` calls happened — NOANSWER, "no retrieval performed".
      * Judge unreachable — NOANSWER, "judge unavailable: …".
    """

    async def score(state: TaskState, target: Target) -> Score:
        lab_agent = _get_lab_agent(state)
        calls = _kb_query_calls(lab_agent)
        if not calls:
            return Score(value=NOANSWER, explanation="no retrieval performed")

        # Collect all retrieved chunk texts (deduped by chunk_id when present).
        seen_ids: set[str] = set()
        chunks: list[str] = []
        for call in calls:
            for hit in _hits_from_call(call):
                cid = hit.get("chunk_id")
                key = str(cid) if cid is not None else f"_anon_{len(chunks)}"
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                text = hit.get("text")
                if isinstance(text, str) and text.strip():
                    src = hit.get("source_url") or ""
                    header = f"[chunk {cid}{f' from {src}' if src else ''}]"
                    chunks.append(f"{header}\n{text}")

        final = _final_assistant_message(state)
        if not final.strip():
            return Score(value=NOANSWER, explanation="agent produced no final response")

        if not chunks:
            # Calls happened but every hit had empty text — judge has
            # nothing to ground against. Treat as NOANSWER rather than
            # silently scoring 0.
            return Score(
                value=NOANSWER,
                explanation="kb_query calls returned no chunk text to judge against",
            )

        body = (
            "Retrieved passages:\n"
            + "\n---\n".join(chunks)
            + "\n\n=== Agent response ===\n"
            + final
        )
        full_prompt = _FAITHFULNESS_RUBRIC + "\n\n" + body
        judge = make_judge(model=judge_model, position_swap=False)
        try:
            raw_score, reasoning = judge(prompt=full_prompt)
        except Exception as exc:
            return Score(value=NOANSWER, explanation=f"judge unavailable: {exc}")
        normalised = _normalise_1_to_5(raw_score, reasoning)
        return Score(
            value=normalised,
            explanation=(reasoning or "no reasoning provided"),
            metadata={
                "judge_model": judge_model,
                "raw_score": raw_score,
                "n_chunks": len(chunks),
            },
        )

    return score


# ---------------------------------------------------------------------------
# 5. attribution
# ---------------------------------------------------------------------------


def _url_anchor(url: str) -> str | None:
    """Cheap "did the agent reference this URL?" anchor.

    Pulls the host + first path segment, e.g.
    ``https://www.gnu.org/software/bash/manual/bash.html`` →
    ``gnu.org/software/bash``. Lets the heuristic catch references like
    "see gnu.org/software/bash" without the full URL.
    """

    m = re.match(r"https?://(?:www\.)?([^/\s?#]+)(/[^\s?#]*)?", url)
    if not m:
        return None
    host = m.group(1).lower()
    path = (m.group(2) or "").strip("/")
    if not path:
        return host
    # Take the first two path segments as the anchor.
    first_two = "/".join(path.split("/")[:2])
    return f"{host}/{first_two}".lower()


@scorer(metrics=[mean()], name="attribution")
def attribution() -> Scorer:
    """Does the agent's final response cite a retrieved source?

    Cheap, deterministic, no LLM call.

      * 1.0 — full ``source_url`` (or its host+path-prefix anchor) appears
        in the final message.
      * 0.5 — ``chunk_id`` or a non-empty ``section_path`` segment is
        referenced, but no URL.
      * 0.0 — no reference.

    NOANSWER if no `kb_query` calls happened.
    """

    async def score(state: TaskState, target: Target) -> Score:
        lab_agent = _get_lab_agent(state)
        calls = _kb_query_calls(lab_agent)
        if not calls:
            return Score(value=NOANSWER, explanation=f"no {_KB_TOOL} calls")

        urls: list[str] = []
        chunk_ids: list[str] = []
        sections: list[str] = []
        for call in calls:
            for hit in _hits_from_call(call):
                url = hit.get("source_url")
                if isinstance(url, str) and url.strip():
                    urls.append(url.strip())
                cid = hit.get("chunk_id")
                if cid is not None:
                    chunk_ids.append(str(cid))
                sp = hit.get("section_path")
                if isinstance(sp, list):
                    for seg in sp:
                        if isinstance(seg, str) and seg.strip():
                            sections.append(seg.strip())

        final = _final_assistant_message(state)
        if not final.strip():
            return Score(value=0.0, explanation="agent produced no final response")
        haystack = final.lower()

        # Step 1: full URL or url anchor match.
        for url in urls:
            if url.lower() in haystack:
                return Score(
                    value=1.0,
                    explanation=f"final message cites source URL {url!r}",
                )
            anchor = _url_anchor(url)
            if anchor and anchor in haystack:
                return Score(
                    value=1.0,
                    explanation=f"final message cites source via anchor {anchor!r}",
                )

        # Step 2: chunk_id or section path reference.
        for cid in chunk_ids:
            if cid and cid.lower() in haystack:
                return Score(
                    value=0.5,
                    explanation=f"final message references chunk_id {cid!r} but no URL",
                )
        for seg in sections:
            # Avoid matching trivially short section headers (e.g. "a").
            if len(seg) >= 4 and seg.lower() in haystack:
                return Score(
                    value=0.5,
                    explanation=f"final message references section {seg!r} but no URL",
                )

        return Score(value=0.0, explanation="no retrieved source referenced in response")

    return score


__all__ = [
    "attribution",
    "faithfulness",
    "mrr",
    "ndcg",
    "recall_at_k",
]
