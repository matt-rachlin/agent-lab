"""Unit tests for `lab.inspect_bridge.scorers.rag.attribution`."""

from __future__ import annotations

import asyncio
from typing import Any

from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import NOANSWER, Target
from inspect_ai.solver import TaskState

from lab.inspect_bridge.scorers.rag import attribution
from lab.tasks.registry import Task


def _state(
    *,
    calls: list[list[dict[str, Any]]],
    final: str,
) -> TaskState:
    task = Task.model_validate(
        {
            "suite": "test",
            "slug": "attr",
            "input": "hi",
            "max_turns": 3,
            "tool_budget": 5,
        }
    )
    turns: list[dict[str, Any]] = []
    for i, hits in enumerate(calls):
        turns.append(
            {
                "turn": i,
                "tool_calls": [
                    {
                        "tool": "kb_query",
                        "args": {"kb_name": "bash", "question": "q"},
                        "result": {"hits": hits, "kb_status": "ok"},
                    }
                ],
            }
        )
    lab_agent = {"turns": turns}
    return TaskState(
        model="x",
        sample_id="s",
        epoch=0,
        input="hi",
        messages=[
            ChatMessageUser(content="hi"),
            ChatMessageAssistant(content=final),
        ],
        metadata={"lab_task": task, "lab_agent": lab_agent},
    )


def _hit(
    *,
    chunk_id: str = "c1",
    source_url: str = "https://www.gnu.org/software/bash/manual/bash.html",
    section: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "source_url": source_url,
        "section_path": section or ["Redirections"],
        "text": "passage",
        "score": 0.5,
    }


def _run(s: Any, state: TaskState) -> Any:
    return asyncio.run(s(state, Target("")))


def test_attribution_full_url_match() -> None:
    state = _state(
        calls=[[_hit()]],
        final="See https://www.gnu.org/software/bash/manual/bash.html for details.",
    )
    out = _run(attribution(), state)
    assert out.value == 1.0


def test_attribution_anchor_match() -> None:
    """Host + first path segment is enough to count as a citation."""

    state = _state(
        calls=[[_hit()]],
        final="Per the bash manual on gnu.org/software/bash, redirections use > and <.",
    )
    out = _run(attribution(), state)
    assert out.value == 1.0


def test_attribution_chunk_id_only() -> None:
    state = _state(
        calls=[[_hit(chunk_id="bash-redir-001")]],
        final="See chunk bash-redir-001 for the canonical answer.",
    )
    out = _run(attribution(), state)
    assert out.value == 0.5


def test_attribution_section_only() -> None:
    state = _state(
        calls=[[_hit(section=["Redirections", "stderr"])]],
        final="The Redirections section in the manual covers this.",
    )
    out = _run(attribution(), state)
    assert out.value == 0.5


def test_attribution_no_reference() -> None:
    state = _state(
        calls=[[_hit()]],
        final="Bash uses 2>&1 to merge stderr into stdout.",
    )
    out = _run(attribution(), state)
    assert out.value == 0.0


def test_attribution_noanswer_when_no_kb_calls() -> None:
    state = _state(calls=[], final="Some response.")
    out = _run(attribution(), state)
    assert out.value == NOANSWER


def test_attribution_url_match_wins_over_chunk_match() -> None:
    """Full URL match returns 1.0 even when chunk_id also appears."""

    state = _state(
        calls=[[_hit(chunk_id="cid1", source_url="https://example.com/page/a")]],
        final="From example.com/page/a reference (also cid1).",
    )
    out = _run(attribution(), state)
    assert out.value == 1.0


def test_attribution_ignores_short_section_segments() -> None:
    """Three-char section names like 'foo' shouldn't trigger spurious 0.5."""

    state = _state(
        calls=[[_hit(chunk_id="zzz999", section=["foo"], source_url="https://x.com/")]],
        final="Nothing relevant here.",
    )
    out = _run(attribution(), state)
    assert out.value == 0.0
