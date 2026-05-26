"""Inspect Scorers for agent runs (Phase 6e).

Four scorer factories — each is an Inspect `@scorer` so the harness
records its value per-cell:

  * `end_state(predicate)` — checks post-run workspace files or DB state
    against a small predicate schema (`workspace_file_equals`,
    `workspace_file_contains`, `workspace_file_exists`, `db_query`).
  * `tool_correctness()` — given a `TaskRubric` of type `tool_call`,
    asserts the agent actually invoked `target_tool` with a superset
    of `expected_args`.
  * `budget_respected()` — `actual_turns ≤ max_turns AND tool_call_count
    ≤ tool_budget AND terminated_reason ∉ {budget_exhausted,
    max_turns_reached}`.
  * `trajectory_judge(judge_model)` — LLM-as-judge over the full
    trajectory; wraps `lab.eval.judge.make_judge` (no position-swap;
    there is no A/B here). Returns NOANSWER on judge transport failure
    rather than scoring 0 — silence on a flaky judge is louder than a
    fake fail.

Each scorer reads the lab `Task` from `state.metadata["lab_task"]` and
the per-turn trajectory from `state.metadata["lab_agent"]`. Workspace
contents come from `state.metadata["lab_agent"]["workspace_snapshot"]`
which the solver writes before its sandbox tears down (the sandbox is
gone by the time scoring runs — see solver.py).
"""

from __future__ import annotations

import json
import re
from typing import Any

import psycopg
from inspect_ai.scorer import NOANSWER, Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from lab.eval.judge import make_judge
from lab.settings import get_settings

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _get_lab_task(state: TaskState) -> Any:
    """Pull the lab Task off `state.metadata`. Raises if missing."""

    md = state.metadata or {}
    task = md.get("lab_task")
    if task is None:
        raise RuntimeError("scorer invoked without 'lab_task' in state.metadata")
    return task


def _get_lab_agent(state: TaskState) -> dict[str, Any]:
    """Pull the trajectory dict off `state.metadata`."""

    md = state.metadata or {}
    return md.get("lab_agent") or {}


def _decode_snapshot(snapshot: dict[str, Any] | None, path: str) -> bytes | None:
    """Resolve a path from the workspace snapshot.

    The solver stashes bytes (or `None`) under `lab_agent.workspace_snapshot`.
    On the round-trip through Inspect's log serialisation, bytes may
    survive as bytes OR be coerced to a base64-ish string — we accept both
    and return `None` if the path isn't present.
    """

    if not snapshot:
        return None
    if path not in snapshot:
        # tolerate leading-slash mismatch
        alt = path.lstrip("/")
        if alt not in snapshot:
            return None
        path = alt
    value = snapshot[path]
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace")
    # Anything else — try to coerce
    return str(value).encode("utf-8", errors="replace")


_WRITE_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|COPY)\b",
    re.IGNORECASE,
)


def _looks_like_write(query: str) -> bool:
    """Defensive check: refuse to run a 'read-only' predicate that contains writes."""

    return bool(_WRITE_RE.search(query))


def _eval_workspace_predicate(
    predicate: dict[str, Any], snapshot: dict[str, Any] | None
) -> tuple[float, str]:
    """Evaluate a `workspace_file_*` predicate against the snapshot.

    Returns (score, explanation). Score is 1.0 on pass, 0.0 on fail.
    """

    ptype = predicate.get("type")
    path = predicate.get("path")
    if not path or not isinstance(path, str):
        return 0.0, f"predicate {ptype!r} missing required 'path'"

    body = _decode_snapshot(snapshot, path)

    if ptype == "workspace_file_exists":
        if body is None:
            return 0.0, f"file {path!r} not present in workspace"
        return 1.0, f"file {path!r} present ({len(body)} bytes)"

    if ptype == "workspace_file_equals":
        expected = predicate.get("expected")
        if expected is None:
            return 0.0, "predicate missing 'expected'"
        if body is None:
            return 0.0, f"file {path!r} not present"
        expected_bytes = expected.encode("utf-8") if isinstance(expected, str) else expected
        # If predicate has case_sensitive=False, compare case-insensitively.
        case_sensitive = predicate.get("case_sensitive", True)
        if not case_sensitive:
            ok = body.decode(errors="replace").strip().lower() == (
                expected_bytes.decode(errors="replace").strip().lower()
            )
        else:
            ok = body.strip() == expected_bytes.strip()
        if ok:
            return 1.0, f"file {path!r} matched expected"
        return (
            0.0,
            f"file {path!r} did not match expected "
            f"(got {body[:80]!r}, expected {expected_bytes[:80]!r})",
        )

    if ptype == "workspace_file_contains":
        substring = predicate.get("substring")
        if substring is None:
            return 0.0, "predicate missing 'substring'"
        if body is None:
            return 0.0, f"file {path!r} not present"
        text = body.decode("utf-8", errors="replace")
        needle = substring if isinstance(substring, str) else substring.decode("utf-8")
        case_sensitive = predicate.get("case_sensitive", True)
        ok = (needle in text) if case_sensitive else (needle.lower() in text.lower())
        if ok:
            return 1.0, f"file {path!r} contained {needle!r}"
        return 0.0, f"file {path!r} did not contain {needle!r}"

    return 0.0, f"unknown workspace predicate type {ptype!r}"


def _eval_db_predicate(predicate: dict[str, Any]) -> tuple[float, str]:
    """Run a `db_query` predicate against Postgres."""

    query = predicate.get("query")
    if not query or not isinstance(query, str):
        return 0.0, "db_query predicate missing 'query'"
    if _looks_like_write(query):
        return 0.0, "db_query predicate rejected: contains write-like keywords"
    expects_rows = predicate.get("expects_rows")
    try:
        with (
            psycopg.connect(get_settings().pg_dsn) as conn,
            conn.cursor() as cur,
        ):
            cur.execute(query)
            rows = cur.fetchall()
    except Exception as exc:
        return 0.0, f"db_query failed: {type(exc).__name__}: {exc}"
    n = len(rows)
    if expects_rows is None:
        # Pass if any row came back.
        if n >= 1:
            return 1.0, f"db_query returned {n} row(s)"
        return 0.0, "db_query returned no rows"
    if int(expects_rows) == n:
        return 1.0, f"db_query returned exactly {n} row(s) (expected)"
    return 0.0, f"db_query returned {n} row(s); expected {expects_rows}"


# ---------------------------------------------------------------------------
# scorers
# ---------------------------------------------------------------------------


@scorer(metrics=[mean()], name="end_state")
def end_state(predicate: dict[str, Any] | None = None) -> Scorer:
    """Score the final workspace / DB state against a predicate.

    The predicate defaults to `task.success_predicate`; the explicit
    argument exists for tests and unusual sweeps that want to override
    per-cell. When neither is set the scorer returns NOANSWER.

    Predicate schema (one of):
        type: workspace_file_exists, path: "x"
        type: workspace_file_equals, path: "x", expected: "...", case_sensitive: bool
        type: workspace_file_contains, path: "x", substring: "...", case_sensitive: bool
        type: db_query, query: "SELECT ...", expects_rows: <int|null>
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Resolve the active predicate: explicit arg wins, else task field.
        active = predicate
        if active is None:
            try:
                task = _get_lab_task(state)
            except RuntimeError as exc:
                return Score(value=NOANSWER, explanation=str(exc))
            active = getattr(task, "success_predicate", None)
        if not active:
            return Score(
                value=NOANSWER,
                explanation="no success_predicate configured for this task",
            )

        ptype = active.get("type") if isinstance(active, dict) else None
        if ptype is None:
            return Score(value=0.0, explanation="predicate missing 'type'")

        lab_agent = _get_lab_agent(state)
        snapshot = lab_agent.get("workspace_snapshot") if lab_agent else None

        if ptype in {
            "workspace_file_exists",
            "workspace_file_equals",
            "workspace_file_contains",
        }:
            value, explanation = _eval_workspace_predicate(active, snapshot)
            return Score(value=value, explanation=explanation)
        if ptype == "db_query":
            value, explanation = _eval_db_predicate(active)
            return Score(value=value, explanation=explanation)
        # Unknown predicate type — return NOANSWER (not 0.0) so a scorer
        # mismatch (e.g. `end_state` paired with a RAG `retrieval_recall`
        # predicate) reads as "not applicable" rather than a false fail.
        # F-005 EXP-002 follow-up: `end_state` was scoring 0.0 on every
        # `retrieval_recall` task, polluting the end_state mean.
        return Score(
            value=NOANSWER,
            explanation=f"end_state not applicable to predicate type {ptype!r}",
        )

    return score


def _tool_calls_from_trajectory(lab_agent: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the per-turn tool-call list out of a trajectory dict.

    Each turn carries a `tool_calls` list (the solver builds it; see
    `_execute_tool_calls`). We return `[{tool, args, ...}, ...]` so the
    correctness check is a flat scan.
    """

    out: list[dict[str, Any]] = []
    for turn in lab_agent.get("turns") or []:
        for tc in turn.get("tool_calls") or []:
            out.append(tc)
    return out


def _arguments_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Expected args must be a subset of actual args (extras allowed)."""

    if not isinstance(actual, dict):
        return False
    for k, v in expected.items():
        if k not in actual:
            return False
        if actual[k] != v:
            return False
    return True


@scorer(metrics=[mean()], name="tool_correctness")
def tool_correctness() -> Scorer:
    """Did the agent invoke `target_tool` with the expected args?

    Reads `task.rubric` — only fires for rubrics of type `tool_call`.
    Returns NOANSWER for non-`tool_call` rubrics so the scorer can be
    safely included in a default list without polluting other tasks.
    """

    async def score(state: TaskState, target: Target) -> Score:
        try:
            task = _get_lab_task(state)
        except RuntimeError as exc:
            return Score(value=NOANSWER, explanation=str(exc))
        rubric = getattr(task, "rubric", None)
        if rubric is None:
            return Score(value=NOANSWER, explanation="no rubric on task")
        rtype = getattr(rubric, "type", None)
        if rtype != "tool_call":
            return Score(value=NOANSWER, explanation="not a tool_call rubric")
        target_tool = getattr(rubric, "target_tool", None)
        if not target_tool:
            return Score(value=0.0, explanation="tool_call rubric missing target_tool")
        expected_args = getattr(rubric, "expected_args", None) or {}

        lab_agent = _get_lab_agent(state)
        flat_calls = _tool_calls_from_trajectory(lab_agent)
        for tc in flat_calls:
            if tc.get("tool") != target_tool:
                continue
            args = tc.get("args") or {}
            # The solver may have truncated args; honour the truncation marker.
            if isinstance(args, dict) and args.get("_truncated"):
                # Best-effort: with truncation we can't validate keys; fail clean.
                continue
            if _arguments_match(args, expected_args):
                return Score(
                    value=1.0,
                    explanation=(
                        f"tool {target_tool!r} called with matching args "
                        f"(expected {expected_args!r})"
                    ),
                )
        # No matching call.
        seen: list[str] = sorted({str(tc.get("tool")) for tc in flat_calls if tc.get("tool")})
        return Score(
            value=0.0,
            explanation=(
                f"no call to {target_tool!r} with expected args matched; "
                f"observed tools={seen}"
            ),
        )

    return score


@scorer(metrics=[mean()], name="budget_respected")
def budget_respected() -> Scorer:
    """Did the agent stay within `max_turns` and `tool_budget`?

    Reads the task's caps and compares to the actual usage recorded in
    the trajectory. Fails (0.0) if either cap was hit, or if the loop
    terminated via `budget_exhausted` / `max_turns_reached`. The
    explanation names the busted budget so the failure is greppable.
    """

    async def score(state: TaskState, target: Target) -> Score:
        try:
            task = _get_lab_task(state)
        except RuntimeError as exc:
            return Score(value=NOANSWER, explanation=str(exc))
        lab_agent = _get_lab_agent(state)
        max_turns = int(getattr(task, "max_turns", 1) or 1)
        tool_budget = int(getattr(task, "tool_budget", 0) or 0)
        actual_turns = int(lab_agent.get("actual_turns") or 0)
        tool_calls = int(lab_agent.get("tool_call_count") or 0)
        terminated = lab_agent.get("terminated_reason") or "unknown"

        problems: list[str] = []
        if actual_turns > max_turns:
            problems.append(f"actual_turns={actual_turns} > max_turns={max_turns}")
        if tool_calls > tool_budget:
            problems.append(f"tool_call_count={tool_calls} > tool_budget={tool_budget}")
        if terminated in {"budget_exhausted", "max_turns_reached"}:
            problems.append(f"terminated_reason={terminated}")
        if problems:
            return Score(value=0.0, explanation="; ".join(problems))
        return Score(
            value=1.0,
            explanation=(
                f"within budget: turns={actual_turns}/{max_turns}, "
                f"tool_calls={tool_calls}/{tool_budget}, terminated={terminated}"
            ),
        )

    return score


def _format_trajectory_for_judge(
    state: TaskState, lab_agent: dict[str, Any], *, per_turn_cap: int = 2048
) -> str:
    """Render the trajectory into a compact text block for the judge.

    Layout:
      Task: <input>
      ---
      Turn 0:
        assistant: <content, truncated>
        tool_calls:
          - <tool>(<args>) -> <result truncated>
      ...
      Final response: <last assistant content>
    """

    try:
        task = _get_lab_task(state)
        task_input = getattr(task, "input", None) or state.input
    except RuntimeError:
        task_input = state.input

    lines: list[str] = [f"Task: {task_input}", "---"]
    turns = lab_agent.get("turns") or []
    for entry in turns:
        idx = entry.get("turn")
        lines.append(f"Turn {idx}:")
        preview = entry.get("content_preview")
        if isinstance(preview, dict) and preview.get("_truncated"):
            preview = preview.get("preview")
        if preview:
            text = str(preview)
            if len(text) > per_turn_cap:
                text = text[:per_turn_cap] + "…"
            lines.append(f"  assistant: {text}")
        tool_calls = entry.get("tool_calls") or []
        if tool_calls:
            lines.append("  tool_calls:")
            for tc in tool_calls:
                args = tc.get("args")
                result = tc.get("result")
                # Truncate big payloads.
                try:
                    args_repr = json.dumps(args, default=str)[:per_turn_cap // 4]
                except Exception:
                    args_repr = str(args)[:per_turn_cap // 4]
                try:
                    result_repr = json.dumps(result, default=str)[:per_turn_cap // 2]
                except Exception:
                    result_repr = str(result)[:per_turn_cap // 2]
                lines.append(f"    - {tc.get('tool')}({args_repr}) -> {result_repr}")
        if entry.get("error"):
            lines.append(f"  error: {entry['error']}")
    # Last assistant message from state.messages, best-effort.
    final = ""
    for msg in reversed(state.messages or []):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "assistant":
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if isinstance(content, str) and content.strip():
                final = content
                break
    if final:
        if len(final) > per_turn_cap:
            final = final[:per_turn_cap] + "…"
        lines.append("---")
        lines.append(f"Final response: {final}")
    return "\n".join(lines)


_JUDGE_RUBRIC = (
    "You are evaluating an AI agent's behaviour on a tool-use task.\n"
    "Score the agent's trajectory on a 1-5 integer scale:\n"
    "  5 — agent's behaviour clearly solves the task; tool calls were correct and minimal\n"
    "  4 — agent's behaviour likely solves the task; minor issues\n"
    "  3 — agent made plausible progress but the outcome is unclear\n"
    "  2 — agent took relevant action but very likely failed\n"
    "  1 — agent did not engage with the task or took clearly wrong action\n"
    "Reply with a JSON object: "
    '{"score": <int 1-5>, "reasoning": "<one short sentence>"}.\n'
    "Output JSON only — no preamble or commentary outside the JSON."
)


@scorer(metrics=[mean()], name="trajectory_judge")
def trajectory_judge(judge_model: str = "gpt-oss-120b-cloud") -> Scorer:
    """LLM-as-judge over the full trajectory; returns score/5 in [0,1].

    Uses the lab's existing `make_judge` helper; no position-swap
    (judging a single trajectory, not an A/B pair). On transport failure
    against LiteLLM, returns NOANSWER with an explanation rather than
    silently scoring 0 — the caller can decide whether to treat that as
    success or failure.
    """

    async def score(state: TaskState, target: Target) -> Score:
        lab_agent = _get_lab_agent(state)
        prompt_body = _format_trajectory_for_judge(state, lab_agent)
        full_prompt = _JUDGE_RUBRIC + "\n\n" + prompt_body
        judge = make_judge(model=judge_model, position_swap=False)
        try:
            raw_score, reasoning = judge(prompt=full_prompt)
        except Exception as exc:
            return Score(value=NOANSWER, explanation=f"judge unavailable: {exc}")
        # make_judge clamps to [0,1]. The judge here returns 1-5, so the
        # tolerant parser will have already turned that into 0.0..1.0 via
        # the `score: N` regex; but if the judge obeyed the rubric exactly
        # we get a value already in [0,1] (since the parser clamps). For
        # the 1-5 case the parser yields raw int (clamped to 1.0 for any
        # int > 1), so we re-normalise by parsing the raw reasoning if
        # present.
        # Robust approach: normalise (1..5) → (0..1) if the parsed value
        # is suspiciously the clamped boundary AND reasoning suggests a
        # 1-5 reply. Otherwise pass through.
        normalised = _normalise_1_to_5(raw_score, reasoning)
        return Score(
            value=normalised,
            explanation=(reasoning or "no reasoning provided"),
            metadata={"judge_model": judge_model, "raw_score": raw_score},
        )

    return score


def _normalise_1_to_5(raw: float, reasoning: str | None) -> float:
    """Re-derive a 1-5 score from the judge's reasoning text when the parser clamped it.

    `parse_judge_response` clamps any value > 1 to 1.0, which destroys
    the 1-5 signal we asked for. We sniff the reasoning for an explicit
    `"score": N` where N is in {1..5} and rescale; otherwise the raw
    parsed value (already in [0,1]) wins.
    """

    if reasoning:
        # Look for an integer "score": N (the most common shape we asked for).
        m = re.search(r'"score"\s*:\s*([1-5])\b', reasoning)
        if m:
            return int(m.group(1)) / 5.0
        m = re.search(r"\bscore\s*[:=]\s*([1-5])\b", reasoning, re.IGNORECASE)
        if m:
            return int(m.group(1)) / 5.0
    # Otherwise the parser already gave us a [0,1] value (or clamped 1.0
    # for a 1-5 reply that was a 5; assume the judge meant 5 in that case
    # and treat as 1.0). That preserves the existing tolerant-parser
    # behaviour for judges that obey the 0-1 schema.
    return max(0.0, min(1.0, float(raw)))


__all__ = [
    "budget_respected",
    "end_state",
    "tool_correctness",
    "trajectory_judge",
]
