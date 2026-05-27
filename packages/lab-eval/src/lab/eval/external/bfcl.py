"""BFCL v3 adapter (Phase 17.5).

Turns Berkeley Function Calling Leaderboard v3 examples into lab ``Task``
rows + a deterministic scorer (``bfcl_ast_match``) that can ride the
existing sweep harness.

Adoption choice (see Phase 17.5 EXP-005 pre-reg, Option B): we vendor a
Python-only subset of BFCL's own AST grader rather than spawn the
upstream evaluator binary or rely on the upstream HuggingFace
``Dataset`` shape. Each example becomes one ``Task`` whose
``rubric.type`` is ``"bfcl_ast"`` — the sweep runner dispatches such
cells to ``_execute_bfcl_cell`` which issues a single tool-calling
LiteLLM request and grades the response off-band.

Out of scope (Phase 17.5):

  * ``live_*`` categories — different upstream schemas, more variable
    output shape.
  * ``multi_turn_*`` categories — require BFCL's stateful simulator.
  * ``sql`` / ``rest`` / ``java`` / ``javascript`` categories — different
    grader path.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lab.eval.external.bfcl_ast_checker import ast_checker
from lab.tasks.registry import Task as LabTask
from lab.tasks.registry import TaskRubric

_HF_RAW_BASE = (
    "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
)

#: Map a BFCL category slug to (questions filename, possible-answer filename).
#: Order of insertion matches the upstream BFCL leaderboard non-live AST suite.
_CATEGORY_FILES: dict[str, tuple[str, str]] = {
    "simple": ("BFCL_v3_simple.json", "possible_answer/BFCL_v3_simple.json"),
    "multiple": ("BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json"),
    "parallel": ("BFCL_v3_parallel.json", "possible_answer/BFCL_v3_parallel.json"),
    "parallel_multiple": (
        "BFCL_v3_parallel_multiple.json",
        "possible_answer/BFCL_v3_parallel_multiple.json",
    ),
}

#: The four non-live AST categories we ship in Phase 17.5.
DEFAULT_CATEGORIES: tuple[str, ...] = ("simple", "multiple", "parallel", "parallel_multiple")

SUITE_NAME = "bfcl-v3-ast"


def dataset_root() -> Path:
    """Where the raw BFCL data lives. Overridable via ``LAB_BFCL_DATA_DIR``."""

    override = os.environ.get("LAB_BFCL_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "datasets" / "bfcl-v3"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with urllib.request.urlopen(url) as resp:  # noqa: S310 — known HF Hub URL
        body = resp.read()
    dest.write_bytes(body)


def fetch_category(
    category: str, *, root: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Download (if missing) and parse one BFCL category.

    Returns ``(examples, answers_by_id)``. Each example is a dict with
    ``id``, ``question``, ``function`` per upstream schema; each answer
    is the corresponding ``ground_truth`` payload keyed by id.
    """

    if category not in _CATEGORY_FILES:
        raise KeyError(f"unknown BFCL category {category!r}; known: {sorted(_CATEGORY_FILES)}")
    q_name, a_name = _CATEGORY_FILES[category]
    root = root or dataset_root()
    q_path = root / q_name
    a_path = root / a_name
    _download(f"{_HF_RAW_BASE}/{q_name}", q_path)
    _download(f"{_HF_RAW_BASE}/{a_name}", a_path)

    examples: list[dict[str, Any]] = []
    for line in q_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        examples.append(json.loads(line))

    answers: dict[str, dict[str, Any]] = {}
    for line in a_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        answers[row["id"]] = row

    return examples, answers


def _to_litellm_tool_spec(fn: dict[str, Any]) -> dict[str, Any]:
    """Wrap a BFCL ``function`` block in the OpenAI tool-call envelope.

    BFCL examples use ``parameters.properties[*].type ∈ {"string",
    "integer", "float", "boolean", "array", "dict", "any", "tuple"}``.
    OpenAI's tool schema expects JSON-Schema types: we translate the
    Python-specific ones (``float`` → ``number``, ``dict`` → ``object``,
    ``tuple`` → ``array``, ``any`` → ``string``).
    """

    def _convert_props(props: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in props.items():
            if not isinstance(v, dict):
                out[k] = v
                continue
            v2 = dict(v)
            t = v2.get("type")
            if t == "float":
                v2["type"] = "number"
            elif t == "dict":
                v2["type"] = "object"
            elif t == "tuple":
                v2["type"] = "array"
            elif t == "any":
                v2["type"] = "string"
            if "items" in v2 and isinstance(v2["items"], dict):
                v2["items"] = _convert_props({"x": v2["items"]})["x"]
            if "properties" in v2 and isinstance(v2["properties"], dict):
                v2["properties"] = _convert_props(v2["properties"])
            out[k] = v2
        return out

    params = fn.get("parameters") or {}
    parameters = {
        "type": "object",
        "properties": _convert_props(params.get("properties") or {}),
        "required": params.get("required") or [],
    }
    return {
        "type": "function",
        "function": {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": parameters,
        },
    }


def _bfcl_system_prompt() -> str:
    return (
        "You are a function-calling assistant. The user will pose one task. "
        "You have access to the provided tools. Respond by issuing exactly the "
        "tool calls needed to accomplish the task. Do not produce any natural-"
        "language answer; emit only tool calls. If the task asks for multiple "
        "operations, emit multiple tool calls."
    )


def bfcl_example_to_task(
    example: dict[str, Any], answer: dict[str, Any], *, category: str
) -> LabTask:
    """Build a lab ``Task`` row from one BFCL example.

    The lab ``Task.input`` is the BFCL ``question`` flattened to plain
    text. ``Task.tools`` carries the LiteLLM-shaped tool list (not our
    MCP tool list — the BFCL cell path bypasses the MCP machinery).
    ``Task.rubric`` is a ``custom`` rubric with the bookkeeping fields
    the sweep runner + scorer use:

      * ``bfcl_category`` — passed to the AST checker.
      * ``ground_truth`` — the answer payload, kept on the task so the
        scorer can fire deterministically without re-loading the file.
    """

    # ``question`` is a list of conversation turns. Each turn is a list
    # of messages (the BFCL multi-turn variants nest deeper). For the
    # AST categories every example collapses to one user message.
    q = example["question"]
    if isinstance(q, list) and q and isinstance(q[0], list):
        msgs = [m for turn in q for m in turn]
    elif isinstance(q, list):
        msgs = q
    else:
        msgs = [{"role": "user", "content": str(q)}]
    user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") in {"user", "human"}]
    text = "\n\n".join(m.get("content", "") for m in user_msgs).strip() or json.dumps(msgs)

    fns = example["function"]
    if isinstance(fns, dict):
        fns = [fns]
    tools_spec = [_to_litellm_tool_spec(fn) for fn in fns]

    slug = example["id"].replace("/", "-").replace(":", "-")

    # `TaskRubric` has ConfigDict(extra="allow"), so the BFCL-specific
    # bookkeeping fields (bfcl_category / bfcl_id / ground_truth /
    # raw_functions) flow through. We use type="custom" because the
    # TaskRubric.type Literal does not enumerate "bfcl_ast"; the runner
    # dispatcher recognises BFCL cells by the presence of
    # rubric.bfcl_category, not by the .type field.
    rubric_obj = TaskRubric.model_validate(
        {
            "type": "custom",
            "bfcl_category": category,
            "bfcl_id": example["id"],
            "ground_truth": answer.get("ground_truth"),
            "raw_functions": fns,
        }
    )

    return LabTask(
        suite=SUITE_NAME,
        slug=slug,
        category=category,
        external_id=example["id"],
        description=f"BFCL v3 {category} example {example['id']}",
        input=text,
        system=_bfcl_system_prompt(),
        tools=tools_spec,
        # BFCL cells use a single chat-completion request, but the runner
        # routes ``max_turns > 1`` to the agent path. We keep this at 1
        # so the new ``_execute_bfcl_cell`` dispatcher picks the BFCL
        # path explicitly via the rubric type.
        max_turns=1,
        tool_budget=len(fns) + 4,  # informational; scorer enforces nothing
        rubric=rubric_obj,
    )


def load_bfcl_tasks(
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    *,
    limit_per_category: int | None = None,
    root: Path | None = None,
) -> list[LabTask]:
    """Load BFCL examples and return them as lab Tasks.

    Args:
        categories: which categories to load.
        limit_per_category: optional cap (smallest N first, by id sort).
            Used by EXP-005 if we want a sampled subset.
        root: dataset root (testing override).
    """

    tasks: list[LabTask] = []
    for cat in categories:
        examples, answers = fetch_category(cat, root=root)
        if limit_per_category is not None:
            examples = sorted(examples, key=lambda e: e["id"])[:limit_per_category]
        for ex in examples:
            ans = answers.get(ex["id"])
            if ans is None:
                # Skip examples missing a ground truth (shouldn't happen
                # in the AST categories but be defensive).
                continue
            tasks.append(bfcl_example_to_task(ex, ans, category=cat))
    return tasks


def grade_bfcl_response(
    *,
    raw_functions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    """Score one model response against the BFCL ground truth.

    ``tool_calls`` is the OpenAI / LiteLLM ``response.choices[0].message
    .tool_calls`` list. Each entry's ``function.arguments`` is a JSON
    string — we parse it and flatten into BFCL's expected
    ``[{name: {arg: value, ...}}, ...]`` shape.
    """

    flat: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        args_raw = fn.get("arguments")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except json.JSONDecodeError:
            return {
                "valid": False,
                "error": [f"non-JSON arguments for {name!r}: {args_raw!r}"],
                "error_type": "model_output:json_parse",
            }
        if not isinstance(args, dict):
            return {
                "valid": False,
                "error": [f"arguments for {name!r} not a JSON object: {args!r}"],
                "error_type": "model_output:not_object",
            }
        flat.append({name: args})

    if not flat:
        return {
            "valid": False,
            "error": ["model emitted zero tool calls"],
            "error_type": "model_output:no_tool_call",
        }

    return ast_checker(raw_functions, flat, ground_truth, test_category=category)


__all__ = [
    "DEFAULT_CATEGORIES",
    "SUITE_NAME",
    "bfcl_example_to_task",
    "dataset_root",
    "fetch_category",
    "grade_bfcl_response",
    "load_bfcl_tasks",
]
