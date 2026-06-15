"""Live BFCL refutation battery for the ADR-008 verifier (Stage 0b #9).

Deliberately an INDEPENDENT execution + grading path from the production sweep
runner — re-grading through different code is itself a battery requirement. Runs
the subject across seeds + prompt variants plus class-spanning anchors, and
returns a BatteryResult for lab.platform.verifier.verdict(). v0 thresholds are
documented inline and tunable.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import httpx
from lab.platform.verifier import BatteryResult

from lab.core.llm import call_litellm_chat
from lab.core.settings import get_settings
from lab.eval.external.bfcl import grade_bfcl_response, load_bfcl_tasks

_VARIANTS = [
    "You are a function-calling assistant. Respond by issuing exactly the tool calls needed.",
    "You are a tool-using agent. Use the provided functions to complete the user's request.",
    "Act as an API caller: select and invoke the appropriate function(s) for the task.",
    "You have tools available. Call the correct function with the correct arguments.",
    "Function-calling mode: emit the tool call(s) that satisfy the request.",
]

_CALL_ERRORS = (httpx.HTTPError, OSError, KeyError, ValueError, TypeError)


def _tool_choice_for(model: str) -> str:
    """Per-model tool_choice (the F-017 fix): 'required' wherever the backend
    supports it, 'auto' only for Ollama-served models which reject 'required'."""
    import psycopg

    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT backend FROM models WHERE litellm_id = %s LIMIT 1", (model,))
        row = cur.fetchone()
    backend = (row[0] if row else "") or ""
    return "auto" if "ollama" in backend.lower() else "required"


@dataclass
class _Task:
    input: str
    system: str
    tools: list[dict[str, Any]]
    raw_functions: list[dict[str, Any]]
    ground_truth: list[dict[str, Any]]
    category: str


def _load(n: int) -> list[_Task]:
    cats = ["simple", "multiple", "parallel", "parallel_multiple"]
    per = max(1, n // len(cats))
    out: list[_Task] = []
    for t in load_bfcl_tasks(cats, limit_per_category=per):
        r = t.rubric
        out.append(
            _Task(
                input=t.input,
                system=t.system or "",
                tools=list(t.tools or []),
                raw_functions=list(getattr(r, "raw_functions", []) or []),
                ground_truth=list(getattr(r, "ground_truth", []) or []),
                category=str(getattr(r, "bfcl_category", "simple")),
            )
        )
    return out[:n]


def _extract_calls(resp: dict[str, Any]) -> list[dict[str, Any]]:
    msg = ((resp.get("choices") or [{}])[0]).get("message") or {}
    calls = msg.get("tool_calls") or []
    return calls if isinstance(calls, list) else []


def _independent_grade(
    tool_calls: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
) -> bool:
    """2nd, independent re-grade path: did the model call the expected function
    name(s)? Name-match only — intentionally simpler than the AST checker, to
    surface checker-specific artefacts."""
    if not tool_calls or not ground_truth:
        return False
    called: set[str] = set()
    for tc in tool_calls:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            called.add(str(fn["name"]))
    expected: set[str] = set()
    for gt in ground_truth:
        if isinstance(gt, dict):
            expected.update(str(k) for k in gt)
    return bool(expected) and expected.issubset(called)


def _call(
    settings: Any, key: str, model: str, task: _Task, *, seed: int, system: str, tool_choice: str
) -> tuple[bool, bool, bool]:
    """Returns (ast_pass, indep_pass, emitted)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task.input},
    ]
    try:
        resp, _ = call_litellm_chat(
            settings=settings,
            litellm_key=key,
            model=model,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=2048,
            tools=task.tools or None,
            tool_choice=tool_choice,
            extra={"seed": seed},
            timeout=120,
        )
    except _CALL_ERRORS:
        return (False, False, False)
    calls = _extract_calls(resp)
    grade = grade_bfcl_response(
        raw_functions=task.raw_functions,
        ground_truth=task.ground_truth,
        tool_calls=calls,
        category=task.category,
    )
    return (bool(grade.get("valid")), _independent_grade(calls, task.ground_truth), bool(calls))


def _acc(flags: list[bool]) -> float:
    return (sum(flags) / len(flags)) if flags else 0.0


def run_bfcl_battery(
    *,
    subject: str,
    subject_tool_choice: str,
    anchors_by_class: dict[str, list[str]],
    n_tasks: int = 30,
    seeds: int = 16,
    variants: int = 5,
) -> BatteryResult:
    settings = get_settings()
    key = settings.litellm_key
    tasks = _load(n_tasks)

    per_seed_acc: list[float] = []
    regrade_pairs: list[bool] = []
    for s in range(1, seeds + 1):
        passes: list[bool] = []
        for t in tasks:
            ast, indep, _ = _call(
                settings, key, subject, t, seed=s, system=t.system, tool_choice=subject_tool_choice
            )
            passes.append(ast)
            regrade_pairs.append(ast == indep)
        per_seed_acc.append(_acc(passes))

    per_variant_acc: list[float] = []
    for v in _VARIANTS[:variants]:
        passes = []
        for t in tasks:
            ast, indep, _ = _call(
                settings, key, subject, t, seed=1, system=v, tool_choice=subject_tool_choice
            )
            passes.append(ast)
            regrade_pairs.append(ast == indep)
        per_variant_acc.append(_acc(passes))

    anchors_per_class: dict[str, int] = {}
    class_emission: dict[str, float] = {}
    for cls, models in anchors_by_class.items():
        anchors_per_class[cls] = len(models)
        emit_rates: list[float] = []
        for m in models:
            tc = _tool_choice_for(m)
            flags = [
                _call(settings, key, m, t, seed=1, system=t.system, tool_choice=tc)[2]
                for t in tasks
            ]
            emit_rates.append(_acc(flags))
        class_emission[cls] = statistics.mean(emit_rates) if emit_rates else 0.0

    seed_spread = (max(per_seed_acc) - min(per_seed_acc)) if per_seed_acc else 1.0
    variant_spread = (max(per_variant_acc) - min(per_variant_acc)) if per_variant_acc else 1.0
    return BatteryResult(
        n_seeds=seeds,
        seed_effect_holds=seed_spread <= 0.05,
        n_prompt_variants=variants,
        variant_effect_holds=variant_spread <= 0.15,
        n_regrade_paths=2,
        regraders_agree=_acc(regrade_pairs) >= 0.90,
        anchors_per_class=anchors_per_class,
        anchor_consistent=len(class_emission) >= 2
        and all(e > 0.5 for e in class_emission.values()),
    )
