"""BFCL perturbed-twin contamination check (EXP-013 P2.R2 followup).

The 2026-06-14 perfect-order audit (wave-2 contamination + research-rigor
review) flagged that the EXP-013 BFCL +19pp result rests on:

  1. The training mix contains 11,341 records (56.7%) using the same
     OpenAI function-call envelope BFCL grades (xLAM/Hermes-FC/ToolACE).
  2. BFCL rows are not in training verbatim — but distribution overlap
     was never quantified.
  3. The contamination-check protocol section 5 (perturbed twin per
     Xu et al. 2024 MMLU-CF) was never executed against BFCL v3.

This script implements section 5 specifically for the function-calling
domain. The intuition: if the FT model has memorized the BFCL
distribution rather than learned the format, a perturbed twin of the
same task — same answer structure, different surface form — should
pass significantly LESS often than the original.

Perturbation strategy (deterministic, no LLM in the perturbation loop):

  * Rename every function in the tools list deterministically
    (e.g. ``calculate_triangle_area`` -> ``f_38b1a``) — slug derived
    from sha256(salt + name).
  * Rename every parameter to a slug.
  * Rewrite the user prompt in lockstep (any reference to the original
    function or parameter name is rewritten to the new slug).
  * Rewrite the BFCL ``ground_truth`` ``[{fn: {arg: [...]}}]`` structure
    in lockstep — so the AST checker still accepts the same answer for
    a model that GENUINELY understands the task structure.

A model that has memorised the BFCL distribution will produce an
answer keyed to the ORIGINAL function names and parameters — that
answer no longer matches the perturbed ground truth, so the AST check
fails. A model that learned the FORMAT will rename in lockstep with
the prompt and still pass.

Output: pass_rate_orig - pass_rate_twin is the contamination signal.

  ~0pp gap   -> robust to distribution memorization; +19pp BFCL holds.
  large gap  -> the FT model is keyed on names, not structure;
                the +19pp claim is bounded by memorization.

Usage:

  uv run python scripts/bfcl_perturbed_twin.py --model qwen3-4b --n 3       # smoke
  uv run python scripts/bfcl_perturbed_twin.py --model qwen3-4b-ft-toolcall-q4-latest --n 50

A smoke run verifies the pipeline + LiteLLM lane without burning meaningful
GPU. The full 50-task probe is 2 arms x ~2s/cell on the ft arm = <5 min.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from jobs_status import Job

# Lazy-import lab so the script is usable from outside a sweep context
from lab.eval.external import bfcl as bfcl_loader

# ---------------------------------------------------------------------
# Perturbation

_SLUG_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _stable_slug(seed: str, prefix: str = "f_") -> str:
    """Deterministic 5-char slug derived from `seed`. Same seed -> same slug."""
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    body = "".join(_SLUG_ALPHABET[int(c, 16) % len(_SLUG_ALPHABET)] for c in h[:5])
    return f"{prefix}{body}"


@dataclass
class Twin:
    rename_map: dict[str, str]
    tools: list[dict[str, Any]]
    prompt: str
    ground_truth: list[dict[str, Any]]


def _rename_in_text(text: str, rename_map: dict[str, str]) -> str:
    out = text
    # Longest names first to avoid prefix collisions.
    for old in sorted(rename_map, key=len, reverse=True):
        new = rename_map[old]
        out = re.sub(rf"\b{re.escape(old)}\b", new, out)
    return out


def make_twin(task: Any, *, salt: str = "bfcl-twin-2026") -> Twin:
    """Build a perturbed twin of one BFCL LabTask."""
    raw_tools: list[dict[str, Any]] = list(task.tools or [])
    raw_prompt: str = task.input or ""
    raw_gt: list[dict[str, Any]] = list(task.rubric.ground_truth or [])

    rename: dict[str, str] = {}

    # Function names (tools)
    for t in raw_tools:
        fn = t.get("function", t)
        name = fn.get("name")
        if isinstance(name, str) and name not in rename:
            rename[name] = _stable_slug(salt + ":fn:" + name)

    # Function names from ground_truth (defensive)
    for gt in raw_gt:
        for name in gt:
            if name not in rename:
                rename[name] = _stable_slug(salt + ":fn:" + name)

    # Parameter names — collect from tools.parameters.properties
    for t in raw_tools:
        fn = t.get("function", t)
        params = fn.get("parameters") or {}
        for prop in params.get("properties") or {}:
            if prop not in rename:
                rename[prop] = _stable_slug(salt + ":arg:" + prop, prefix="p_")
    # And from ground_truth arg names
    for gt in raw_gt:
        for fn_args in gt.values():
            for arg in fn_args:
                if arg not in rename:
                    rename[arg] = _stable_slug(salt + ":arg:" + arg, prefix="p_")

    # Build perturbed tools
    twin_tools: list[dict[str, Any]] = []
    for t in raw_tools:
        new_t = json.loads(json.dumps(t))
        fn = new_t.get("function", new_t)
        if isinstance(fn.get("name"), str):
            fn["name"] = rename.get(fn["name"], fn["name"])
        params = fn.get("parameters") or {}
        old_props = params.get("properties") or {}
        if old_props:
            params["properties"] = {rename.get(k, k): v for k, v in old_props.items()}
        old_required = params.get("required") or []
        params["required"] = [rename.get(r, r) for r in old_required]
        if isinstance(fn.get("description"), str):
            fn["description"] = _rename_in_text(fn["description"], rename)
        twin_tools.append(new_t)

    # Perturbed prompt (lockstep substitution)
    twin_prompt = _rename_in_text(raw_prompt, rename)

    # Perturbed ground_truth
    twin_gt: list[dict[str, Any]] = []
    for gt in raw_gt:
        new_gt: dict[str, Any] = {}
        for fn_name, args in gt.items():
            new_args = {rename.get(k, k): v for k, v in args.items()}
            new_gt[rename.get(fn_name, fn_name)] = new_args
        twin_gt.append(new_gt)

    return Twin(rename_map=rename, tools=twin_tools, prompt=twin_prompt, ground_truth=twin_gt)


# ---------------------------------------------------------------------
# LLM lane

LITELLM_URL = os.environ.get("LAB_LITELLM_URL", "http://127.0.0.1:4000")


def _read_litellm_key() -> str:
    p = Path("/data/lab/services/litellm-master-key")
    if p.exists():
        return p.read_text().strip()
    return os.environ.get("LITELLM_MASTER_KEY", "dummy")


def call_model(
    *,
    model: str,
    prompt: str,
    tools: list[dict[str, Any]],
    system: str | None = None,
    timeout_sec: float = 90.0,
) -> list[dict[str, Any]]:
    """Single-turn function-calling, return raw tool_calls list."""
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {_read_litellm_key()}",
        "Content-Type": "application/json",
    }
    r = httpx.post(
        f"{LITELLM_URL}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout_sec,
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return msg.get("tool_calls") or []


def _sample_tasks(tasks: list[Any], n: int) -> list[Any]:
    # Deterministic sample: sorted by slug, take first n. Reproducible without
    # a seed parameter, which matters for a contamination diagnostic.
    return sorted(tasks, key=lambda t: t.slug)[:n]


def _unwrap_raw_functions(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI [{type:function, function:{...}}] -> BFCL raw [{name,description,parameters}]."""
    out: list[dict[str, Any]] = []
    for t in openai_tools:
        out.append(t.get("function", t))
    return out


def _evaluate(
    *,
    raw_functions: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    category: str,
) -> bool:
    if not tool_calls:
        return False
    try:
        result = bfcl_loader.grade_bfcl_response(
            raw_functions=raw_functions,
            ground_truth=ground_truth,
            tool_calls=tool_calls,
            category=category,
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return bool(result.get("valid", False))


def run_arm(
    tasks: list[Any],
    *,
    model: str,
    bar_advance: Any,
    log_fn: Any,
) -> tuple[int, int, list[dict[str, Any]]]:
    orig_pass = 0
    twin_pass = 0
    rows: list[dict[str, Any]] = []
    for i, t in enumerate(tasks):
        twin = make_twin(t)
        cat = t.rubric.bfcl_category or t.category or "simple"

        orig_tools = list(t.tools or [])
        orig_raw_fns = _unwrap_raw_functions(orig_tools)
        twin_raw_fns = _unwrap_raw_functions(twin.tools)

        # Original arm
        try:
            orig_calls = call_model(
                model=model,
                prompt=t.input or "",
                tools=orig_tools,
                system=t.system,
            )
            orig_ok = _evaluate(
                raw_functions=orig_raw_fns,
                tool_calls=orig_calls,
                ground_truth=list(t.rubric.ground_truth or []),
                category=cat,
            )
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            orig_ok = False
            log_fn(f"[{i + 1}] {t.slug} orig error: {e}")

        # Twin arm
        try:
            twin_calls = call_model(
                model=model,
                prompt=twin.prompt,
                tools=twin.tools,
                system=t.system,
            )
            twin_ok = _evaluate(
                raw_functions=twin_raw_fns,
                tool_calls=twin_calls,
                ground_truth=twin.ground_truth,
                category=cat,
            )
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            twin_ok = False
            log_fn(f"[{i + 1}] {t.slug} twin error: {e}")

        orig_pass += int(orig_ok)
        twin_pass += int(twin_ok)
        rows.append(
            {
                "slug": t.slug,
                "category": cat,
                "orig_pass": orig_ok,
                "twin_pass": twin_ok,
                "rename_count": len(twin.rename_map),
            }
        )
        bar_advance(1)
        log_fn(f"[{i + 1}/{len(tasks)}] {t.slug} orig={orig_ok} twin={twin_ok}")
    return orig_pass, twin_pass, rows


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="BFCL perturbed-twin contamination probe.")
    p.add_argument("--model", required=True, help="LiteLLM model name")
    p.add_argument("--n", type=int, default=50, help="Number of BFCL tasks (default 50)")
    p.add_argument("--categories", nargs="*", default=None, help="Override default categories")
    p.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = p.parse_args(list(argv) if argv is not None else None)

    cats = args.categories or list(bfcl_loader.DEFAULT_CATEGORIES)
    tasks = bfcl_loader.load_bfcl_tasks(cats)
    sampled = _sample_tasks(tasks, args.n)
    print(f"Loaded {len(tasks)} BFCL tasks; sampling {len(sampled)} for probe.")
    print(f"Model: {args.model}")

    with Job(f"bfcl-twin {args.model} n={len(sampled)}") as job:
        bar_obj = job.bar("tasks", total=len(sampled))
        orig_pass, twin_pass, rows = run_arm(
            sampled,
            model=args.model,
            bar_advance=bar_obj.advance,
            log_fn=job.log,
        )

    n = len(sampled)
    orig_rate = orig_pass / n if n else 0.0
    twin_rate = twin_pass / n if n else 0.0
    gap = orig_rate - twin_rate

    result = {
        "model": args.model,
        "n": n,
        "orig_pass": orig_pass,
        "twin_pass": twin_pass,
        "orig_pass_rate": orig_rate,
        "twin_pass_rate": twin_rate,
        "gap_pp": round(gap * 100, 1),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.out}")

    if gap >= 0.10:
        print(
            f"\n[CONTAMINATION-FLAG] gap {gap * 100:+.1f}pp >= 10pp — distribution memorization likely"
        )
    elif gap >= 0.05:
        print(
            f"\n[CONTAMINATION-MAYBE] gap {gap * 100:+.1f}pp in [5,10) — soft signal; check sample size"
        )
    else:
        print(
            f"\n[CONTAMINATION-CLEAN] gap {gap * 100:+.1f}pp < 5pp — format generalization defensible"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
