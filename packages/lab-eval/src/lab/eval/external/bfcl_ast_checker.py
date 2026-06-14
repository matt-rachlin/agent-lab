"""Vendored, Python-only subset of the BFCL v3 AST checker.

Upstream:
    https://github.com/ShishirPatil/gorilla
    berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py

Vendored on 2026-05-27 against HEAD-of-main. License: Apache-2.0
(SPDX-License-Identifier: Apache-2.0; upstream NOTICE preserved below).
The vendored code is restricted to:

  * Python-language schemas (no Java / JavaScript / type-converter shims).
  * The ``simple``, ``multiple``, and ``parallel`` non-live categories
    of BFCL v3 (the ones whose ``ground_truth`` is a flat list of
    ``{func_name: {param: [allowed_values]}}`` dicts).
  * The ``simple_function_checker`` / ``multiple_function_checker`` /
    ``parallel_function_checker_no_order`` entry points.

What we deliberately do NOT vendor:

  * Live / multi-turn / agentic categories (they require BFCL's stateful
    simulator). Phase 17.5 ships AST categories; multi-turn is a follow-up.
  * Per-model name-mangling (the upstream ``MODEL_CONFIG_MAPPING`` /
    ``convert_func_name`` knob exists only because OpenAI's tool-name
    grammar rejects dots; we always run via LiteLLM with our own model
    ids and pass tool names through unchanged).
  * SQL / REST categories (different output shape, different grader).

NOTICE
------
Copyright 2024 Berkeley Function Calling Leaderboard authors. Licensed
under the Apache License, Version 2.0. See
https://github.com/ShishirPatil/gorilla/blob/main/LICENSE
"""

from __future__ import annotations

import re
from typing import Any

#: Python type-name → built-in type.
_PY_TYPE_MAPPING: dict[str, type] = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

#: Types whose ``items`` field carries the nested element type.
_NESTED_TYPE_CHECK: set[str] = {"array", "tuple"}


def _standardize_string(s: str) -> str:
    """Strip ' ,./-_*^', lowercase, normalise quotes.

    From upstream ``standardize_string``. Used to compare strings without
    punishing the model for ``"April 1, 2024"`` vs ``"April 1 2024"``.
    """

    return re.sub(r"[ ,./\-_*^]", "", s).lower().replace("'", '"')


def _string_checker(param: str, model_output: str, possible_answer: list[Any]) -> dict[str, Any]:
    pa: list[str] = []
    mo = _standardize_string(model_output)
    for entry in possible_answer:
        if isinstance(entry, str):
            pa.append(_standardize_string(entry))
    if mo not in pa:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {param!r}: {model_output!r}. "
                f"Expected one of {possible_answer}. Case insensitive."
            ],
            "error_type": "value_error:string",
        }
    return {"valid": True, "error": []}


def _list_checker(
    param: str, model_output: list[Any], possible_answer: list[Any]
) -> dict[str, Any]:
    mo = list(model_output)
    for i, item in enumerate(mo):
        if isinstance(item, str):
            mo[i] = _standardize_string(item)
    pa: list[list[Any]] = []
    for entry in possible_answer:
        bucket: list[Any] = []
        for item in entry:
            bucket.append(_standardize_string(item) if isinstance(item, str) else item)
        pa.append(bucket)
    if mo not in pa:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {param!r}: {model_output!r}. Expected one of {possible_answer}."
            ],
            "error_type": "value_error:list/tuple",
        }
    return {"valid": True, "error": []}


def _dict_checker(
    param: str, model_output: dict[Any, Any], possible_answers: list[Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for pa in possible_answers:
        if pa == "":
            continue
        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        for key, value in model_output.items():
            if key not in pa:
                result["valid"] = False
                result["error"].append(f"Unexpected dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break
            sv = _standardize_string(value) if isinstance(value, str) else value
            std_pa: list[Any] = []
            for cand in pa[key]:
                std_pa.append(_standardize_string(cand) if isinstance(cand, str) else cand)
            if sv not in std_pa:
                result["valid"] = False
                result["error"].append(
                    f"Invalid value for parameter {key!r}: {value!r}. Expected one of {std_pa}."
                )
                result["error_type"] = "value_error:dict_value"
                flag = False
                break
        if not flag:
            continue
        for key, values in pa.items():
            if key not in model_output and "" not in values:
                result["valid"] = False
                result["error"].append(f"Missing dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def _list_dict_checker(
    param: str, model_output: list[Any], possible_answers: list[Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": False,
        "error": [],
        "error_type": "list_dict_checker:unclear",
    }
    for answer in possible_answers:
        if len(model_output) != len(answer):
            result = {
                "valid": False,
                "error": ["Wrong number of dictionaries in the list."],
                "error_type": "value_error:list_dict_count",
            }
            continue
        flag = True
        for idx, mo_item in enumerate(model_output):
            sub = _dict_checker(param, mo_item, [answer[idx]])
            if not sub["valid"]:
                flag = False
                result = sub
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def _possible_answer_type(possible_answer: list[Any]) -> type | None:
    for answer in possible_answer:
        if answer != "":
            return type(answer)
    return None


def _type_checker(
    param: str,
    value: Any,
    possible_answer: list[Any],
    expected_type_description: str,
    expected_type_converted: type,
    nested_type_converted: type | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": True,
        "error": [],
        "is_variable": False,
        "error_type": "type_error:simple",
    }
    is_variable = False
    pa_type = _possible_answer_type(possible_answer)
    if pa_type is not None and pa_type != expected_type_converted:
        is_variable = True

    if type(value) is expected_type_converted:
        if nested_type_converted is None:
            result["is_variable"] = is_variable
            return result
        for pa_item in possible_answer:
            flag = True
            if isinstance(pa_item, list):
                # ``value`` is the type we just checked; for nested types
                # (array/tuple) that's a list/tuple and iteration is fine.
                assert isinstance(value, list | tuple)
                for v in value:
                    sub = _type_checker(
                        param, v, pa_item, str(nested_type_converted), nested_type_converted, None
                    )
                    if not sub["valid"]:
                        flag = False
                        break
            if flag:
                return {"valid": True, "error": [], "is_variable": is_variable}
        result["valid"] = False
        result["error"] = [
            f"Nested type checking failed for parameter {param!r}. Expected outer type "
            f"{expected_type_description} with inner type {nested_type_converted!s}. "
            f"Parameter value: {value!r}."
        ]
        result["error_type"] = "type_error:nested"
        return result

    # Variable fallback: the model produced a string that may stand in for
    # the literal value (e.g. ``API_KEY`` instead of the resolved key).
    if pa_type is not None and isinstance(value, pa_type):
        result["is_variable"] = True
        return result

    result["valid"] = False
    result["error"].append(
        f"Incorrect type for parameter {param!r}. Expected type "
        f"{expected_type_description}, got {type(value).__name__}. "
        f"Parameter value: {value!r}."
    )
    result["error_type"] = "type_error:simple"
    return result


def _find_description(
    func_descriptions: list[dict[str, Any]] | dict[str, Any], name: str
) -> dict[str, Any] | None:
    if isinstance(func_descriptions, list):
        for fd in func_descriptions:
            if fd.get("name") == name:
                return fd
        return None
    return func_descriptions


def simple_function_checker(
    func_description: dict[str, Any],
    model_output: dict[str, Any],
    possible_answer: dict[str, Any],
) -> dict[str, Any]:
    """Score one model function call against one possible-answer entry.

    Python-only; assumes the BFCL ``simple`` category schema. Returns a
    dict with ``valid: bool`` plus diagnostic ``error`` / ``error_type``.
    """

    pa_values = next(iter(possible_answer.values()))
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required = func_description["parameters"].get("required") or []

    result: dict[str, Any] = {
        "valid": True,
        "error": [],
        "error_type": None,  # success path returns this dict unchanged; failure paths set their own
    }

    if func_name not in model_output:
        return {
            "valid": False,
            "error": [f"Function name {func_name!r} not found in model output."],
            "error_type": "simple_function_checker:wrong_func_name",
        }
    model_params = model_output[func_name]

    for p in required:
        if p not in model_params:
            return {
                "valid": False,
                "error": [f"Missing required parameter: {p!r}."],
                "error_type": "simple_function_checker:missing_required",
            }

    for param, value in model_params.items():
        if param not in param_details or param not in pa_values:
            return {
                "valid": False,
                "error": [f"Unexpected parameter: {param!r}."],
                "error_type": "simple_function_checker:unexpected_param",
            }
        full = param_details[param]
        expected_desc = full["type"]
        expected_converted = _PY_TYPE_MAPPING.get(expected_desc, str)
        nested_converted: type | None = None
        if expected_desc in _NESTED_TYPE_CHECK:
            inner = full.get("items", {}).get("type", "any")
            nested_converted = _PY_TYPE_MAPPING.get(inner, str)
        # tuple-as-list & int→float auto-conversion: upstream behaviour.
        if expected_desc == "tuple" and isinstance(value, tuple):
            value = list(value)
        if expected_desc == "float" and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)

        tc = _type_checker(
            param, value, pa_values[param], expected_desc, expected_converted, nested_converted
        )
        is_variable = tc.get("is_variable", False)
        if not tc["valid"]:
            return tc

        if not is_variable:
            # The earlier _type_checker call has already verified that
            # ``value`` has type ``expected_converted``; the casts below
            # are for the type checker only.
            if expected_converted is dict and isinstance(value, dict):
                sub = _dict_checker(param, value, pa_values[param])
                if not sub["valid"]:
                    return sub
                continue
            if expected_converted is list and nested_converted is dict and isinstance(value, list):
                sub = _list_dict_checker(param, value, pa_values[param])
                if not sub["valid"]:
                    return sub
                continue
            if expected_converted is str and isinstance(value, str):
                sub = _string_checker(param, value, pa_values[param])
                if not sub["valid"]:
                    return sub
                continue
            if expected_converted is list and isinstance(value, list):
                sub = _list_checker(param, value, pa_values[param])
                if not sub["valid"]:
                    return sub
                continue

        if value not in pa_values[param]:
            return {
                "valid": False,
                "error": [
                    f"Invalid value for parameter {param!r}: {value!r}. "
                    f"Expected one of {pa_values[param]}."
                ],
                "error_type": "value_error:others",
            }

    for param in pa_values:
        if param not in model_params and "" not in pa_values[param]:
            return {
                "valid": False,
                "error": [f"Optional parameter {param!r} not provided and not marked as optional."],
                "error_type": "simple_function_checker:missing_optional",
            }

    return result


def multiple_function_checker(
    func_descriptions: list[dict[str, Any]],
    model_output: list[dict[str, Any]],
    possible_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    """``multiple`` category: one ground-truth call, exactly one model call."""

    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "multiple_function_checker:wrong_count",
        }
    expected_name = next(iter(possible_answers[0].keys()))
    fd = _find_description(func_descriptions, expected_name)
    if fd is None:
        return {
            "valid": False,
            "error": [f"No description for expected function {expected_name!r}."],
            "error_type": "multiple_function_checker:missing_description",
        }
    return simple_function_checker(fd, model_output[0], possible_answers[0])


def parallel_function_checker_no_order(
    func_descriptions: list[dict[str, Any]],
    model_output: list[dict[str, Any]],
    possible_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    """``parallel`` category: many calls, order-insensitive."""

    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "parallel_function_checker_no_order:wrong_count",
        }
    matched: list[int] = []
    last_result: dict[str, Any] = {"valid": False, "error": [], "error_type": ""}
    for i, pa in enumerate(possible_answers):
        expected_name = next(iter(pa.keys()))
        fd = _find_description(func_descriptions, expected_name)
        if fd is None:
            return {
                "valid": False,
                "error": [f"No description for expected function {expected_name!r}."],
                "error_type": "parallel_function_checker_no_order:missing_description",
            }
        accumulated: list[Any] = []
        found = False
        for idx, mo in enumerate(model_output):
            if idx in matched:
                continue
            last_result = simple_function_checker(fd, mo, pa)
            if last_result["valid"]:
                matched.append(idx)
                found = True
                break
            accumulated.append({f"index_{idx}": last_result.get("error")})
        if not found:
            return {
                "valid": False,
                "error": [
                    f"Could not match possible-answer index {i} ({expected_name}).",
                    *accumulated,
                ],
                "error_type": "parallel_function_checker_no_order:cannot_find_match",
            }
    return {"valid": True, "error": []}


def ast_checker(
    func_description: list[dict[str, Any]] | dict[str, Any],
    model_output: list[dict[str, Any]],
    possible_answer: list[dict[str, Any]],
    *,
    test_category: str,
) -> dict[str, Any]:
    """Dispatch to the right category-specific checker.

    Args:
        func_description: BFCL example's ``function`` field (list of OpenAPI
            tool schemas).
        model_output: list of ``{func_name: {arg: value, ...}}`` dicts —
            the model's calls flattened from ``tool_calls``.
        possible_answer: BFCL example's ``ground_truth`` list.
        test_category: one of ``"simple"``, ``"multiple"``, ``"parallel"``,
            ``"parallel_multiple"``.
    """

    if not isinstance(func_description, list):
        func_description = [func_description]

    if "parallel" in test_category:
        return parallel_function_checker_no_order(func_description, model_output, possible_answer)
    if "multiple" in test_category:
        return multiple_function_checker(func_description, model_output, possible_answer)
    # "simple" — exactly one call.
    if len(model_output) != 1:
        return {
            "valid": False,
            "error": [f"Wrong number of functions (got {len(model_output)}, expected 1)."],
            "error_type": "simple_function_checker:wrong_count",
        }
    return simple_function_checker(func_description[0], model_output[0], possible_answer[0])


__all__ = [
    "ast_checker",
    "multiple_function_checker",
    "parallel_function_checker_no_order",
    "simple_function_checker",
]
