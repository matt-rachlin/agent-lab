"""Unit tests for the vendored BFCL AST checker (Phase 17.5).

We pin behaviour against the upstream BFCL grader semantics by
exercising each category path with hand-built fixtures that match the
on-disk schema of BFCL_v3_simple/multiple/parallel/parallel_multiple
JSON Lines files.
"""

from __future__ import annotations

import pytest

from lab.eval.external.bfcl_ast_checker import (
    ast_checker,
    multiple_function_checker,
    parallel_function_checker_no_order,
    simple_function_checker,
)

_SIMPLE_FUNC = {
    "name": "calculate_triangle_area",
    "description": "Calculate the area of a triangle given its base and height.",
    "parameters": {
        "type": "dict",
        "properties": {
            "base": {"type": "integer", "description": "The base."},
            "height": {"type": "integer", "description": "The height."},
            "unit": {"type": "string", "description": "Unit (default 'units')."},
        },
        "required": ["base", "height"],
    },
}

_SIMPLE_GROUND_TRUTH = {
    "calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]},
}


def test_simple_pass_all_required_fields():
    out = {"calculate_triangle_area": {"base": 10, "height": 5, "unit": "units"}}
    res = simple_function_checker(_SIMPLE_FUNC, out, _SIMPLE_GROUND_TRUTH)
    assert res["valid"] is True


def test_simple_pass_optional_omitted():
    out = {"calculate_triangle_area": {"base": 10, "height": 5}}
    res = simple_function_checker(_SIMPLE_FUNC, out, _SIMPLE_GROUND_TRUTH)
    assert res["valid"] is True


def test_simple_fail_wrong_func_name():
    out = {"calc_triangle": {"base": 10, "height": 5}}
    res = simple_function_checker(_SIMPLE_FUNC, out, _SIMPLE_GROUND_TRUTH)
    assert res["valid"] is False
    assert res["error_type"] == "simple_function_checker:wrong_func_name"


def test_simple_fail_missing_required():
    out = {"calculate_triangle_area": {"base": 10}}  # height missing
    res = simple_function_checker(_SIMPLE_FUNC, out, _SIMPLE_GROUND_TRUTH)
    assert res["valid"] is False
    assert res["error_type"] == "simple_function_checker:missing_required"


def test_simple_fail_wrong_value():
    out = {"calculate_triangle_area": {"base": 99, "height": 5, "unit": "units"}}
    res = simple_function_checker(_SIMPLE_FUNC, out, _SIMPLE_GROUND_TRUTH)
    assert res["valid"] is False


def test_simple_int_to_float_autocoerce():
    func = {
        "name": "f",
        "parameters": {
            "type": "dict",
            "properties": {"x": {"type": "float"}},
            "required": ["x"],
        },
    }
    gt = {"f": {"x": [1.0]}}
    out = {"f": {"x": 1}}  # int — should be promoted to float
    res = simple_function_checker(func, out, gt)
    assert res["valid"] is True


def test_string_case_insensitive_via_standardize():
    func = {
        "name": "g",
        "parameters": {
            "type": "dict",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    gt = {"g": {"city": ["San Francisco", "SF"]}}
    out = {"g": {"city": "san francisco"}}
    res = simple_function_checker(func, out, gt)
    assert res["valid"] is True


def test_multiple_picks_correct_function():
    funcs = [
        _SIMPLE_FUNC,
        {
            "name": "calculate_rectangle_area",
            "parameters": {
                "type": "dict",
                "properties": {"w": {"type": "integer"}, "h": {"type": "integer"}},
                "required": ["w", "h"],
            },
        },
    ]
    out = [{"calculate_triangle_area": {"base": 10, "height": 5}}]
    gt = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    res = multiple_function_checker(funcs, out, gt)
    assert res["valid"] is True


def test_parallel_no_order_two_calls():
    func = {
        "name": "f",
        "parameters": {
            "type": "dict",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    }
    funcs = [func]
    out = [{"f": {"x": 2}}, {"f": {"x": 1}}]
    gt = [{"f": {"x": [1]}}, {"f": {"x": [2]}}]
    res = parallel_function_checker_no_order(funcs, out, gt)
    assert res["valid"] is True


def test_ast_checker_dispatch_simple_wrong_count():
    out = [
        {"calculate_triangle_area": {"base": 10, "height": 5}},
        {"calculate_triangle_area": {"base": 11, "height": 5}},
    ]
    res = ast_checker([_SIMPLE_FUNC], out, [_SIMPLE_GROUND_TRUTH], test_category="simple")
    assert res["valid"] is False
    assert res["error_type"] == "simple_function_checker:wrong_count"


@pytest.mark.parametrize(
    ("category", "expected_call_count"),
    [("simple", 1), ("multiple", 1), ("parallel", 2)],
)
def test_ast_checker_dispatches_to_each_category(category, expected_call_count):
    func = {
        "name": "f",
        "parameters": {
            "type": "dict",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    }
    out = [{"f": {"x": 1}}] * expected_call_count
    gt = [{"f": {"x": [1]}}] * expected_call_count
    res = ast_checker([func], out, gt, test_category=category)
    assert res["valid"] is True
