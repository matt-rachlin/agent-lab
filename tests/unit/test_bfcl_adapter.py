"""Unit tests for the BFCL → lab Task adapter (Phase 17.5)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from lab.eval.external.bfcl import (
    DEFAULT_CATEGORIES,
    SUITE_NAME,
    bfcl_example_to_task,
    grade_bfcl_response,
    load_bfcl_tasks,
)


def test_default_categories_are_the_four_ast_ones():
    assert set(DEFAULT_CATEGORIES) == {"simple", "multiple", "parallel", "parallel_multiple"}


def test_suite_name_constant():
    assert SUITE_NAME == "bfcl-v3-ast"


def _fixture_example(prefix: str = "simple") -> tuple[dict, dict]:
    ex = {
        "id": f"{prefix}_0",
        "question": [[{"role": "user", "content": "Find the area of a triangle (b=10, h=5)."}]],
        "function": [
            {
                "name": "calculate_triangle_area",
                "description": "Calc triangle area.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer", "description": "Base."},
                        "height": {"type": "integer", "description": "Height."},
                        "unit": {"type": "string", "description": "Unit."},
                    },
                    "required": ["base", "height"],
                },
            }
        ],
    }
    ans = {
        "id": f"{prefix}_0",
        "ground_truth": [
            {
                "calculate_triangle_area": {
                    "base": [10],
                    "height": [5],
                    "unit": ["units", ""],
                }
            }
        ],
    }
    return ex, ans


def test_example_to_task_carries_rubric_fields():
    ex, ans = _fixture_example("simple")
    task = bfcl_example_to_task(ex, ans, category="simple")
    assert task.suite == "bfcl-v3-ast"
    assert task.slug == "simple_0"
    assert task.category == "simple"
    assert task.external_id == "simple_0"
    assert task.system is not None
    assert "function-calling" in task.system.lower()
    assert "triangle" in task.input.lower()
    assert task.tools is not None
    assert len(task.tools) == 1
    # OpenAI tool envelope shape.
    t = task.tools[0]
    assert t["type"] == "function"
    assert t["function"]["name"] == "calculate_triangle_area"
    # 'dict' must be translated to 'object' for the JSON Schema layer.
    schema = t["function"]["parameters"]
    assert schema["type"] == "object"
    assert "base" in schema["properties"]
    # Rubric carries the AST grader bookkeeping.
    rubric = task.rubric
    assert rubric is not None
    rubric_extra = rubric.model_dump(by_alias=True)
    assert rubric_extra["bfcl_category"] == "simple"
    assert rubric_extra["bfcl_id"] == "simple_0"
    assert rubric_extra["ground_truth"] == ans["ground_truth"]
    assert isinstance(rubric_extra["raw_functions"], list)


def test_load_bfcl_tasks_from_local_fixture_root(tmp_path):
    # Build a minimal on-disk fixture that mimics the HF Hub layout.
    ex, ans = _fixture_example("simple")
    (tmp_path / "BFCL_v3_simple.json").write_text(json.dumps(ex) + "\n", encoding="utf-8")
    (tmp_path / "possible_answer").mkdir()
    (tmp_path / "possible_answer" / "BFCL_v3_simple.json").write_text(
        json.dumps(ans) + "\n", encoding="utf-8"
    )

    tasks = load_bfcl_tasks(["simple"], root=tmp_path)
    assert len(tasks) == 1
    assert tasks[0].slug == "simple_0"
    assert tasks[0].external_id == "simple_0"


def test_grade_bfcl_response_pass():
    ex, ans = _fixture_example("simple")
    fns = ex["function"]
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "calculate_triangle_area",
                "arguments": json.dumps({"base": 10, "height": 5, "unit": "units"}),
            },
        }
    ]
    res = grade_bfcl_response(
        raw_functions=fns,
        ground_truth=ans["ground_truth"],
        tool_calls=tool_calls,
        category="simple",
    )
    assert res["valid"] is True


def test_grade_bfcl_response_fail_no_tool_calls():
    ex, ans = _fixture_example("simple")
    res = grade_bfcl_response(
        raw_functions=ex["function"],
        ground_truth=ans["ground_truth"],
        tool_calls=[],
        category="simple",
    )
    assert res["valid"] is False
    assert res["error_type"] == "model_output:no_tool_call"


def test_grade_bfcl_response_fail_wrong_value():
    ex, ans = _fixture_example("simple")
    tc = [
        {
            "type": "function",
            "function": {
                "name": "calculate_triangle_area",
                "arguments": json.dumps({"base": 999, "height": 5}),
            },
        }
    ]
    res = grade_bfcl_response(
        raw_functions=ex["function"],
        ground_truth=ans["ground_truth"],
        tool_calls=tc,
        category="simple",
    )
    assert res["valid"] is False


def test_grade_bfcl_response_handles_non_json_arguments():
    ex, ans = _fixture_example("simple")
    tc = [
        {
            "type": "function",
            "function": {
                "name": "calculate_triangle_area",
                "arguments": "not-a-json-string",
            },
        }
    ]
    res = grade_bfcl_response(
        raw_functions=ex["function"],
        ground_truth=ans["ground_truth"],
        tool_calls=tc,
        category="simple",
    )
    assert res["valid"] is False
    assert res["error_type"] == "model_output:json_parse"


def test_unknown_category_raises():
    from lab.eval.external.bfcl import fetch_category

    with pytest.raises(KeyError):
        fetch_category("not-a-real-category", root=Path(tempfile.mkdtemp()))
