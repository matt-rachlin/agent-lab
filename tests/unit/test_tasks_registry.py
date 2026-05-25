"""Task registry parsing tests (no DB)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.tasks.registry import Task, load_tasks

YAML_DICT_FORMAT = """
suite: ut
category: ut
tasks:
  - slug: a1
    input: "what is 2+2?"
    gold_answer: "4"
  - slug: a2
    difficulty: hard
    input: "harder one"
    rubric:
      type: regex
      pattern: "answer"
"""


def test_load_yaml_with_dict_root(tmp_path: Path) -> None:
    p = tmp_path / "tasks.yaml"
    p.write_text(YAML_DICT_FORMAT, encoding="utf-8")
    rows = load_tasks(p)
    assert len(rows) == 2
    assert rows[0].suite == "ut"
    assert rows[0].slug == "a1"
    assert rows[0].gold_answer == "4"
    assert rows[1].difficulty == "hard"
    assert rows[1].rubric is not None
    assert rows[1].rubric.type == "regex"


def test_extra_forbid() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Task.model_validate(
            {
                "suite": "x",
                "slug": "y",
                "input": "z",
                "surprise": "rejected",
            }
        )


def test_load_unknown_extension(tmp_path: Path) -> None:
    p = tmp_path / "tasks.xyz"
    p.write_text("ignore", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_tasks(p)
