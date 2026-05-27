"""Task registry — load YAML/JSON task suites, persist to the lab.tasks table.

A "task" is one evaluable unit: input + (gold answer | rubric) + metadata.
A "suite" is a named collection (e.g. PBS-v0.1, smoke, BFCL-v3-subset).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import psycopg
import yaml
from lab.core.settings import get_settings
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskRubric(BaseModel):
    """How to score a task's output."""

    model_config = ConfigDict(extra="allow")

    type: Literal["exact_match", "regex", "json_schema", "tool_call", "llm_judge", "custom"]
    case_sensitive: bool = False
    pattern: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    judge_model: str | None = None
    judge_prompt: str | None = None
    target_tool: str | None = None
    expected_args: dict[str, Any] | None = None


class Task(BaseModel):
    """A single evaluable task."""

    model_config = ConfigDict(extra="forbid")

    suite: str
    slug: str
    category: str | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    external_id: str | None = None
    description: str | None = None

    # Required input
    input: str
    system: str | None = None
    tools: list[dict[str, Any]] | None = None

    # Agent loop knobs (Phase 6). Defaults preserve single-turn no-tool behavior.
    max_turns: int = 1
    tool_budget: int = 0
    success_predicate: dict[str, Any] | None = None
    sandbox: dict[str, Any] | None = None

    # Either gold_answer (for exact_match etc) or rubric (for richer scoring)
    gold_answer: str | None = None
    rubric: TaskRubric | None = None

    @field_validator("max_turns")
    @classmethod
    def _validate_max_turns(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_turns must be >= 1")
        return v

    @field_validator("tool_budget")
    @classmethod
    def _validate_tool_budget(cls, v: int) -> int:
        if v < 0:
            raise ValueError("tool_budget must be >= 0")
        return v


def load_tasks(path: Path) -> list[Task]:
    """Load a task suite from a YAML or JSON file.

    Format: top-level dict with `suite` and `tasks` keys, OR a list of full task objects.
    """
    text = path.read_text(encoding="utf-8")
    raw: Any
    if path.suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    elif path.suffix == ".json":
        raw = json.loads(text)
    elif path.suffix == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raise ValueError(f"unsupported task file extension: {path.suffix}")

    if isinstance(raw, dict) and "tasks" in raw:
        suite = raw["suite"]
        defaults: dict[str, Any] = {k: v for k, v in raw.items() if k not in {"suite", "tasks"}}
        rows: list[Task] = []
        for entry in raw["tasks"]:
            merged = {**defaults, "suite": suite, **entry}
            rows.append(Task.model_validate(merged))
        return rows
    if isinstance(raw, list):
        return [Task.model_validate(entry) for entry in raw]
    raise ValueError("file must be a dict with `tasks` key or a top-level list")


_UPSERT_SQL = """
INSERT INTO tasks (suite, external_id, slug, category, difficulty, payload, added_at)
VALUES (%(suite)s, %(external_id)s, %(slug)s, %(category)s, %(difficulty)s, %(payload)s, NOW())
ON CONFLICT (suite, slug) DO UPDATE SET
    external_id = EXCLUDED.external_id,
    category = EXCLUDED.category,
    difficulty = EXCLUDED.difficulty,
    payload = EXCLUDED.payload,
    retired_at = NULL;
"""


def register_tasks(tasks: Iterable[Task]) -> int:
    """Upsert tasks into the lab.tasks table. Returns the number registered."""
    rows = list(tasks)
    if not rows:
        return 0
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        for task in rows:
            payload = {
                "input": task.input,
                "system": task.system,
                "tools": task.tools,
                "max_turns": task.max_turns,
                "tool_budget": task.tool_budget,
                "success_predicate": task.success_predicate,
                "sandbox": task.sandbox,
                "gold_answer": task.gold_answer,
                "rubric": task.rubric.model_dump(by_alias=True) if task.rubric else None,
                "description": task.description,
            }
            cur.execute(
                _UPSERT_SQL,
                {
                    "suite": task.suite,
                    "external_id": task.external_id,
                    "slug": task.slug,
                    "category": task.category,
                    "difficulty": task.difficulty,
                    "payload": json.dumps(payload),
                },
            )
    return len(rows)


def list_suites() -> list[tuple[str, int]]:
    """Return [(suite, count), ...] for all active task suites."""
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT suite, COUNT(*) FROM tasks WHERE retired_at IS NULL GROUP BY suite ORDER BY suite"
        )
        return [(r[0], r[1]) for r in cur.fetchall()]


def get_tasks(suite: str, slugs: list[str] | None = None) -> list[dict[str, Any]]:
    """Fetch tasks from a suite (optionally filtered to specific slugs)."""
    sql = (
        "SELECT task_id, suite, slug, category, difficulty, payload "
        "FROM tasks WHERE suite = %s AND retired_at IS NULL"
    )
    params: list[Any] = [suite]
    if slugs:
        sql += " AND slug = ANY(%s)"
        params.append(slugs)
    sql += " ORDER BY slug"
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [
            {
                "task_id": row[0],
                "suite": row[1],
                "slug": row[2],
                "category": row[3],
                "difficulty": row[4],
                "payload": row[5],
            }
            for row in cur.fetchall()
        ]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Load a task suite into the lab.tasks table")
    parser.add_argument("path", type=Path, help="YAML/JSON/JSONL file of tasks")
    args = parser.parse_args()

    tasks = load_tasks(args.path)
    n = register_tasks(tasks)
    print(f"registered {n} task(s) from {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
