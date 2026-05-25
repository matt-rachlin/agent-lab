"""Evaluator framework — decorator registry + Postgres-backed result persistence.

An evaluator is a function that takes a (run, task) pair and returns an
EvalResult. Evaluators may optionally take a `judge` keyword (LLM-as-judge),
in which case they are routed through the judge-call infrastructure.

Usage:

    from lab.eval import EvalResult, evaluator

    @evaluator(name="my_check", version="1.0", threshold=0.8)
    def my_check(run: RunRow, task: TaskRow) -> EvalResult:
        passed = some_test(run.response_text, task.gold_answer)
        return EvalResult.scored(1.0 if passed else 0.0, reasoning="...")
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import psycopg
from psycopg.types.json import Json

from lab.settings import get_settings

# ---------------------------------------------------------------------------
# Row types — minimal projections of the underlying tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRow:
    """One row from experiment_runs joined with model name + trace pointer."""

    run_id: str
    experiment_id: int | None
    model_id: int
    model_litellm_id: str
    task_id: int
    seed: int
    status: str
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: int | None
    cost_usd: float | None
    trace_path: str | None
    response_text: str | None


@dataclass(frozen=True)
class TaskRow:
    """One row from tasks plus its parsed payload."""

    task_id: int
    suite: str
    slug: str
    category: str | None
    difficulty: str | None
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalResult:
    """Outcome of evaluating one (run, task)."""

    score: float | None = None  # None when skipped
    passed: bool | None = None
    skipped: bool = False
    skip_reason: str | None = None
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def scored(
        cls,
        score: float,
        *,
        passed: bool | None = None,
        reasoning: str | None = None,
        **metadata: Any,
    ) -> EvalResult:
        return cls(score=score, passed=passed, reasoning=reasoning, metadata=dict(metadata))

    @classmethod
    def passed_(cls, *, reasoning: str | None = None, **metadata: Any) -> EvalResult:
        return cls(score=1.0, passed=True, reasoning=reasoning, metadata=dict(metadata))

    @classmethod
    def failed(cls, *, reasoning: str | None = None, **metadata: Any) -> EvalResult:
        return cls(score=0.0, passed=False, reasoning=reasoning, metadata=dict(metadata))

    @classmethod
    def skip(cls, reason: str, **metadata: Any) -> EvalResult:
        return cls(skipped=True, skip_reason=reason, metadata=dict(metadata))


class Judge(Protocol):
    """LLM-as-judge interface; consumed by llm_judge evaluators (Phase 2.4)."""

    def __call__(
        self,
        *,
        prompt: str,
        expected_format: Literal["score_only", "score_reasoning"] = "score_reasoning",
    ) -> tuple[float, str | None]: ...


# ---------------------------------------------------------------------------
# Evaluator decorator + registry
# ---------------------------------------------------------------------------


Evaluator = Callable[..., EvalResult]


@dataclass(frozen=True)
class RegisteredEvaluator:
    """Metadata + the underlying callable."""

    name: str
    version: str
    fn: Evaluator
    description: str
    threshold: float  # score >= threshold → passed (when fn doesn't set passed)
    category: Literal["deterministic", "llm_judge", "human", "external"]
    judge_model: str | None
    module_path: str


_REGISTRY: dict[str, RegisteredEvaluator] = {}


def evaluator(
    *,
    name: str,
    version: str,
    description: str = "",
    threshold: float = 1.0,
    category: Literal["deterministic", "llm_judge", "human", "external"] = "deterministic",
    judge_model: str | None = None,
) -> Callable[[Evaluator], Evaluator]:
    """Decorator that registers an evaluator."""

    def wrap(fn: Evaluator) -> Evaluator:
        entry = RegisteredEvaluator(
            name=name,
            version=version,
            fn=fn,
            description=description,
            threshold=threshold,
            category=category,
            judge_model=judge_model,
            module_path=f"{fn.__module__}.{fn.__qualname__}",
        )
        _REGISTRY[name] = entry
        return fn

    return wrap


def get_registry() -> dict[str, RegisteredEvaluator]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()


def load_evaluators_from(path: Path) -> list[str]:
    """Import every .py under `path` so its @evaluator decorators run.

    Returns the list of evaluator names that became (newly) registered.
    """
    before = set(_REGISTRY)
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    for file in files:
        if file.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"_lab_user_eval_{file.stem}", file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sorted(set(_REGISTRY) - before)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _evaluator_db_id(entry: RegisteredEvaluator) -> int:
    """Upsert an evaluator row; return its evaluator_id."""
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO evaluators
                (name, version, category, module_path, threshold, registered_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (name, version) DO UPDATE SET
                module_path = EXCLUDED.module_path,
                threshold   = EXCLUDED.threshold,
                category    = EXCLUDED.category
            RETURNING evaluator_id
            """,
            (entry.name, entry.version, entry.category, entry.module_path, entry.threshold),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def _runs_for_experiment(experiment_slug: str) -> tuple[list[RunRow], dict[int, TaskRow]]:
    """Fetch all done runs + their tasks for an experiment."""
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.run_id, r.experiment_id, r.model_id, m.litellm_id,
                   r.task_id, r.seed, r.status,
                   r.tokens_in, r.tokens_out, r.latency_ms, r.cost_usd, r.trace_path
            FROM experiment_runs r
            JOIN models m       ON m.model_id      = r.model_id
            JOIN experiments e  ON e.experiment_id = r.experiment_id
            WHERE e.slug = %s AND r.status = 'done'
            ORDER BY r.started_at
            """,
            (experiment_slug,),
        )
        run_rows = cur.fetchall()
        task_ids = sorted({int(r[4]) for r in run_rows})
        cur.execute(
            "SELECT task_id, suite, slug, category, difficulty, payload "
            "FROM tasks WHERE task_id = ANY(%s)",
            (task_ids,),
        )
        tasks = {
            int(t[0]): TaskRow(
                task_id=int(t[0]),
                suite=t[1],
                slug=t[2],
                category=t[3],
                difficulty=t[4],
                payload=t[5],
            )
            for t in cur.fetchall()
        }

    runs: list[RunRow] = []
    for r in run_rows:
        runs.append(
            RunRow(
                run_id=r[0],
                experiment_id=int(r[1]) if r[1] is not None else None,
                model_id=int(r[2]),
                model_litellm_id=r[3],
                task_id=int(r[4]),
                seed=int(r[5]),
                status=r[6],
                tokens_in=int(r[7]) if r[7] is not None else None,
                tokens_out=int(r[8]) if r[8] is not None else None,
                latency_ms=int(r[9]) if r[9] is not None else None,
                cost_usd=float(r[10]) if r[10] is not None else None,
                trace_path=r[11],
                response_text=None,  # lazy: fetch from MinIO on demand
            )
        )
    return runs, tasks


def _fetch_response_text(trace_path: str) -> str | None:
    """Pull the response_text field out of a trace JSONL in MinIO."""
    if not trace_path or not trace_path.startswith("s3://"):
        return None
    settings = get_settings()
    from minio import Minio

    client = Minio(
        settings.s3_endpoint.removeprefix("http://").removeprefix("https://"),
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        secure=settings.s3_endpoint.startswith("https://"),
    )
    # parse s3://bucket/path
    rest = trace_path.removeprefix("s3://")
    bucket, _, key = rest.partition("/")
    resp = client.get_object(bucket, key)
    try:
        data = resp.read()
    finally:
        resp.close()
        resp.release_conn()
    line = data.decode("utf-8").splitlines()[0]
    blob = json.loads(line)
    text = blob.get("response_text")
    if not isinstance(text, str):
        return None
    return text


def _insert_eval_result(
    *, run_id: str, evaluator_id: int, result: EvalResult, threshold: float
) -> None:
    if result.skipped:
        return  # don't persist skips for now
    score = float(result.score if result.score is not None else 0.0)
    passed = result.passed if result.passed is not None else (score >= threshold)
    raw: dict[str, Any] = {
        "reasoning": result.reasoning,
        "metadata": result.metadata,
        "skipped": result.skipped,
    }
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO eval_results
                (run_id, evaluator_id, score, passed, raw, evaluated_at)
            VALUES (%(run_id)s, %(evaluator_id)s, %(score)s, %(passed)s, %(raw)s, NOW())
            ON CONFLICT (run_id, evaluator_id) DO UPDATE SET
                score    = EXCLUDED.score,
                passed   = EXCLUDED.passed,
                raw      = EXCLUDED.raw,
                evaluated_at = NOW();
            """,
            {
                "run_id": run_id,
                "evaluator_id": evaluator_id,
                "score": score,
                "passed": passed,
                "raw": Json(raw),
            },
        )


# ---------------------------------------------------------------------------
# Top-level: apply evaluators to an experiment's runs
# ---------------------------------------------------------------------------


@dataclass
class ApplyReport:
    evaluator: str
    evaluator_id: int
    n_runs: int
    n_scored: int
    n_skipped: int
    n_passed: int
    n_failed: int


def apply_to_experiment(
    experiment_slug: str,
    *,
    evaluator_names: list[str] | None = None,
    judge: Judge | None = None,
    skip_response_fetch: bool = False,
) -> list[ApplyReport]:
    """Apply selected (or all registered) evaluators to every done run of an experiment.

    Returns one ApplyReport per evaluator.
    """
    registry = get_registry()
    if evaluator_names:
        unknown = sorted(set(evaluator_names) - set(registry))
        if unknown:
            raise ValueError(f"unknown evaluator(s): {unknown}")
        entries = [registry[n] for n in evaluator_names]
    else:
        entries = list(registry.values())
    if not entries:
        raise RuntimeError("no evaluators registered; run `load_evaluators_from(...)` first")

    runs, tasks = _runs_for_experiment(experiment_slug)
    if not runs:
        raise RuntimeError(f"no done runs found for experiment {experiment_slug!r}")

    # Fetch response_text once per run, share across evaluators
    if not skip_response_fetch:
        for i, run in enumerate(runs):
            if run.trace_path and run.response_text is None:
                runs[i] = RunRow(  # immutable: replace
                    **{**run.__dict__, "response_text": _fetch_response_text(run.trace_path)}
                )

    reports: list[ApplyReport] = []
    for entry in entries:
        eid = _evaluator_db_id(entry)
        n_scored = n_skipped = n_passed = n_failed = 0
        for run in runs:
            task = tasks.get(run.task_id)
            if task is None:
                continue
            kwargs: dict[str, Any] = {}
            if entry.category == "llm_judge":
                if judge is None:
                    continue
                kwargs["judge"] = judge
            try:
                result = entry.fn(run, task, **kwargs)
            except Exception as exc:
                result = EvalResult.scored(0.0, reasoning=f"evaluator error: {exc}")
            _insert_eval_result(
                run_id=run.run_id, evaluator_id=eid, result=result, threshold=entry.threshold
            )
            if result.skipped:
                n_skipped += 1
            else:
                n_scored += 1
                if (
                    result.passed
                    if result.passed is not None
                    else (result.score or 0.0) >= entry.threshold
                ):
                    n_passed += 1
                else:
                    n_failed += 1
        reports.append(
            ApplyReport(
                evaluator=entry.name,
                evaluator_id=eid,
                n_runs=len(runs),
                n_scored=n_scored,
                n_skipped=n_skipped,
                n_passed=n_passed,
                n_failed=n_failed,
            )
        )
    return reports


def main() -> int:
    """Entry point: `uv run python -m lab.eval` to list registered evaluators."""
    from lab.eval.builtin import register_all

    register_all()
    print(f"registered {len(_REGISTRY)} evaluator(s):")
    for entry in sorted(_REGISTRY.values(), key=lambda e: e.name):
        print(
            f"  {entry.name:30s} v{entry.version}  {entry.category:14s}  "
            f"threshold={entry.threshold}  ({entry.description})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
