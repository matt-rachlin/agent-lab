"""Frozen golden outputs — store + comparator for regression detection.

A "golden" is a canonical capture of a model's response to a task, written
to ``evals/golden/<suite>/<task_slug>/<model>.json``. The capture includes:

* the response text the model emitted
* the flattened tool-call list (tool name + args, one entry per call)
* per-scorer outcomes (e.g. ``end_state``, ``tool_correctness``)
* a config-hash + capture timestamp so we can tell when goldens drift

Regression tests use :func:`compare_to_golden` to detect harness changes that
silently alter trajectories — without needing to re-run the model. If a
prompt edit, scorer tweak, or solver refactor changes any of the captured
fields, the diff is surfaced loudly.

This module is intentionally lightweight: it does not run any model. The
:mod:`tools.sync_golden_outputs` script handles the (expensive) capture.

See also ``evals/golden/README.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_GOLDEN_ROOT",
    "GoldenComparison",
    "GoldenOutput",
    "compare_to_golden",
    "golden_path",
    "load_golden",
    "save_golden",
]


DEFAULT_GOLDEN_ROOT = Path("evals/golden")


@dataclass(frozen=True)
class GoldenOutput:
    """A frozen capture for one (suite, task, model) triple."""

    task_slug: str
    model: str
    suite: str
    config_hash: str
    captured_at: str
    response_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    scorer_outcomes: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_slug": self.task_slug,
            "model": self.model,
            "suite": self.suite,
            "config_hash": self.config_hash,
            "captured_at": self.captured_at,
            "response_text": self.response_text,
            "tool_calls": list(self.tool_calls),
            "scorer_outcomes": dict(self.scorer_outcomes),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> GoldenOutput:
        return cls(
            task_slug=str(raw["task_slug"]),
            model=str(raw["model"]),
            suite=str(raw.get("suite") or ""),
            config_hash=str(raw.get("config_hash") or ""),
            captured_at=str(raw.get("captured_at") or ""),
            response_text=str(raw.get("response_text") or ""),
            tool_calls=list(raw.get("tool_calls") or []),
            scorer_outcomes={
                str(k): float(v) for k, v in (raw.get("scorer_outcomes") or {}).items()
            },
        )


@dataclass(frozen=True)
class GoldenComparison:
    """Result of comparing an actual trajectory to a stored golden."""

    found: bool
    same_response: bool
    same_tool_calls: bool
    scorer_drift: dict[str, float]
    response_diff: str | None = None
    tool_call_diff: str | None = None

    @property
    def is_match(self) -> bool:
        """All three checks pass and no scorer drift exceeds tolerance."""
        return self.found and self.same_response and self.same_tool_calls and not self.scorer_drift

    @classmethod
    def not_found(cls) -> GoldenComparison:
        """Sentinel for missing golden — neither pass nor fail; caller decides."""
        return cls(
            found=False,
            same_response=False,
            same_tool_calls=False,
            scorer_drift={},
            response_diff=None,
            tool_call_diff=None,
        )


def golden_path(
    task_slug: str,
    model: str,
    *,
    suite: str,
    root: Path | None = None,
) -> Path:
    """Where the golden for (suite, task_slug, model) lives."""
    base = root if root is not None else DEFAULT_GOLDEN_ROOT
    return base / suite / task_slug / f"{model}.json"


def load_golden(
    task_slug: str,
    model: str,
    *,
    suite: str,
    root: Path | None = None,
) -> GoldenOutput | None:
    """Load a golden from disk; return ``None`` if missing."""
    path = golden_path(task_slug, model, suite=suite, root=root)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GoldenOutput.from_dict(raw)


def save_golden(golden: GoldenOutput, *, root: Path | None = None) -> Path:
    """Serialise a golden to disk; creates parent dirs as needed."""
    path = golden_path(golden.task_slug, golden.model, suite=golden.suite, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(golden.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _normalise_tool_calls(
    calls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reduce a tool-call list to (tool, args) tuples for comparison."""
    out: list[dict[str, Any]] = []
    for tc in calls:
        out.append(
            {
                "tool": tc.get("tool") or tc.get("name") or "",
                "args": tc.get("args") or tc.get("arguments") or {},
            }
        )
    return out


def _diff_response(expected: str, actual: str) -> str:
    """Compact diff suitable for an assertion message."""
    if expected == actual:
        return ""
    # Cap each side at 200 chars so the error stays readable.
    exp_snip = expected if len(expected) <= 200 else expected[:200] + "...[trunc]"
    act_snip = actual if len(actual) <= 200 else actual[:200] + "...[trunc]"
    return f"expected: {exp_snip!r}\nactual:   {act_snip!r}"


def _diff_tool_calls(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
) -> str:
    """Render a tool-call delta as a short multi-line summary."""
    if list(expected) == list(actual):
        return ""
    return (
        f"expected {len(expected)} calls: "
        f"{[c.get('tool') for c in expected]}\n"
        f"actual   {len(actual)} calls: "
        f"{[c.get('tool') for c in actual]}"
    )


def compare_to_golden(
    task_slug: str,
    model: str,
    actual: Mapping[str, Any],
    *,
    suite: str,
    tolerance: float = 0.0,
    root: Path | None = None,
) -> GoldenComparison:
    """Compare an actual trajectory dict to the frozen golden.

    ``actual`` is a dict (typically derived from a ``RunRow`` /
    ``TrajectoryResult``) with optional keys:

    * ``response_text`` — the final assistant message
    * ``tool_calls`` — flattened list of ``{"tool": str, "args": dict}``
    * ``scorer_outcomes`` — ``{scorer_name: float}``

    Returns a :class:`GoldenComparison`. If the golden file is missing, all
    boolean fields are ``False`` and ``found=False`` so callers can decide
    whether to fail-fast or just warn.

    ``tolerance`` is the absolute difference allowed per scorer before drift
    is recorded. ``0.0`` (the default) requires exact equality.
    """
    golden = load_golden(task_slug, model, suite=suite, root=root)
    if golden is None:
        return GoldenComparison.not_found()

    actual_response = str(actual.get("response_text") or "")
    actual_calls = _normalise_tool_calls(actual.get("tool_calls") or [])
    actual_scorers = {str(k): float(v) for k, v in (actual.get("scorer_outcomes") or {}).items()}

    expected_calls = _normalise_tool_calls(golden.tool_calls)
    same_response = actual_response == golden.response_text
    same_calls = actual_calls == expected_calls

    drift: dict[str, float] = {}
    # Scorer drift is computed over the union: missing-on-either-side
    # counts as max drift (1.0) so silent removal is visible.
    keys = set(golden.scorer_outcomes) | set(actual_scorers)
    for k in sorted(keys):
        exp = golden.scorer_outcomes.get(k)
        act = actual_scorers.get(k)
        if exp is None or act is None:
            drift[k] = 1.0
            continue
        delta = abs(exp - act)
        if delta > tolerance:
            drift[k] = round(delta, 6)

    return GoldenComparison(
        found=True,
        same_response=same_response,
        same_tool_calls=same_calls,
        scorer_drift=drift,
        response_diff=_diff_response(golden.response_text, actual_response) or None,
        tool_call_diff=_diff_tool_calls(expected_calls, actual_calls) or None,
    )
