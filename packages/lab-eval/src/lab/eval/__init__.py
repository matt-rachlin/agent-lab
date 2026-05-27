"""Eval package — evaluator framework + judge + analysis primitives."""

from lab.eval.framework import (
    EvalResult,
    Evaluator,
    RegisteredEvaluator,
    apply_to_experiment,
    apply_to_runs,
    clear_registry,
    evaluator,
    get_registry,
    load_evaluators_from,
)
from lab.eval.golden import (
    DEFAULT_GOLDEN_ROOT,
    GoldenComparison,
    GoldenOutput,
    compare_to_golden,
    golden_path,
    load_golden,
    save_golden,
)

__all__ = [
    "DEFAULT_GOLDEN_ROOT",
    "EvalResult",
    "Evaluator",
    "GoldenComparison",
    "GoldenOutput",
    "RegisteredEvaluator",
    "apply_to_experiment",
    "apply_to_runs",
    "clear_registry",
    "compare_to_golden",
    "evaluator",
    "get_registry",
    "golden_path",
    "load_evaluators_from",
    "load_golden",
    "save_golden",
]
