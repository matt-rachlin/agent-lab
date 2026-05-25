"""Eval package — evaluator framework + judge + analysis primitives."""

from lab.eval.framework import (
    EvalResult,
    Evaluator,
    RegisteredEvaluator,
    apply_to_experiment,
    clear_registry,
    evaluator,
    get_registry,
    load_evaluators_from,
)

__all__ = [
    "EvalResult",
    "Evaluator",
    "RegisteredEvaluator",
    "apply_to_experiment",
    "clear_registry",
    "evaluator",
    "get_registry",
    "load_evaluators_from",
]
