"""Built-in evaluators — opt in by calling `register_all()`."""

from __future__ import annotations


def register_all() -> None:
    """Import all built-in evaluator modules so their @evaluator decorators run."""
    from lab.eval.builtin import (  # noqa: F401
        bfcl_ast_match,
        constraint_violations,
        cost_under,
        exact_match,
        json_valid,
        latency_under,
        llm_judge_quality,
        not_empty,
        regex_match,
        tokens_under,
        useful_latency,
    )


__all__ = ["register_all"]
