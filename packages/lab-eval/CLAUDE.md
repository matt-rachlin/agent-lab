# lab-eval

Eval surface:
- `lab.eval.framework` — `@evaluator` decorator, `EvalResult`, registry
- `lab.eval.judge` — LLM-judge helper (calls `lab.core.llm`)
- `lab.eval.builtin.*` — out-of-box scorers (`not_empty`, `latency_under`, ...)
- `lab.tasks.registry` — task ID → metadata index

## Gotchas
- The evaluator registry is module-global; tests must `clear_registry()` in teardown.
- Judge calls go through `lab.core.llm` so they share retry/cost logic.
