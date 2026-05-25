"""llm_judge_quality: ask a judge whether the response addresses the task input."""

from __future__ import annotations

from lab.eval.framework import EvalResult, Judge, RunRow, TaskRow, evaluator


@evaluator(
    name="llm_judge_quality",
    version="1.0",
    description="LLM judge scores 0-1 whether response addresses the question well",
    threshold=0.7,
    category="llm_judge",
    judge_model="gpt-oss-20b-cloud",
)
def llm_judge_quality(run: RunRow, task: TaskRow, *, judge: Judge | None = None) -> EvalResult:
    if judge is None:
        return EvalResult.skip("no judge configured")
    if not run.response_text:
        return EvalResult.failed(reasoning="empty response")
    user_input = task.payload.get("input") or ""
    gold = task.payload.get("gold_answer")
    rubric_block = f"\nReference answer (for guidance, not strict match): {gold}\n" if gold else ""
    prompt = (
        f"Task:\n{user_input}\n\n"
        f"Response:\n{run.response_text}\n"
        f"{rubric_block}\n"
        "Score 0.0 to 1.0 — 1.0 means the response fully addresses the task; "
        "0.0 means it does not address the task at all. Reply with JSON only."
    )
    score, reasoning = judge(prompt=prompt)
    return EvalResult.scored(score, reasoning=reasoning)
