"""Apply cheap LLM-judge (gpt-oss-20b-cloud) to all done EXP-001 runs.

The oracle slice + calibration step requires a function we don't have in the
framework yet (apply_to_runs subset). For now this script only handles the
cheap-judge pass via the existing framework API; calibration becomes a Phase 6
follow-up when we add `apply_to_runs`.

Usage:
    uv run python scripts/judge_exp001.py
"""

from __future__ import annotations

import time

from rich.console import Console

from lab.eval.builtin import register_all
from lab.eval.framework import apply_to_experiment
from lab.eval.judge import make_judge

console = Console()
SLUG = "EXP-001"


def main() -> None:
    t0 = time.time()
    register_all()
    judge_callable = make_judge(model="gpt-oss-20b-cloud", position_swap=False, timeout=120)
    console.log("[bold]applying cheap judge (gpt-oss-20b-cloud) to EXP-001 done runs[/bold]")
    reports = apply_to_experiment(SLUG, evaluator_names=["llm_judge_quality"], judge=judge_callable)
    for r in reports:
        console.log(f"  {r}")
    console.log(f"[bold green]judge slice done in {time.time() - t0:.0f}s[/bold green]")


if __name__ == "__main__":
    main()
