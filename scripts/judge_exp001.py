"""Apply LLM-judge (gpt-oss-20b-cloud) to all done EXP-001 runs, then re-judge a
10% slice with gpt-oss-120b-cloud as oracle for calibration.

Per protocols/judge-calibration.md:
- Standard judge: gpt-oss-20b-cloud
- Oracle judge: gpt-oss-120b-cloud
- 10% slice sampled with seed=42 for reproducibility
- Report Cohen's kappa + Pearson r between cheap-judge and oracle
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import duckdb
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr

from lab.eval.framework import apply_to_experiment, ensure_db_evaluator
from lab.eval.judge import make_judge

console = Console()
SLUG = "EXP-001"
PG = "dbname=lab host=/var/run/postgresql"
ORACLE_SEED = 42
ORACLE_FRACTION = 0.10


def run_cheap_judge() -> None:
    """Apply gpt-oss-20b-cloud judge to all done runs."""
    judge = make_judge(
        evaluator_name="llm_judge_quality_cheap",
        judge_model="gpt-oss-20b-cloud",
        position_swap=False,
        timeout_sec=120,
    )
    ensure_db_evaluator(
        evaluator_name="llm_judge_quality_cheap",
        eval_type="llm_judge",
        judge_model="gpt-oss-20b-cloud",
    )
    console.log("[bold]applying cheap judge (gpt-oss-20b-cloud) to EXP-001 done runs[/bold]")
    apply_to_experiment(SLUG, judge)


def select_oracle_slice(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Random 10% sample of done run_ids, deterministic w/ seed=42."""
    rows = con.execute(
        f"""
        SELECT r.run_id
        FROM postgres_scan('{PG}', 'public', 'experiment_runs') r
        WHERE r.experiment_id = (
            SELECT experiment_id FROM postgres_scan('{PG}', 'public', 'experiments')
            WHERE slug = '{SLUG}'
        )
          AND r.status = 'done'
        ORDER BY r.run_id
        """
    ).fetchall()
    all_ids = [r[0] for r in rows]
    rng = random.Random(ORACLE_SEED)
    sample_n = int(len(all_ids) * ORACLE_FRACTION)
    return sorted(rng.sample(all_ids, sample_n))


def run_oracle_judge(slice_ids: list[str]) -> None:
    """Apply gpt-oss-120b-cloud judge to oracle slice only."""
    from lab.eval.framework import apply_to_runs

    judge = make_judge(
        evaluator_name="llm_judge_quality_oracle",
        judge_model="gpt-oss-120b-cloud",
        position_swap=False,
        timeout_sec=300,
    )
    ensure_db_evaluator(
        evaluator_name="llm_judge_quality_oracle",
        eval_type="llm_judge",
        judge_model="gpt-oss-120b-cloud",
    )
    console.log(f"[bold]applying oracle judge to {len(slice_ids)} run slice[/bold]")
    apply_to_runs(slice_ids, judge)


def calibration_report(con: duckdb.DuckDBPyConnection, slice_ids: list[str]) -> str:
    """Compute kappa + pearson r between cheap and oracle on slice."""
    ids_csv = ",".join(f"'{i}'" for i in slice_ids)
    sql = f"""
    SELECT
        e.run_id,
        MAX(CASE WHEN e.evaluator_name='llm_judge_quality_cheap'  THEN e.score END) AS cheap,
        MAX(CASE WHEN e.evaluator_name='llm_judge_quality_oracle' THEN e.score END) AS oracle
    FROM postgres_scan('{PG}', 'public', 'eval_results') e
    WHERE e.run_id IN ({ids_csv})
    GROUP BY 1
    HAVING cheap IS NOT NULL AND oracle IS NOT NULL
    """
    rows = con.execute(sql).fetchall()
    if not rows:
        return "## Calibration\n\nno overlapping rows — calibration could not be run\n"
    cheap = [float(r[1]) for r in rows]
    oracle = [float(r[2]) for r in rows]
    # Bin to {0.0, 0.5, 1.0} for kappa
    def _bin(x: float) -> int:
        if x >= 0.75:
            return 2
        if x >= 0.25:
            return 1
        return 0
    cheap_bin = [_bin(c) for c in cheap]
    oracle_bin = [_bin(o) for o in oracle]
    kappa = cohen_kappa_score(oracle_bin, cheap_bin)
    pr, _ = pearsonr(oracle, cheap)
    out = ["## Judge calibration (cheap vs oracle on 10% slice)\n"]
    out.append(f"- n: **{len(rows)}**")
    out.append(f"- Pearson r: **{pr:.3f}**")
    out.append(f"- Cohen's kappa (3-bin): **{kappa:.3f}**")
    if pr < 0.6 or kappa < 0.4:
        out.append("\n**KILL CRITERION FIRED**: judge agreement below threshold (r<0.6 OR kappa<0.4).")
        out.append("Cheap judge results should be treated as unreliable. Either re-run with oracle on all,")
        out.append("or restrict claims to deterministic-evaluator-only verdicts.\n")
    else:
        out.append("\nJudge agreement passes calibration threshold. Cheap-judge results can be trusted.")
    return "\n".join(out) + "\n"


def main() -> None:
    t0 = time.time()
    run_cheap_judge()
    con = duckdb.connect(":memory:")
    con.execute("INSTALL postgres_scanner; LOAD postgres_scanner;")
    slice_ids = select_oracle_slice(con)
    run_oracle_judge(slice_ids)
    report = calibration_report(con, slice_ids)
    print(report)
    out_path = Path("/data/lab/code/docs/findings/F-003-judge-calibration.tmp.md")
    out_path.write_text(report)
    console.log(f"[bold green]judge slice + calibration done in {time.time() - t0:.0f}s[/bold green]")
    console.log(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
