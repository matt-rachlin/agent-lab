"""EXP-002b — qwen3-14b-q4 think:true vs EXP-002 think:false ablation.

Pre-registered hypothesis (docs/exp/EXP-002b-qwen3-reasoning-on-ablation.md):

    H1: think:true qwen3-14b-q4 end_state mean is materially lower than
    EXP-002's measured 0.750 baseline at think:false. Decision rule:
      >= 0.55      -> REFUTED
      0.30 - 0.55  -> MIXED
      < 0.30       -> CONFIRMED

This script reads EXP-002b cells AND the matched EXP-002 qwen3-14b-q4
cells from the lab DB, computes per-cell end_state from
agent_logs.turns->'score_breakdown', and writes:

  - analysis/EXP-002b/SUMMARY.md   — top-line H1 verdict + comparison
  - analysis/EXP-002b/verdicts.md  — full decision-rule application
  - analysis/EXP-002b/per_task_endstate.csv — per-task means side-by-side
  - analysis/EXP-002b/per_cell.csv          — per-cell scorer breakdown

Cost: free (purely a DB read).
"""

from __future__ import annotations

import csv
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

PG_DSN = "dbname=lab host=/var/run/postgresql"
EXP_NEW = "EXP-002b"  # think: true (this experiment)
EXP_REF = "EXP-002"  # think: false (baseline)
MODEL = "qwen3-14b-q4"

OUT_DIR = Path("analysis/EXP-002b")


@dataclass
class Cell:
    exp: str
    model: str
    task: str
    seed: int
    status: str
    end_state: float | None
    tool_correctness: float | None
    budget_respected: float | None
    trajectory_judge: float | None
    actual_turns: int | None
    tool_call_count: int | None
    error: str | None


def fetch_cells(exp_slug: str, only_model: str | None = None) -> list[Cell]:
    sql = """
    SELECT
      m.litellm_id AS model,
      t.slug       AS task,
      r.seed       AS seed,
      r.status     AS status,
      r.actual_turns AS actual_turns,
      r.tool_call_count AS tool_call_count,
      r.error      AS error,
      a.turns      AS turns
    FROM experiment_runs r
    JOIN models m USING (model_id)
    JOIN tasks  t ON t.task_id = r.task_id
    LEFT JOIN agent_logs a ON a.run_id = r.run_id
    WHERE r.experiment_id = (SELECT experiment_id FROM experiments WHERE slug = %s)
    """
    params: list[Any] = [exp_slug]
    if only_model is not None:
        sql += " AND m.litellm_id = %s"
        params.append(only_model)
    sql += " ORDER BY 1, 2, 3"
    out: list[Cell] = []
    with psycopg.connect(PG_DSN, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        for r in cur.fetchall():
            agent = (r.get("turns") or {}) if isinstance(r.get("turns"), dict) else {}
            scores = agent.get("score_breakdown") or {}

            def _score(name: str, _scores: dict[str, Any] = scores) -> float | None:
                v = (_scores.get(name) or {}).get("value")
                try:
                    return float(v) if v is not None else None
                except (TypeError, ValueError):
                    return None

            out.append(
                Cell(
                    exp=exp_slug,
                    model=r["model"],
                    task=r["task"],
                    seed=int(r["seed"]),
                    status=r["status"],
                    end_state=_score("end_state"),
                    tool_correctness=_score("tool_correctness"),
                    budget_respected=_score("budget_respected"),
                    trajectory_judge=_score("trajectory_judge"),
                    actual_turns=r.get("actual_turns"),
                    tool_call_count=r.get("tool_call_count"),
                    error=r.get("error"),
                )
            )
    return out


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _bootstrap_ci(
    values: list[float], n: int = 2000, alpha: float = 0.05
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the mean."""
    if not values:
        return None
    rng = random.Random(42)
    k = len(values)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(k)] for _ in range(k)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(alpha / 2 * n)]
    hi = means[int((1 - alpha / 2) * n) - 1]
    return lo, hi


def _wilcoxon_paired(diffs: list[float]) -> float | None:
    """One-sided Wilcoxon (think:false minus think:true > 0) p-value via permutation.

    Tiny permutation test: for each of 1000 sign-flip permutations of the
    paired differences, recompute the mean; the p-value is the fraction
    of permutations with mean >= the observed mean. Skips zeros, mirrors
    SciPy's default behaviour.
    """

    nonzero = [d for d in diffs if d != 0.0]
    if not nonzero:
        return None
    observed = statistics.fmean(nonzero)
    rng = random.Random(42)
    perms = 1000
    hits = 0
    for _ in range(perms):
        sample = [d if rng.random() < 0.5 else -d for d in nonzero]
        if statistics.fmean(sample) >= observed:
            hits += 1
    return hits / perms


def cell_endstate(cells: list[Cell]) -> list[float]:
    """Return the per-cell end_state value, defaulting None to 0.0.

    Pre-reg semantics: errored / scoreless cells count as failures for the
    decision-rule denominator, not silent skips. Same convention used in
    F-005's analyze_exp002.py.
    """
    return [(c.end_state if c.end_state is not None else 0.0) for c in cells]


def verdict(think_true_mean: float) -> str:
    if think_true_mean < 0.30:
        return "CONFIRMED"
    if think_true_mean < 0.55:
        return "MIXED"
    return "REFUTED"


def write_summary(
    *,
    new_cells: list[Cell],
    ref_cells: list[Cell],
    new_mean: float,
    ref_mean: float,
    new_ci: tuple[float, float] | None,
    ref_ci: tuple[float, float] | None,
    wilcoxon_p: float | None,
    err_rate: float,
    h1_verdict: str,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    delta = ref_mean - new_mean
    new_ci_str = f"[{new_ci[0]:.3f}, {new_ci[1]:.3f}]" if new_ci else "n/a"
    ref_ci_str = f"[{ref_ci[0]:.3f}, {ref_ci[1]:.3f}]" if ref_ci else "n/a"
    wp_str = f"{wilcoxon_p:.3f}" if wilcoxon_p is not None else "n/a (all zeros)"

    summary = f"""# EXP-002b summary — qwen3-14b-q4 reasoning-ON ablation

Pre-registration: `docs/exp/EXP-002b-qwen3-reasoning-on-ablation.md`
Baseline reference: EXP-002 / F-005 (`qwen3-14b-q4` at `think: false`)

## Top line

- **H1 verdict: {h1_verdict}**
- think:true `end_state` mean: **{new_mean:.3f}** (95 % CI {new_ci_str}, n={len(new_cells)})
- think:false `end_state` mean (EXP-002 baseline): **{ref_mean:.3f}** (95 % CI {ref_ci_str}, n={len(ref_cells)})
- Delta (false − true): **{delta:+.3f} pp** ({delta * 100:+.1f} percentage points)
- Wilcoxon one-sided p (think:false > think:true, paired by task): **{wp_str}**
- Cell error rate: **{err_rate * 100:.1f} %** ({sum(1 for c in new_cells if c.status != "done")} / {len(new_cells)})

## Decision rule applied (pre-registered, no peeking)

| think:true end_state mean | Verdict   |
| ------------------------- | --------- |
| >= 0.55                   | REFUTED   |
| 0.30 – 0.55 (exclusive)   | MIXED     |
| < 0.30                    | CONFIRMED |

Observed: **{new_mean:.3f}** → **{h1_verdict}**.

See `verdicts.md` for the full per-task breakdown.
"""
    (OUT_DIR / "SUMMARY.md").write_text(summary)


def write_verdicts(
    *,
    new_cells: list[Cell],
    ref_cells: list[Cell],
    new_mean: float,
    ref_mean: float,
    h1_verdict: str,
    per_task: dict[str, dict[str, float | None]],
) -> None:
    lines = [
        "# EXP-002b verdicts",
        "",
        f"H1: **{h1_verdict}**.",
        "",
        f"think:true  qwen3-14b-q4 end_state mean = **{new_mean:.3f}** (n={len(new_cells)} cells).",
        f"think:false qwen3-14b-q4 end_state mean = **{ref_mean:.3f}** (n={len(ref_cells)} cells, EXP-002 baseline).",
        "",
        "## Per-task pass@1 (end_state)",
        "",
        "| task | think:false | think:true | delta (false − true) |",
        "| ---- | ----------- | ---------- | -------------------- |",
    ]
    for task, row in sorted(per_task.items()):
        t = row.get("true")
        f = row.get("false")
        if t is None or f is None:
            lines.append(
                f"| {task} | {f if f is not None else '-'} | {t if t is not None else '-'} | n/a |"
            )
            continue
        lines.append(f"| {task} | {f:.3f} | {t:.3f} | {f - t:+.3f} |")
    (OUT_DIR / "verdicts.md").write_text("\n".join(lines) + "\n")


def write_per_task_csv(per_task: dict[str, dict[str, float | None]]) -> None:
    with (OUT_DIR / "per_task_endstate.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "think_false_mean", "think_true_mean", "delta_false_minus_true"])
        for task, row in sorted(per_task.items()):
            f = row.get("false")
            t = row.get("true")
            delta = (f - t) if (f is not None and t is not None) else ""
            w.writerow(
                [
                    task,
                    f"{f:.4f}" if f is not None else "",
                    f"{t:.4f}" if t is not None else "",
                    f"{delta:+.4f}" if isinstance(delta, float) else "",
                ]
            )


def write_per_cell_csv(cells: list[Cell]) -> None:
    with (OUT_DIR / "per_cell.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "exp",
                "model",
                "task",
                "seed",
                "status",
                "end_state",
                "tool_correctness",
                "budget_respected",
                "trajectory_judge",
                "actual_turns",
                "tool_call_count",
                "error",
            ]
        )
        for c in cells:
            w.writerow(
                [
                    c.exp,
                    c.model,
                    c.task,
                    c.seed,
                    c.status,
                    "" if c.end_state is None else f"{c.end_state:.4f}",
                    "" if c.tool_correctness is None else f"{c.tool_correctness:.4f}",
                    "" if c.budget_respected is None else f"{c.budget_respected:.4f}",
                    "" if c.trajectory_judge is None else f"{c.trajectory_judge:.4f}",
                    "" if c.actual_turns is None else c.actual_turns,
                    "" if c.tool_call_count is None else c.tool_call_count,
                    c.error or "",
                ]
            )


def main() -> int:
    new_cells = fetch_cells(EXP_NEW, only_model=MODEL)
    ref_cells = fetch_cells(EXP_REF, only_model=MODEL)

    if not new_cells:
        print(f"ERROR: no cells found for {EXP_NEW} model {MODEL}", file=sys.stderr)
        return 2
    if not ref_cells:
        print(f"WARN: no baseline cells from {EXP_REF}; H1 cannot be evaluated", file=sys.stderr)

    new_es = cell_endstate(new_cells)
    ref_es = cell_endstate(ref_cells)
    new_mean = statistics.fmean(new_es)
    ref_mean = statistics.fmean(ref_es) if ref_es else 0.0
    new_ci = _bootstrap_ci(new_es)
    ref_ci = _bootstrap_ci(ref_es) if ref_es else None

    # Per-task means
    per_task: dict[str, dict[str, float | None]] = {}
    for c in new_cells:
        per_task.setdefault(c.task, {}).setdefault("true_vals", [])  # type: ignore[arg-type]
        per_task[c.task].setdefault("true_vals", []).append(  # type: ignore[union-attr]
            c.end_state if c.end_state is not None else 0.0
        )
    for c in ref_cells:
        per_task.setdefault(c.task, {}).setdefault("false_vals", [])  # type: ignore[arg-type]
        per_task[c.task].setdefault("false_vals", []).append(  # type: ignore[union-attr]
            c.end_state if c.end_state is not None else 0.0
        )

    # Collapse to means
    collapsed: dict[str, dict[str, float | None]] = {}
    for task, vals in per_task.items():
        t_vals = vals.get("true_vals") or []
        f_vals = vals.get("false_vals") or []
        collapsed[task] = {
            "true": _mean(t_vals) if isinstance(t_vals, list) else None,
            "false": _mean(f_vals) if isinstance(f_vals, list) else None,
        }

    # Paired Wilcoxon by task on (false_mean - true_mean)
    diffs = []
    for row in collapsed.values():
        t = row.get("true")
        f = row.get("false")
        if t is None or f is None:
            continue
        diffs.append(f - t)
    wp = _wilcoxon_paired(diffs)

    err_count = sum(1 for c in new_cells if c.status != "done")
    err_rate = err_count / len(new_cells)

    h1 = verdict(new_mean)

    write_summary(
        new_cells=new_cells,
        ref_cells=ref_cells,
        new_mean=new_mean,
        ref_mean=ref_mean,
        new_ci=new_ci,
        ref_ci=ref_ci,
        wilcoxon_p=wp,
        err_rate=err_rate,
        h1_verdict=h1,
    )
    write_verdicts(
        new_cells=new_cells,
        ref_cells=ref_cells,
        new_mean=new_mean,
        ref_mean=ref_mean,
        h1_verdict=h1,
        per_task=collapsed,
    )
    write_per_task_csv(collapsed)
    write_per_cell_csv(new_cells + ref_cells)

    print(f"EXP-002b cells:     n={len(new_cells)}  end_state mean={new_mean:.3f}")
    print(f"EXP-002  baseline:  n={len(ref_cells)}  end_state mean={ref_mean:.3f}")
    if math.isfinite(new_mean) and math.isfinite(ref_mean):
        print(f"delta (false − true): {ref_mean - new_mean:+.3f}")
    if wp is not None:
        print(f"Wilcoxon one-sided p: {wp:.3f}")
    print(f"cell error rate: {err_rate * 100:.1f} % ({err_count}/{len(new_cells)})")
    print(f"H1 verdict: {h1}")
    print(f"artifacts written to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
