"""EXP-009 — HARD-BENCH-003: N=8 seed confirmation of the hard-suite ranking.

Five pre-registered hypotheses (docs/exp/EXP-009-hard-bench-multiseed.md):

  H1 (ranking):    gemma4 > qwen3-coder > devstral in pass@1 AND the
                   gemma4-vs-qwen3 bootstrap 95% CIs do not overlap.
                   Ordering holds but CIs overlap -> INCONCLUSIVE.
  H2 (anchoring):  every model's pass@1 within +/-0.05 of its
                   HARD-BENCH-002 single-seed number.
  H3 (variance):   per-model max-min seed pass-rate spread in [0.02, 0.06];
                   spread == 0 for any model -> REFUTED.
  H4 (reliability): pass^8 < pass@1 for all models AND gemma4 pass^8 >= 0.75.
  H5 (structure):  the four named qwen3-coder code tasks score 0/8.

Reads cells from the lab DB (experiment_runs + agent_logs score_breakdown),
writes analysis/EXP-009/{SUMMARY.md,per_model.csv,per_category.csv,
per_task_seed_matrix.csv}.

Smoke-testable on any agent experiment: --slug CODER-BENCH-001 (3 seeds)
exercises the same code paths with degenerate pass^8.

Cost: free (purely a DB read).
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from math import comb
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

PG_DSN = "dbname=lab host=/var/run/postgresql"
DEFAULT_SLUG = "HARD-BENCH-003"
EXP_DIR_NAME = "EXP-009"

MODEL_A = "gemma4-12b"
MODEL_B = "qwen3-coder-30b"
MODEL_C = "devstral-24b"

# Pre-registered thresholds (do NOT change after sweep starts).
H2_ANCHORS = {MODEL_A: 0.938, MODEL_B: 0.812, MODEL_C: 0.531}
H2_BAND = 0.05
H3_LO, H3_HI = 0.02, 0.06
H4_GEMMA_PASS8 = 0.75
H5_TASKS = (
    "code-fibonacci-bug-fix",
    "code-interval-merge-fix",
    "code-topo-sort",
    "code-expr-parser-fix",
)
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 9


@dataclass
class Cell:
    model: str
    task: str
    category: str
    seed: int
    score: float


def load_cells(slug: str) -> tuple[list[Cell], list[dict[str, object]]]:
    sql = """
        select m.litellm_id as model, t.slug as task,
               coalesce(t.category, '?') as category, er.seed,
               er.status, er.error,
               (al.turns->'score_breakdown'->'end_state'->>'value')::float
                   as score
        from experiment_runs er
        join experiments e on e.experiment_id = er.experiment_id
        join models m on m.model_id = er.model_id
        join tasks t on t.task_id = er.task_id
        left join agent_logs al on al.run_id = er.run_id
        where e.slug = %s
        order by m.litellm_id, t.slug, er.seed
    """
    cells: list[Cell] = []
    excluded: list[dict[str, object]] = []
    with psycopg.connect(PG_DSN, row_factory=dict_row) as conn:
        for r in conn.execute(sql, (slug,)):
            if r["score"] is None:
                excluded.append(
                    {
                        "model": r["model"],
                        "task": r["task"],
                        "seed": r["seed"],
                        "status": r["status"],
                        "error": (r["error"] or "")[:120],
                    }
                )
                continue
            cells.append(
                Cell(
                    model=str(r["model"]),
                    task=str(r["task"]),
                    category=str(r["category"]),
                    seed=int(r["seed"]),
                    score=float(r["score"]),
                )
            )
    return cells, excluded


def by_model_task(cells: list[Cell]) -> dict[str, dict[str, list[Cell]]]:
    out: dict[str, dict[str, list[Cell]]] = defaultdict(lambda: defaultdict(list))
    for c in cells:
        out[c.model][c.task].append(c)
    return out


def pass_caret_k(passes: int, n: int, k: int) -> float:
    """Empirical P(all of k draws without replacement pass). NaN if n < k."""
    if n < k:
        return float("nan")
    return comb(passes, k) / comb(n, k) if passes >= k else 0.0


def task_pass_counts(mt: dict[str, list[Cell]]) -> dict[str, tuple[int, int]]:
    return {task: (sum(1 for c in cl if c.score >= 1.0), len(cl)) for task, cl in mt.items()}


def model_pass1(mt: dict[str, list[Cell]]) -> float:
    counts = task_pass_counts(mt)
    return sum(p / n for p, n in counts.values()) / len(counts)


def model_pass_k(mt: dict[str, list[Cell]], k: int) -> float:
    counts = task_pass_counts(mt)
    vals = [pass_caret_k(p, n, k) for p, n in counts.values()]
    if any(v != v for v in vals):  # any NaN -> undefined at this k
        return float("nan")
    return sum(vals) / len(vals)


def fmt(v: float) -> str:
    return "-" if v != v else f"{v:.3f}"


def bootstrap_ci(mt: dict[str, list[Cell]], rng: random.Random) -> tuple[float, float]:
    """Cluster bootstrap over tasks (seeds within a task are correlated)."""
    rates = [p / n for p, n in task_pass_counts(mt).values()]
    boots = sorted(sum(rng.choices(rates, k=len(rates))) / len(rates) for _ in range(BOOTSTRAP_N))
    return boots[int(0.025 * BOOTSTRAP_N)], boots[int(0.975 * BOOTSTRAP_N)]


def seed_spread(model_cells: list[Cell]) -> tuple[dict[int, float], float]:
    per_seed: dict[int, list[float]] = defaultdict(list)
    for c in model_cells:
        per_seed[c.seed].append(c.score)
    rates = {s: sum(v) / len(v) for s, v in sorted(per_seed.items())}
    return rates, (max(rates.values()) - min(rates.values()) if rates else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument("--out", default=f"analysis/{EXP_DIR_NAME}")
    args = ap.parse_args()

    cells, excluded = load_cells(args.slug)
    if not cells:
        raise SystemExit(f"no scored cells for {args.slug}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(BOOTSTRAP_SEED)
    grouped = by_model_task(cells)
    models = sorted(grouped)

    stats: dict[str, dict[str, float]] = {}
    cis: dict[str, tuple[float, float]] = {}
    spreads: dict[str, tuple[dict[int, float], float]] = {}
    for m in models:
        mt = grouped[m]
        stats[m] = {
            "pass@1": model_pass1(mt),
            "pass^4": model_pass_k(mt, 4),
            "pass^8": model_pass_k(mt, 8),
            "n_tasks": float(len(mt)),
            "n_cells": float(sum(len(v) for v in mt.values())),
        }
        cis[m] = bootstrap_ci(mt, rng)
        spreads[m] = seed_spread([c for tc in mt.values() for c in tc])

    # --- verdicts -----------------------------------------------------------
    verdicts: dict[str, str] = {}
    have_abc = all(m in stats for m in (MODEL_A, MODEL_B, MODEL_C))
    if have_abc:
        order_ok = stats[MODEL_A]["pass@1"] > stats[MODEL_B]["pass@1"] > stats[MODEL_C]["pass@1"]
        ci_sep = cis[MODEL_A][0] > cis[MODEL_B][1]
        verdicts["H1"] = (
            "CONFIRMED"
            if order_ok and ci_sep
            else "INCONCLUSIVE (ordering holds, CIs overlap)"
            if order_ok
            else "REFUTED"
        )
        h2_fail = [
            m for m, anchor in H2_ANCHORS.items() if abs(stats[m]["pass@1"] - anchor) > H2_BAND
        ]
        verdicts["H2"] = "CONFIRMED" if not h2_fail else f"REFUTED for {', '.join(h2_fail)}"
        zero = [m for m in models if spreads[m][1] == 0.0]
        in_band = all(H3_LO <= spreads[m][1] <= H3_HI for m in models)
        verdicts["H3"] = (
            f"REFUTED (zero spread: {', '.join(zero)})"
            if zero
            else "CONFIRMED"
            if in_band
            else "REFUTED (spread outside [0.02, 0.06])"
        )
        p8 = {m: stats[m]["pass^8"] for m in models}
        if any(v != v for v in p8.values()):
            verdicts["H4"] = "UNDEFINED (fewer than 8 seeds)"
        else:
            h4_strict = all(p8[m] < stats[m]["pass@1"] for m in models)
            verdicts["H4"] = (
                "CONFIRMED" if h4_strict and p8[MODEL_A] >= H4_GEMMA_PASS8 else "REFUTED"
            )
        qb = task_pass_counts(grouped[MODEL_B])
        h5_rows = {t: qb.get(t, (0, 0)) for t in H5_TASKS}
        verdicts["H5"] = (
            "CONFIRMED" if all(p == 0 and n > 0 for p, n in h5_rows.values()) else "REFUTED"
        )
    else:
        verdicts["note"] = "expected models absent; smoke-test mode, no verdicts"

    # --- outputs ------------------------------------------------------------
    with (out_dir / "per_model.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "pass@1",
                "ci95_lo",
                "ci95_hi",
                "pass^4",
                "pass^8",
                "seed_spread",
                "n_tasks",
                "n_cells",
            ]
        )
        for m in models:
            w.writerow(
                [
                    m,
                    f"{stats[m]['pass@1']:.4f}",
                    f"{cis[m][0]:.4f}",
                    f"{cis[m][1]:.4f}",
                    fmt(stats[m]["pass^4"]),
                    fmt(stats[m]["pass^8"]),
                    f"{spreads[m][1]:.4f}",
                    int(stats[m]["n_tasks"]),
                    int(stats[m]["n_cells"]),
                ]
            )

    cats = sorted({c.category for c in cells})
    with (out_dir / "per_category.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", *cats])
        for m in models:
            row: list[str] = [m]
            for cat in cats:
                cc = [c for tc in grouped[m].values() for c in tc if c.category == cat]
                row.append(f"{sum(c.score for c in cc) / len(cc):.3f}" if cc else "-")
            w.writerow(row)

    seeds = sorted({c.seed for c in cells})
    with (out_dir / "per_task_seed_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "category"] + [f"s{s}" for s in seeds] + ["passes", "n"])
        for m in models:
            for task in sorted(grouped[m]):
                tc = {c.seed: c for c in grouped[m][task]}
                marks = [
                    "1" if s in tc and tc[s].score >= 1.0 else "0" if s in tc else "-"
                    for s in seeds
                ]
                p = sum(1 for c in tc.values() if c.score >= 1.0)
                cat = next(iter(tc.values())).category
                w.writerow([m, task, cat, *marks, p, len(tc)])

    lines = [f"# {EXP_DIR_NAME} / {args.slug} — summary", ""]
    lines.append("| model | pass@1 | 95% CI | pass^4 | pass^8 | seed spread |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for m in models:
        lines.append(
            f"| {m} | {stats[m]['pass@1']:.3f} "
            f"| [{cis[m][0]:.3f}, {cis[m][1]:.3f}] "
            f"| {fmt(stats[m]['pass^4'])} | {fmt(stats[m]['pass^8'])} "
            f"| {spreads[m][1]:.3f} |"
        )
    lines.append("")
    lines.append("## Verdicts")
    lines.append("")
    for h, v in verdicts.items():
        lines.append(f"- **{h}**: {v}")
    if excluded:
        lines.append("")
        lines.append(f"## Excluded cells ({len(excluded)})")
        lines.append("")
        for e in excluded[:20]:
            lines.append(f"- {e['model']} / {e['task']} / s{e['seed']}: {e['status']} {e['error']}")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
