"""EXP-005-local-followup analyzer — 3 dropped local models on BFCL v3 AST.

Mirrors ``scripts/analyze_exp005.py`` but against the
``EXP-005-local-followup`` experiment_runs slug, with the four
hypotheses adjusted per the follow-up pre-reg
(docs/exp/EXP-005-local-followup.md):

  H1 — cloud_best (from EXP-005, baked in here as glm-5.1-cloud @ 0.9250)
       beats each new local by >= 10pp on overall accuracy.
  H2 — qwen3-30b-a3b-moe AND phi-4-reasoning-14b each land in [0.50, 0.95].
  H3 — per-category profile simple >= multiple >= parallel >= parallel_multiple
       holds for every new local.
  H4 — at least one new local beats qwen3-14b-q4 (0.910 from EXP-005)
       on overall.

Output:

  analysis/EXP-005-local-followup/SUMMARY.md           - H1..H4 + headline
  analysis/EXP-005-local-followup/per_model_overall.csv- overall + 95% CI per model
  analysis/EXP-005-local-followup/per_category.csv     - per-(model, category) means
  analysis/EXP-005-local-followup/per_cell.csv         - one row per cell

Cost: free (DB read only).
"""

from __future__ import annotations

import csv
import itertools
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

PG_DSN = "dbname=lab host=/var/run/postgresql"
EXP_SLUG = "EXP-005-local-followup"

#: Baselines pulled from EXP-005 (analysis/EXP-005/SUMMARY.md). Hard-
#: coded so the follow-up analyzer doesn't accidentally re-grade the
#: cloud arm or drift relative to the published F-011 number.
EXP005_CLOUD_BEST_NAME = "glm-5.1-cloud"
EXP005_CLOUD_BEST_MEAN = 0.9250
EXP005_DENSE_NAME = "qwen3-14b-q4"
EXP005_DENSE_MEAN = 0.9100

NEW_MODELS = ("qwen3-30b-a3b-moe", "phi-4-reasoning-14b", "hermes-4.3-36b")
H1_DELTA = 0.10
H2_MODELS = ("qwen3-30b-a3b-moe", "phi-4-reasoning-14b")
H2_LO, H2_HI = 0.50, 0.95
CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")
OUT_DIR = Path("analysis/EXP-005-local-followup")


@dataclass
class Cell:
    model: str
    task_slug: str
    category: str
    status: str
    score: float | None
    error_type: str | None
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: int | None
    error: str | None


def fetch_cells() -> list[Cell]:
    sql = """
    SELECT
      m.litellm_id AS model,
      t.slug AS task_slug,
      t.category AS category,
      r.status,
      r.error,
      r.tokens_in,
      r.tokens_out,
      r.latency_ms,
      e.score,
      (e.raw->'bfcl'->>'error_type') AS error_type
    FROM experiment_runs r
    JOIN experiments x ON x.experiment_id = r.experiment_id
    JOIN models m ON m.model_id = r.model_id
    JOIN tasks t ON t.task_id = r.task_id
    LEFT JOIN eval_results e ON e.run_id = r.run_id
    LEFT JOIN evaluators ev ON ev.evaluator_id = e.evaluator_id AND ev.name = 'bfcl_ast_match'
    WHERE x.slug = %s
    ORDER BY m.litellm_id, t.slug;
    """
    with psycopg.connect(PG_DSN, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(sql, (EXP_SLUG,))
        return [
            Cell(
                model=row["model"],
                task_slug=row["task_slug"],
                category=row["category"],
                status=row["status"],
                score=float(row["score"]) if row["score"] is not None else None,
                error_type=row["error_type"],
                tokens_in=row["tokens_in"],
                tokens_out=row["tokens_out"],
                latency_ms=row["latency_ms"],
                error=row["error"],
            )
            for row in cur.fetchall()
        ]


def _bootstrap_ci(values: list[float], n: int = 2000) -> tuple[float, float] | None:
    if not values:
        return None
    rng = random.Random(42)
    k = len(values)
    means: list[float] = []
    for _ in range(n):
        sample = [values[rng.randrange(k)] for _ in range(k)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n) - 1]
    return lo, hi


def _score_or_zero(cell: Cell) -> float:
    """Errored / unscored cell counts as 0 (same convention as analyze_exp005.py)."""

    if cell.score is None:
        return 0.0
    return cell.score


def per_model(cells: list[Cell]) -> dict[str, dict[str, float | int | tuple[float, float] | None]]:
    out: dict[str, dict[str, float | int | tuple[float, float] | None]] = {}
    by_model: dict[str, list[Cell]] = defaultdict(list)
    for c in cells:
        by_model[c.model].append(c)
    for model, lst in by_model.items():
        vals = [_score_or_zero(c) for c in lst]
        n_total = len(lst)
        n_done = sum(1 for c in lst if c.status == "done")
        n_error = sum(1 for c in lst if c.status == "error")
        mean = statistics.fmean(vals) if vals else 0.0
        out[model] = {
            "n_total": n_total,
            "n_done": n_done,
            "n_error": n_error,
            "mean": mean,
            "ci": _bootstrap_ci(vals),
        }
    return out


def per_category(cells: list[Cell]) -> dict[tuple[str, str], dict[str, float | int]]:
    out: dict[tuple[str, str], dict[str, float | int]] = {}
    by_key: dict[tuple[str, str], list[Cell]] = defaultdict(list)
    for c in cells:
        by_key[(c.model, c.category)].append(c)
    for (model, cat), lst in by_key.items():
        vals = [_score_or_zero(c) for c in lst]
        out[(model, cat)] = {
            "n": len(lst),
            "mean": statistics.fmean(vals) if vals else 0.0,
        }
    return out


def _fmt_ci(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def write_per_model_csv(stats: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["model", "n_total", "n_done", "n_error", "mean_accuracy", "ci_lower", "ci_upper"]
        )
        for model, s in sorted(stats.items()):
            ci = s["ci"]
            w.writerow(
                [
                    model,
                    s["n_total"],
                    s["n_done"],
                    s["n_error"],
                    f"{s['mean']:.4f}",
                    f"{ci[0]:.4f}" if ci else "",
                    f"{ci[1]:.4f}" if ci else "",
                ]
            )


def write_per_category_csv(stats: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "category", "n", "mean_accuracy"])
        for (model, cat), s in sorted(stats.items()):
            w.writerow([model, cat, s["n"], f"{s['mean']:.4f}"])


def write_per_cell_csv(cells: list[Cell], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "task_slug",
                "category",
                "status",
                "score",
                "error_type",
                "tokens_in",
                "tokens_out",
                "latency_ms",
                "error",
            ]
        )
        for c in cells:
            w.writerow(
                [
                    c.model,
                    c.task_slug,
                    c.category,
                    c.status,
                    c.score if c.score is not None else "",
                    c.error_type or "",
                    c.tokens_in or "",
                    c.tokens_out or "",
                    c.latency_ms or "",
                    c.error or "",
                ]
            )


def write_summary(
    cells: list[Cell],
    by_model: dict,
    by_cat: dict,
    path: Path,
) -> None:
    n_total = len(cells)
    n_error = sum(1 for c in cells if c.status == "error")
    error_rate = n_error / n_total if n_total else 0.0
    sweep_invalid = error_rate > 0.05

    # H1 — cloud_best (from EXP-005) beats each new local by >= 10pp
    h1_per_model: dict[str, tuple[bool, float]] = {}
    for m in NEW_MODELS:
        mean = by_model.get(m, {}).get("mean")
        if mean is None:
            continue
        delta = EXP005_CLOUD_BEST_MEAN - mean
        h1_per_model[m] = (delta >= H1_DELTA, delta)
    h1_pass = (not sweep_invalid) and all(ok for ok, _ in h1_per_model.values())

    # H2 — qwen3-30b-a3b-moe & phi-4-reasoning-14b each in [0.50, 0.95]
    h2_per_model: dict[str, tuple[bool, float]] = {}
    for m in H2_MODELS:
        mean = by_model.get(m, {}).get("mean")
        if mean is None:
            continue
        h2_per_model[m] = (H2_LO <= mean <= H2_HI, mean)
    h2_pass = (not sweep_invalid) and all(ok for ok, _ in h2_per_model.values())

    # H3 — per-category profile for each new model
    h3_failures: list[str] = []
    for model in NEW_MODELS:
        means_by_cat = {
            cat: by_cat[(model, cat)]["mean"] for cat in CATEGORIES if (model, cat) in by_cat
        }
        if not means_by_cat:
            continue
        ordered = [(cat, means_by_cat.get(cat)) for cat in CATEGORIES]
        clean = [(c, m) for c, m in ordered if m is not None]
        ok = True
        for (_, a), (_, b) in itertools.pairwise(clean):
            if a < b - 1e-9:
                ok = False
                break
        if not ok:
            h3_failures.append(f"{model}: {[f'{c}={v:.3f}' for c, v in clean]}")

    # H4 — at least one new local beats qwen3-14b-q4 (0.910)
    h4_per_model: dict[str, tuple[bool, float]] = {}
    for m in NEW_MODELS:
        mean = by_model.get(m, {}).get("mean")
        if mean is None:
            continue
        h4_per_model[m] = (mean >= EXP005_DENSE_MEAN, mean)
    h4_pass = (not sweep_invalid) and any(ok for ok, _ in h4_per_model.values())

    lines: list[str] = []
    lines.append("# EXP-005-local-followup — BFCL v3 local follow-up — summary\n")
    lines.append(
        f"Cells: total={n_total} done={n_total - n_error} error={n_error} ({error_rate:.1%})\n"
    )
    lines.append(
        f"EXP-005 anchors: cloud_best = {EXP005_CLOUD_BEST_NAME} ({EXP005_CLOUD_BEST_MEAN:.4f}); "
        f"dense = {EXP005_DENSE_NAME} ({EXP005_DENSE_MEAN:.4f}).\n"
    )
    if sweep_invalid:
        lines.append(
            f"\n**INVALID — sweep killed**: cell error rate {error_rate:.1%} exceeds 5% kill criterion.\n"
        )

    lines.append("\n## Per-model overall accuracy (n=1000 per model)\n")
    lines.append("| model | n | mean | 95% CI |")
    lines.append("|---|---|---|---|")
    for model in sorted(by_model):
        s = by_model[model]
        lines.append(f"| {model} | {s['n_total']} | {s['mean']:.4f} | {_fmt_ci(s['ci'])} |")

    lines.append("\n## Per-(model, category) accuracy\n")
    header = "| model | " + " | ".join(CATEGORIES) + " |"
    sep = "|---" * (len(CATEGORIES) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for model in sorted({m for (m, _) in by_cat}):
        cells_row = []
        for cat in CATEGORIES:
            v = by_cat.get((model, cat))
            cells_row.append(f"{v['mean']:.3f}" if v else "")
        lines.append(f"| {model} | " + " | ".join(cells_row) + " |")

    lines.append("\n## Hypothesis verdicts\n")
    if sweep_invalid:
        lines.append("All hypotheses: **INVALID — sweep killed**\n")
    else:
        details_h1 = ", ".join(
            f"{m}: delta={d:+.4f} {'OK' if ok else 'FAIL'}" for m, (ok, d) in h1_per_model.items()
        )
        lines.append(
            f"- **H1** (cloud_best - each_new_local >= 0.10): "
            f"{'CONFIRMED' if h1_pass else 'REFUTED'}. "
            f"{details_h1}.\n"
        )
        details_h2 = ", ".join(
            f"{m}={v:.4f} {'OK' if ok else 'FAIL'}" for m, (ok, v) in h2_per_model.items()
        )
        lines.append(
            f"- **H2** ({', '.join(H2_MODELS)} each in [{H2_LO}, {H2_HI}]): "
            f"{'CONFIRMED' if h2_pass else 'REFUTED'}. "
            f"{details_h2}.\n"
        )
        if h3_failures:
            lines.append(
                f"- **H3** (per-category profile): REFUTED.\n  - Failing models: {h3_failures}\n"
            )
        else:
            lines.append(
                "- **H3** (per-category profile simple ≥ multiple ≥ parallel ≥ parallel_multiple): CONFIRMED for all new models.\n"
            )
        details_h4 = ", ".join(
            f"{m}={v:.4f} {'OK' if ok else 'FAIL'}" for m, (ok, v) in h4_per_model.items()
        )
        lines.append(
            f"- **H4** (at least one new local beats {EXP005_DENSE_NAME} @ {EXP005_DENSE_MEAN:.4f}): "
            f"{'CONFIRMED' if h4_pass else 'REFUTED'}. "
            f"{details_h4}.\n"
        )

    # Headline
    lines.append("\n## Headline\n")
    if sweep_invalid:
        headline = f"INVALID — sweep killed at {error_rate:.0%} cell error rate."
    else:
        local_means = {m: by_model[m]["mean"] for m in NEW_MODELS if m in by_model}
        if local_means:
            best_local_name = max(local_means, key=lambda m: local_means[m])
            best_local_mean = local_means[best_local_name]
            beats_dense = best_local_mean >= EXP005_DENSE_MEAN
            headline = (
                f"Best new local = {best_local_name} ({best_local_mean:.3f}); "
                f"EXP-005 dense baseline = {EXP005_DENSE_NAME} ({EXP005_DENSE_MEAN:.3f}); "
                f"H4 {'CONFIRMED' if beats_dense else 'REFUTED'}."
            )
        else:
            headline = "no new local cells found."
    lines.append(headline + "\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(headline)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = fetch_cells()
    if not cells:
        print(f"no cells found for {EXP_SLUG}", file=sys.stderr)
        return 2
    print(f"loaded {len(cells)} cell(s)")
    by_model = per_model(cells)
    by_cat = per_category(cells)
    write_per_model_csv(by_model, OUT_DIR / "per_model_overall.csv")
    write_per_category_csv(by_cat, OUT_DIR / "per_category.csv")
    write_per_cell_csv(cells, OUT_DIR / "per_cell.csv")
    write_summary(cells, by_model, by_cat, OUT_DIR / "SUMMARY.md")
    print(f"wrote analysis to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
