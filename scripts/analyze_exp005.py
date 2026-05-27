"""EXP-005 analyzer — BFCL v3 external benchmark, local vs cloud.

Reads BFCL AST grades out of ``eval_results`` (the runner pre-writes
``bfcl_ast_match`` rows at cell execution time) and computes:

  H1 — cloud_best beats qwen3-14b-q4 by >= 10pp on overall accuracy
  H2 — qwen3-14b-q4 overall in [0.35, 0.65]
  H3 — model ordering gpt-oss-120b >= glm-5.1 >= gpt-oss-20b >= qwen3-14b-q4
  H4 — per-category profile simple >= multiple >= parallel >= parallel_multiple

Output:

  analysis/EXP-005/SUMMARY.md           - top-line H1..H4 + 1-line headline
  analysis/EXP-005/per_model_overall.csv- accuracy + 95% bootstrap CI per model
  analysis/EXP-005/per_category.csv     - per-(model, category) means
  analysis/EXP-005/per_cell.csv         - one row per cell

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
EXP_SLUG = "EXP-005"
MODEL_DENSE = "qwen3-14b-q4"
MODELS_CLOUD = ("gpt-oss-20b-cloud", "glm-5.1-cloud", "gpt-oss-120b-cloud")
H1_DELTA = 0.10
H2_LO, H2_HI = 0.35, 0.65
CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")
OUT_DIR = Path("analysis/EXP-005")


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
    """An errored or unscored cell counts as 0.0 — matches our
    PBS-Agent analyzer convention. Document any error-rate kill in
    the SUMMARY before computing point estimates."""

    if cell.score is None:
        return 0.0
    return cell.score


def per_model(cells: list[Cell]) -> dict[str, dict[str, float | int | tuple[float, float] | None]]:
    """Compute overall accuracy + CI per model. Errored cells count 0."""

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


def _safe_filename(name: str) -> str:
    """Replace forward slashes / nul / control chars so analysis files
    can be written safely."""

    return name.replace("/", "_").replace("\\", "_")


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

    # H1 — cloud_best beats local by >= 10pp
    dense_mean = (
        by_model.get(MODEL_DENSE, {}).get("mean", 0.0) if by_model.get(MODEL_DENSE) else 0.0
    )
    cloud_means = {m: by_model[m]["mean"] for m in MODELS_CLOUD if m in by_model}
    cloud_best_name = max(cloud_means, key=lambda m: cloud_means[m]) if cloud_means else None
    cloud_best_mean = cloud_means[cloud_best_name] if cloud_best_name else 0.0
    h1_delta = cloud_best_mean - dense_mean
    h1_pass = (not sweep_invalid) and (h1_delta >= H1_DELTA)

    # H2 — qwen3-14b-q4 in [0.35, 0.65]
    h2_pass = (not sweep_invalid) and (H2_LO <= dense_mean <= H2_HI)

    # H3 — model ordering
    expected_order = ["gpt-oss-120b-cloud", "glm-5.1-cloud", "gpt-oss-20b-cloud", "qwen3-14b-q4"]
    measured_order = [m for m in expected_order if m in by_model]
    h3_pass = not sweep_invalid
    for a, b in itertools.pairwise(measured_order):
        if by_model[a]["mean"] < by_model[b]["mean"]:
            h3_pass = False
            break

    # H4 — per-category profile
    h4_failures: list[str] = []
    for model in sorted({m for (m, _) in by_cat}):
        means_by_cat = {
            cat: by_cat[(model, cat)]["mean"] for cat in CATEGORIES if (model, cat) in by_cat
        }
        ordered = [means_by_cat.get(cat) for cat in CATEGORIES]
        # ignore None
        clean = [(c, m) for c, m in zip(CATEGORIES, ordered, strict=True) if m is not None]
        ok = True
        for (_, a), (_, b) in itertools.pairwise(clean):
            if a < b - 1e-9:
                ok = False
                break
        if not ok:
            h4_failures.append(f"{model}: {[f'{c}={v:.3f}' for c, v in clean]}")

    lines: list[str] = []
    lines.append("# EXP-005 — BFCL v3 external benchmark — summary\n")
    lines.append(
        f"Cells: total={n_total} done={n_total - n_error} error={n_error} ({error_rate:.1%})\n"
    )
    if sweep_invalid:
        lines.append(
            f"\n**INVALID — sweep killed**: cell error rate {error_rate:.1%} exceeds 5% kill criterion.\n"
        )

    # Per-model table
    lines.append("\n## Per-model overall accuracy (n=1000 per model)\n")
    lines.append("| model | n | mean | 95% CI |")
    lines.append("|---|---|---|---|")
    for model in sorted(by_model):
        s = by_model[model]
        lines.append(f"| {model} | {s['n_total']} | {s['mean']:.4f} | {_fmt_ci(s['ci'])} |")

    # Per-category table
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

    # Hypothesis verdicts
    lines.append("\n## Hypothesis verdicts\n")
    if sweep_invalid:
        lines.append("All hypotheses: **INVALID — sweep killed**\n")
    else:
        lines.append(
            f"- **H1** (cloud_best - dense >= 0.10): "
            f"{'CONFIRMED' if h1_pass else 'REFUTED'}. "
            f"Best cloud = {cloud_best_name} ({cloud_best_mean:.4f}); "
            f"dense = {MODEL_DENSE} ({dense_mean:.4f}); delta = {h1_delta:.4f}.\n"
        )
        lines.append(
            f"- **H2** ({MODEL_DENSE} in [{H2_LO}, {H2_HI}]): "
            f"{'CONFIRMED' if h2_pass else 'REFUTED'}. "
            f"Measured {dense_mean:.4f}.\n"
        )
        ordering_str = " >= ".join(f"{m}({by_model[m]['mean']:.3f})" for m in measured_order)
        lines.append(
            f"- **H3** (model ordering): "
            f"{'CONFIRMED' if h3_pass else 'REFUTED'}. "
            f"Measured: {ordering_str}.\n"
        )
        if h4_failures:
            lines.append(
                f"- **H4** (per-category profile): REFUTED.\n  - Failing models: {h4_failures}\n"
            )
        else:
            lines.append(
                "- **H4** (per-category profile simple ≥ multiple ≥ parallel ≥ parallel_multiple): CONFIRMED for all models.\n"
            )

    # Headline
    lines.append("\n## Headline\n")
    if sweep_invalid:
        headline = f"INVALID — sweep killed at {error_rate:.0%} cell error rate."
    else:
        headline = (
            f"{cloud_best_name} ({cloud_best_mean:.3f}) > {MODEL_DENSE} ({dense_mean:.3f}) "
            f"on BFCL v3 AST — H1 {'CONFIRMED' if h1_pass else 'REFUTED'} "
            f"(delta {h1_delta:+.3f})."
        )
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
