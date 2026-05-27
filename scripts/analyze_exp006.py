"""EXP-006 — qwen3-30b-a3b-moe vs qwen3-14b-q4 vs gpt-oss-120b-cloud
on PBS-Agent v0.1.

Four pre-registered hypotheses (docs/exp/EXP-006-qwen3-30b-moe-vs-14b-dense.md):

  H1 (replication):  |mean(end_state | qwen3-14b-q4) - 0.750| <= 0.05.
                     Outside band -> H1 REFUTED, sweep INVALID.
  H2 (headline):     mean(end_state | qwen3-30b-a3b-moe) >= 0.850.
  H3 (gap closure):  gap_closure := (moe - dense) / (cloud - dense) >= 0.50.
                     Denominator non-positive -> UNDEFINED.
  H4 (tool ceiling): mean(tool_correctness | qwen3-30b-a3b-moe) >= 0.95.

Reads EXP-006 cells from the lab DB, computes per-cell end_state /
tool_correctness from agent_logs.turns->'score_breakdown', and writes:

  analysis/EXP-006/SUMMARY.md          — top-line H1..H4 verdicts
  analysis/EXP-006/verdicts.md         — full decision-rule application
  analysis/EXP-006/per_task_endstate.csv  — per-task means side-by-side
  analysis/EXP-006/per_cell.csv        — per-cell scorer breakdown
  analysis/EXP-006/gap_closure.csv     — by category + overall

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
EXP_SLUG = "EXP-006"
MODEL_DENSE = "qwen3-14b-q4"
MODEL_MOE = "qwen3-30b-a3b-moe"
MODEL_CLOUD = "gpt-oss-120b-cloud"

# Pre-registered thresholds (do NOT change after sweep starts).
H1_ANCHOR = 0.750
H1_BAND = 0.05
H2_THRESH = 0.850
H3_THRESH = 0.50
H4_THRESH = 0.95

OUT_DIR = Path("analysis/EXP-006")


@dataclass
class Cell:
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


def fetch_cells(exp_slug: str) -> list[Cell]:
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
    ORDER BY 1, 2, 3
    """
    out: list[Cell] = []
    with psycopg.connect(PG_DSN, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(sql, [exp_slug])
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


def _cell_score(cells: list[Cell], attr: str) -> list[float]:
    """Errored / scoreless cells count as 0.0 (matches EXP-002 / F-005)."""
    out: list[float] = []
    for c in cells:
        v = getattr(c, attr)
        out.append(float(v) if v is not None else 0.0)
    return out


def _category(slug: str) -> str:
    return slug.split("-", 1)[0]


def per_task_means(cells: list[Cell], attr: str) -> dict[tuple[str, str], float]:
    """{(model, task): mean over seeds}."""
    buckets: dict[tuple[str, str], list[float]] = {}
    for c in cells:
        buckets.setdefault((c.model, c.task), []).append(
            float(getattr(c, attr)) if getattr(c, attr) is not None else 0.0
        )
    return {k: statistics.fmean(v) for k, v in buckets.items()}


def model_mean(
    cells: list[Cell], model: str, attr: str
) -> tuple[float, tuple[float, float] | None, int]:
    subset = [c for c in cells if c.model == model]
    vals = _cell_score(subset, attr)
    if not vals:
        return float("nan"), None, 0
    return statistics.fmean(vals), _bootstrap_ci(vals), len(vals)


def gap_closure(dense: float, moe: float, cloud: float) -> tuple[float | None, str]:
    """Return (value, status). status in {DEFINED, UNDEFINED}."""
    denom = cloud - dense
    if denom <= 0:
        return None, "UNDEFINED"
    return (moe - dense) / denom, "DEFINED"


def write_summary(
    *,
    cells: list[Cell],
    h1_mean: float,
    h1_ci: tuple[float, float] | None,
    h2_mean: float,
    h2_ci: tuple[float, float] | None,
    h3_value: float | None,
    h3_status: str,
    h4_mean: float,
    h4_ci: tuple[float, float] | None,
    cloud_endstate: float,
    n_dense: int,
    n_moe: int,
    n_cloud: int,
    errored: int,
) -> str:
    def _ci_str(ci: tuple[float, float] | None) -> str:
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "(n/a)"

    h1_verdict = "CONFIRMED" if abs(h1_mean - H1_ANCHOR) <= H1_BAND else "REFUTED"
    h1_invalid = h1_verdict == "REFUTED"
    h2_verdict = "CONFIRMED" if h2_mean >= H2_THRESH else "REFUTED"
    if h3_status == "UNDEFINED" or h3_value is None:
        h3_verdict = "UNDEFINED"
    else:
        h3_verdict = "CONFIRMED" if h3_value >= H3_THRESH else "REFUTED"
    h4_verdict = "CONFIRMED" if h4_mean >= H4_THRESH else "REFUTED"

    if h1_invalid:
        h2_verdict = "INVALID — H1 replication failed; H2 result not load-bearing"
        h3_verdict = "INVALID — H1 replication failed; H3 result not load-bearing"
        h4_verdict = "INVALID — H1 replication failed; H4 result not load-bearing"

    h3_val_str = f"{h3_value:.3f}" if h3_value is not None else "UNDEFINED"

    lines = [
        "# EXP-006 — SUMMARY",
        "",
        f"Cells: {len(cells)}/288 (dense={n_dense}, moe={n_moe}, cloud={n_cloud}, errored={errored}).",
        "",
        "## Pre-registered hypotheses",
        "",
        f"- **H1 — Replication.** qwen3-14b-q4 end_state = **{h1_mean:.3f}** {_ci_str(h1_ci)} ; "
        f"target {H1_ANCHOR:.3f} ± {H1_BAND:.2f}. → **{h1_verdict}**",
        f"- **H2 — Headline lift.** qwen3-30b-a3b-moe end_state = **{h2_mean:.3f}** {_ci_str(h2_ci)} ; "
        f"threshold ≥ {H2_THRESH:.3f}. → **{h2_verdict}**",
        f"- **H3 — Gap closure.** gap_closure = **{h3_val_str}** "
        f"(dense={h1_mean:.3f}, moe={h2_mean:.3f}, cloud={cloud_endstate:.3f}); "
        f"threshold ≥ {H3_THRESH:.2f}. → **{h3_verdict}**",
        f"- **H4 — Tool-correctness ceiling.** qwen3-30b-a3b-moe tool_correctness = **{h4_mean:.3f}** "
        f"{_ci_str(h4_ci)} ; threshold ≥ {H4_THRESH:.2f}. → **{h4_verdict}**",
        "",
        "## Headline",
        "",
    ]

    # Headline narrative
    if h1_invalid:
        lines.append(
            "Sweep INVALID — H1 (replication of F-005's qwen3-14b-q4 baseline) is outside "
            "the ±0.05 pp band. H2/H3/H4 verdicts are not load-bearing."
        )
    elif h2_verdict == "CONFIRMED" and h3_verdict == "CONFIRMED":
        lines.append(
            "qwen3-30b-a3b-moe closes ≥50% of the local-vs-cloud end_state gap and "
            "clears the +10pp absolute-lift threshold. Phase 19a's local-headline thesis "
            "is supported on PBS-Agent v0.1."
        )
    elif h2_verdict == "CONFIRMED":
        lines.append(
            "qwen3-30b-a3b-moe clears the +10pp absolute lift over qwen3-14b-q4, but "
            "does NOT close ≥50% of the local-vs-cloud gap. Local MoE helps; cloud "
            "still dominates as a ceiling."
        )
    else:
        lines.append(
            "qwen3-30b-a3b-moe fails to clear the +10pp lift over qwen3-14b-q4. "
            "Phase 19a's local-headline thesis is REFUTED on PBS-Agent v0.1."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "SUMMARY.md").write_text(summary, encoding="utf-8")
    return summary


def write_verdicts(
    *,
    h1_mean: float,
    h1_ci: tuple[float, float] | None,
    n_dense: int,
    h2_mean: float,
    h2_ci: tuple[float, float] | None,
    n_moe: int,
    h3_value: float | None,
    h3_status: str,
    h4_mean: float,
    h4_ci: tuple[float, float] | None,
    cloud_endstate: float,
    n_cloud: int,
) -> None:
    def _ci_str(ci: tuple[float, float] | None) -> str:
        return f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "(n/a)"

    h3_val_str = f"{h3_value:.4f}" if h3_value is not None else "UNDEFINED"
    body = f"""# EXP-006 verdicts

Pre-registered decision rules, applied to the post-sweep DB read.

## H1 — Replication of F-005 qwen3-14b-q4 baseline

Rule: `|mean(end_state | qwen3-14b-q4, all 96 cells) − {H1_ANCHOR:.3f}| ≤ {H1_BAND:.2f}`.

- n = {n_dense}
- mean end_state = {h1_mean:.4f}
- 95% bootstrap CI = {_ci_str(h1_ci)}
- |observed − anchor| = {abs(h1_mean - H1_ANCHOR):.4f}
- pre-reg band = {H1_BAND:.2f}

Verdict: **{"CONFIRMED" if abs(h1_mean - H1_ANCHOR) <= H1_BAND else "REFUTED — sweep INVALID"}**.

## H2 — Headline lift (qwen3-30b-a3b-moe end_state ≥ {H2_THRESH:.3f})

Rule: `mean(end_state | qwen3-30b-a3b-moe, all 96 cells) ≥ {H2_THRESH:.3f}`.

- n = {n_moe}
- mean end_state = {h2_mean:.4f}
- 95% bootstrap CI = {_ci_str(h2_ci)}
- threshold = {H2_THRESH:.3f}

Verdict: **{"CONFIRMED" if h2_mean >= H2_THRESH else "REFUTED"}**.

## H3 — Gap closure (≥ {H3_THRESH:.2f})

Rule: `gap_closure := (moe − dense) / (cloud − dense) ≥ {H3_THRESH:.2f}`,
all three terms on the same 96-cell denominator (12 tasks × 8 seeds).

- dense  end_state = {h1_mean:.4f}
- moe    end_state = {h2_mean:.4f}
- cloud  end_state = {cloud_endstate:.4f}
- denom (cloud − dense) = {cloud_endstate - h1_mean:.4f}
- numer (moe − dense)   = {h2_mean - h1_mean:.4f}
- gap_closure = {h3_val_str}

Verdict: **{
        "CONFIRMED"
        if h3_status == "DEFINED" and h3_value is not None and h3_value >= H3_THRESH
        else ("REFUTED" if h3_status == "DEFINED" else "UNDEFINED — denominator non-positive")
    }**.

## H4 — Tool-correctness ceiling (qwen3-30b-a3b-moe ≥ {H4_THRESH:.2f})

Rule: `mean(tool_correctness | qwen3-30b-a3b-moe, all 96 cells) ≥ {H4_THRESH:.2f}`.

- n = {n_moe}
- mean tool_correctness = {h4_mean:.4f}
- 95% bootstrap CI = {_ci_str(h4_ci)}
- threshold = {H4_THRESH:.2f}

Verdict: **{"CONFIRMED" if h4_mean >= H4_THRESH else "REFUTED"}**.

Cloud-arm reference (n = {n_cloud}): end_state = {cloud_endstate:.4f}.
"""
    (OUT_DIR / "verdicts.md").write_text(body, encoding="utf-8")


def write_per_task_csv(cells: list[Cell]) -> None:
    per = per_task_means(cells, "end_state")
    tasks_sorted = sorted({c.task for c in cells})
    rows = [["task", "category", MODEL_DENSE, MODEL_MOE, MODEL_CLOUD]]
    for t in tasks_sorted:
        rows.append(
            [
                t,
                _category(t),
                f"{per.get((MODEL_DENSE, t), float('nan')):.3f}",
                f"{per.get((MODEL_MOE, t), float('nan')):.3f}",
                f"{per.get((MODEL_CLOUD, t), float('nan')):.3f}",
            ]
        )
    with (OUT_DIR / "per_task_endstate.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def write_per_cell_csv(cells: list[Cell]) -> None:
    rows = [
        [
            "model",
            "task",
            "category",
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
    ]
    for c in sorted(cells, key=lambda c: (c.model, c.task, c.seed)):
        rows.append(
            [
                c.model,
                c.task,
                _category(c.task),
                str(c.seed),
                c.status,
                "" if c.end_state is None else f"{c.end_state:.3f}",
                "" if c.tool_correctness is None else f"{c.tool_correctness:.3f}",
                "" if c.budget_respected is None else f"{c.budget_respected:.3f}",
                "" if c.trajectory_judge is None else f"{c.trajectory_judge:.3f}",
                "" if c.actual_turns is None else str(c.actual_turns),
                "" if c.tool_call_count is None else str(c.tool_call_count),
                (c.error or "")[:200],
            ]
        )
    with (OUT_DIR / "per_cell.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def write_gap_closure_csv(cells: list[Cell]) -> None:
    cats = sorted({_category(c.task) for c in cells})
    rows = [
        ["category", "dense_end_state", "moe_end_state", "cloud_end_state", "gap_closure", "status"]
    ]
    for cat in [*cats, "__overall__"]:
        subset = cells if cat == "__overall__" else [c for c in cells if _category(c.task) == cat]
        d = (
            statistics.fmean(
                _cell_score([c for c in subset if c.model == MODEL_DENSE], "end_state")
            )
            if any(c.model == MODEL_DENSE for c in subset)
            else float("nan")
        )
        m = (
            statistics.fmean(_cell_score([c for c in subset if c.model == MODEL_MOE], "end_state"))
            if any(c.model == MODEL_MOE for c in subset)
            else float("nan")
        )
        cl = (
            statistics.fmean(
                _cell_score([c for c in subset if c.model == MODEL_CLOUD], "end_state")
            )
            if any(c.model == MODEL_CLOUD for c in subset)
            else float("nan")
        )
        if not (math.isnan(d) or math.isnan(m) or math.isnan(cl)) and (cl - d) > 0:
            g = (m - d) / (cl - d)
            status = "DEFINED"
            gstr = f"{g:.3f}"
        else:
            status = "UNDEFINED"
            gstr = ""
        rows.append([cat, f"{d:.3f}", f"{m:.3f}", f"{cl:.3f}", gstr, status])
    with (OUT_DIR / "gap_closure.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def main() -> None:
    cells = fetch_cells(EXP_SLUG)
    if not cells:
        print(f"no cells found for {EXP_SLUG}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    h1_mean, h1_ci, n_dense = model_mean(cells, MODEL_DENSE, "end_state")
    h2_mean, h2_ci, n_moe = model_mean(cells, MODEL_MOE, "end_state")
    h4_mean, h4_ci, _ = model_mean(cells, MODEL_MOE, "tool_correctness")
    cloud_endstate, _, n_cloud = model_mean(cells, MODEL_CLOUD, "end_state")
    h3_value, h3_status = gap_closure(h1_mean, h2_mean, cloud_endstate)

    errored = sum(1 for c in cells if c.status == "error")

    summary = write_summary(
        cells=cells,
        h1_mean=h1_mean,
        h1_ci=h1_ci,
        h2_mean=h2_mean,
        h2_ci=h2_ci,
        h3_value=h3_value,
        h3_status=h3_status,
        h4_mean=h4_mean,
        h4_ci=h4_ci,
        cloud_endstate=cloud_endstate,
        n_dense=n_dense,
        n_moe=n_moe,
        n_cloud=n_cloud,
        errored=errored,
    )
    write_verdicts(
        h1_mean=h1_mean,
        h1_ci=h1_ci,
        n_dense=n_dense,
        h2_mean=h2_mean,
        h2_ci=h2_ci,
        n_moe=n_moe,
        h3_value=h3_value,
        h3_status=h3_status,
        h4_mean=h4_mean,
        h4_ci=h4_ci,
        cloud_endstate=cloud_endstate,
        n_cloud=n_cloud,
    )
    write_per_task_csv(cells)
    write_per_cell_csv(cells)
    write_gap_closure_csv(cells)
    print(summary)


if __name__ == "__main__":
    main()
