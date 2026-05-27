"""EXP-006b — qwen3-30b-a3b-moe vs qwen3-14b-q4 vs gpt-oss-120b-cloud
on PBS-Agent v0.1, re-anchored after F-009 follow-up fixes.

Four pre-registered hypotheses (docs/exp/EXP-006b-qwen3-30b-moe-re-anchored.md):

  H1 (baseline measurement, no gate):
       Report mean(end_state | qwen3-14b-q4) with 95% bootstrap CI.
  H2 (headline, RELATIVE DELTA — promotion gate):
       lower_95_CI(MoE.end_state) >= mean(qwen3-14b-q4.end_state) + 0.10.
  H3 (gap closure, RATIO — promotion gate):
       gap_closure_pe := (moe_pe - dense_pe) / (cloud_pe - dense_pe) >= 0.50.
       Denominator non-positive -> UNDEFINED (gate fails for promotion).
  H4 (tool ceiling, relaxed — promotion gate):
       lower_95_CI(MoE.tool_correctness) >= 0.90.

Promotion rule: MoE promotes to lab-default-local iff H2 AND H3 both pass.
H4 must also pass for promotion to be quality-clean; if H4 fails but H2+H3
pass, promote and record H4 as a quality caveat.

Reads EXP-006b cells from the lab DB, computes per-cell end_state /
tool_correctness from agent_logs.turns->'score_breakdown', and writes:

  analysis/EXP-006b/SUMMARY.md          - top-line H1..H4 verdicts
  analysis/EXP-006b/verdicts.md         - full decision-rule application
  analysis/EXP-006b/per_task_endstate.csv  - per-task means side-by-side
  analysis/EXP-006b/per_cell.csv        - per-cell scorer breakdown
  analysis/EXP-006b/gap_closure.csv     - by category + overall
  analysis/EXP-006b/tokens_summary.csv  - tokens_in/out by model (new)

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
EXP_SLUG = "EXP-006b"
MODEL_DENSE = "qwen3-14b-q4"
MODEL_MOE = "qwen3-30b-a3b-moe"
MODEL_CLOUD = "gpt-oss-120b-cloud"

# Pre-registered thresholds (do NOT change after sweep starts).
H2_DELTA = 0.10  # MoE lower-CI must exceed dense PE by this much
H3_THRESH = 0.50
H4_THRESH = 0.90  # MoE tool_correctness lower-CI must exceed this

OUT_DIR = Path("analysis/EXP-006b")


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
    tokens_in: int | None
    tokens_out: int | None
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
      r.tokens_in  AS tokens_in,
      r.tokens_out AS tokens_out,
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
                    tokens_in=r.get("tokens_in"),
                    tokens_out=r.get("tokens_out"),
                    error=r.get("error"),
                )
            )
    return out


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
    """Errored / scoreless cells count as 0.0 (matches EXP-002 / F-005 / EXP-006)."""
    out: list[float] = []
    for c in cells:
        v = getattr(c, attr)
        out.append(float(v) if v is not None else 0.0)
    return out


def _category(slug: str) -> str:
    return slug.split("-", 1)[0]


def per_task_means(cells: list[Cell], attr: str) -> dict[tuple[str, str], float]:
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

    # H2: lower 95% CI on MoE.end_state >= dense.end_state + 0.10
    h2_threshold = h1_mean + H2_DELTA
    h2_lower = h2_ci[0] if h2_ci else float("nan")
    h2_pass = h2_ci is not None and h2_lower >= h2_threshold
    h2_verdict = "CONFIRMED" if h2_pass else "REFUTED"

    # H3: gap_closure_pe >= 0.50
    if h3_status == "UNDEFINED" or h3_value is None:
        h3_verdict = "UNDEFINED"
        h3_pass = False
    else:
        h3_pass = h3_value >= H3_THRESH
        h3_verdict = "CONFIRMED" if h3_pass else "REFUTED"

    # H4: lower 95% CI on MoE.tool_correctness >= 0.90
    h4_lower = h4_ci[0] if h4_ci else float("nan")
    h4_pass = h4_ci is not None and h4_lower >= H4_THRESH
    h4_verdict = "CONFIRMED" if h4_pass else "REFUTED"

    h3_val_str = f"{h3_value:.3f}" if h3_value is not None else "UNDEFINED"

    # Promotion decision
    promote = h2_pass and h3_pass
    if promote and h4_pass:
        promotion_line = (
            "**Promotion: YES — qwen3-30b-a3b-moe promotes to `lab-default-local`. "
            "H2 + H3 both pass; H4 pass means promotion is quality-clean.**"
        )
    elif promote and not h4_pass:
        promotion_line = (
            "**Promotion: YES (with caveat) — qwen3-30b-a3b-moe promotes to `lab-default-local`. "
            "H2 + H3 both pass; H4 fails — record as a quality caveat (follow-up: MoE "
            "tool-emission template audit).**"
        )
    else:
        promotion_line = (
            "**Promotion: NO — qwen3-30b-a3b-moe does NOT promote. "
            f"H2 = {h2_verdict}, H3 = {h3_verdict}. Lab default stays on qwen3-14b-q4 (reasoning-OFF).**"
        )

    lines = [
        "# EXP-006b — SUMMARY",
        "",
        f"Cells: {len(cells)}/288 (dense={n_dense}, moe={n_moe}, cloud={n_cloud}, errored={errored}).",
        "",
        "## Pre-registered hypotheses",
        "",
        f"- **H1 — Baseline measurement (no gate).** qwen3-14b-q4 end_state = "
        f"**{h1_mean:.3f}** {_ci_str(h1_ci)} (n={n_dense}). "
        f"This is the new lab reference for the post-fix PBS-Agent v0.1 surface.",
        f"- **H2 — Headline (relative delta).** qwen3-30b-a3b-moe end_state = "
        f"**{h2_mean:.3f}** {_ci_str(h2_ci)}; lower-CI = {h2_lower:.3f}; "
        f"threshold = dense_pe + {H2_DELTA:.2f} = {h2_threshold:.3f}. -> **{h2_verdict}**",
        f"- **H3 — Gap closure.** gap_closure_pe = **{h3_val_str}** "
        f"(dense={h1_mean:.3f}, moe={h2_mean:.3f}, cloud={cloud_endstate:.3f}); "
        f"threshold >= {H3_THRESH:.2f}. -> **{h3_verdict}**",
        f"- **H4 — Tool-correctness ceiling (relaxed).** qwen3-30b-a3b-moe "
        f"tool_correctness = **{h4_mean:.3f}** {_ci_str(h4_ci)}; lower-CI = {h4_lower:.3f}; "
        f"threshold >= {H4_THRESH:.2f}. -> **{h4_verdict}**",
        "",
        "## Decision",
        "",
        promotion_line,
        "",
        "## Headline",
        "",
    ]

    if promote and h4_pass:
        lines.append(
            "qwen3-30b-a3b-moe closes >=50% of the local-vs-cloud end_state gap and "
            f"beats dense by >= +{H2_DELTA*100:.0f}pp at the lower CI bound, with tool "
            "ceiling at cloud-tier. Phase 19a's local-headline thesis is supported on "
            "PBS-Agent v0.1 — lab default flips to MoE."
        )
    elif promote and not h4_pass:
        lines.append(
            "qwen3-30b-a3b-moe wins on end_state metrics (H2 + H3 both pass) but "
            "falls short on H4 tool-correctness ceiling. Promotion proceeds with "
            "a quality caveat; tool-emission template audit is the follow-up."
        )
    else:
        lines.append(
            "qwen3-30b-a3b-moe does not clear the H2 + H3 promotion gate on the "
            "re-anchored surface. Lab default stays on qwen3-14b-q4 (reasoning-OFF). "
            "F-010 records the verdicts and operational notes; further MoE work is "
            "deferred to a separate experiment."
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
    h2_threshold = h1_mean + H2_DELTA
    h2_lower = h2_ci[0] if h2_ci else float("nan")
    h2_pass = h2_ci is not None and h2_lower >= h2_threshold
    h4_lower = h4_ci[0] if h4_ci else float("nan")
    h4_pass = h4_ci is not None and h4_lower >= H4_THRESH

    if h3_status == "UNDEFINED" or h3_value is None:
        h3_outcome = "UNDEFINED — denominator non-positive"
        h3_pass = False
    else:
        h3_pass = h3_value >= H3_THRESH
        h3_outcome = "CONFIRMED" if h3_pass else "REFUTED"

    body = f"""# EXP-006b verdicts

Pre-registered decision rules, applied to the post-sweep DB read.

## H1 — Baseline measurement (NOT a gate)

Pre-reg: report mean(end_state | qwen3-14b-q4) over all 96 dense cells
with a 95% bootstrap CI. No pass/fail threshold.

- n = {n_dense}
- mean end_state = {h1_mean:.4f}
- 95% bootstrap CI = {_ci_str(h1_ci)}

This number is the new lab reference for the post-fix PBS-Agent v0.1
surface. It supersedes F-005's 0.750 anchor and EXP-006's 0.583
measurement (both of which were on different surfaces).

## H2 — Headline (RELATIVE DELTA — promotion gate)

Rule: `lower_95_CI(end_state | qwen3-30b-a3b-moe, n=96)` >=
`mean(end_state | qwen3-14b-q4, n=96) + {H2_DELTA:.2f}`.

- n = {n_moe}
- mean end_state = {h2_mean:.4f}
- 95% bootstrap CI = {_ci_str(h2_ci)}
- lower CI bound = {h2_lower:.4f}
- threshold = dense_pe + {H2_DELTA:.2f} = {h2_threshold:.4f}

Verdict: **{"CONFIRMED" if h2_pass else "REFUTED"}**.

## H3 — Gap closure (RATIO — promotion gate)

Rule: `gap_closure_pe := (moe_pe - dense_pe) / (cloud_pe - dense_pe) >= {H3_THRESH:.2f}`,
all on the same 96-cell denominator (12 tasks x 8 seeds).

- dense  end_state = {h1_mean:.4f}
- moe    end_state = {h2_mean:.4f}
- cloud  end_state = {cloud_endstate:.4f}
- denom (cloud - dense) = {cloud_endstate - h1_mean:.4f}
- numer (moe - dense)   = {h2_mean - h1_mean:.4f}
- gap_closure_pe = {h3_val_str}

Verdict: **{h3_outcome}**.

## H4 — Tool-correctness ceiling (RELAXED — promotion gate)

Rule: `lower_95_CI(tool_correctness | qwen3-30b-a3b-moe, n=96) >= {H4_THRESH:.2f}`.

- n = {n_moe}
- mean tool_correctness = {h4_mean:.4f}
- 95% bootstrap CI = {_ci_str(h4_ci)}
- lower CI bound = {h4_lower:.4f}
- threshold = {H4_THRESH:.2f}

Verdict: **{"CONFIRMED" if h4_pass else "REFUTED"}**.

Cloud-arm reference (n = {n_cloud}): end_state = {cloud_endstate:.4f}.

## Promotion rule

Pre-registered rule: promote MoE iff H2 AND H3 both pass. If H4 also
passes, promotion is quality-clean; if H4 fails but H2 + H3 pass,
promote with H4 recorded as a quality caveat (follow-up: MoE template
audit).

- H2 pass: {h2_pass}
- H3 pass: {h3_pass}
- H4 pass: {h4_pass}
- Promote: **{"YES" if (h2_pass and h3_pass) else "NO"}**
{"- Quality caveat (H4 fail): YES — follow-up audit required" if (h2_pass and h3_pass and not h4_pass) else ""}
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
            "tokens_in",
            "tokens_out",
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
                "" if c.tokens_in is None else str(c.tokens_in),
                "" if c.tokens_out is None else str(c.tokens_out),
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


def write_tokens_summary(cells: list[Cell]) -> None:
    """New for EXP-006b: report tokens_in / tokens_out by model."""
    rows = [
        [
            "model",
            "n_cells",
            "n_with_tokens",
            "tokens_in_mean",
            "tokens_in_p50",
            "tokens_out_mean",
            "tokens_out_p50",
        ]
    ]
    for model in (MODEL_DENSE, MODEL_MOE, MODEL_CLOUD):
        subset = [c for c in cells if c.model == model]
        with_tok = [c for c in subset if c.tokens_in is not None or c.tokens_out is not None]
        if with_tok:
            tin = [c.tokens_in for c in with_tok if c.tokens_in is not None]
            tout = [c.tokens_out for c in with_tok if c.tokens_out is not None]
            rows.append(
                [
                    model,
                    str(len(subset)),
                    str(len(with_tok)),
                    f"{statistics.fmean(tin):.1f}" if tin else "",
                    f"{statistics.median(tin):.0f}" if tin else "",
                    f"{statistics.fmean(tout):.1f}" if tout else "",
                    f"{statistics.median(tout):.0f}" if tout else "",
                ]
            )
        else:
            rows.append([model, str(len(subset)), "0", "", "", "", ""])
    with (OUT_DIR / "tokens_summary.csv").open("w", newline="", encoding="utf-8") as f:
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
    write_tokens_summary(cells)
    print(summary)


if __name__ == "__main__":
    main()
