"""EXP-003b hypothesis verdicts + per-(model, condition) tables.

Pre-registered four hypotheses (docs/exp/EXP-003b.md). Reads per-cell
scorer breakdowns from agent_logs.turns->'score_breakdown', distinguishes
the two conditions (with-kb / without-kb) via experiment_runs.config->>'name',
joins to models / tasks, and emits the verdict for each hypothesis per
its pre-registered decision rule.

Outputs:
  - analysis/EXP-003b/SUMMARY.md
  - analysis/EXP-003b/verdicts.md
  - analysis/EXP-003b/per_model_condition.csv
  - analysis/EXP-003b/per_cell_runs.csv
  - analysis/EXP-003b/faithfulness_slice.csv
  - analysis/EXP-003b/kb_query_invocations.csv

Tables read (lab DB):
  - experiments, experiment_runs, models, tasks, agent_logs (turns JSONB)
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

PG_DSN = "dbname=lab host=/var/run/postgresql"
SLUG = "EXP-003b"

LOCAL_MODELS = ["qwen3-14b-q4", "llama3.1-8b-q4"]
CLOUD_MODELS = ["gpt-oss-20b-cloud", "glm-5.1-cloud", "gpt-oss-120b-cloud"]
ALL_MODELS = LOCAL_MODELS + CLOUD_MODELS

CONDITIONS = ["with-kb", "without-kb"]
FAITHFULNESS_TASK = "rag-bash-faithful-answer-shopt"

OUT_DIR = Path(__file__).resolve().parents[1] / "analysis" / "EXP-003b"


@dataclass
class Cell:
    model: str
    task: str
    condition: str
    seed: int
    status: str
    end_state: float | None
    tool_correctness: float | None
    budget_respected: float | None
    recall_at_k: float | None
    mrr: float | None
    ndcg: float | None
    attribution: float | None
    faithfulness: float | None
    trajectory_judge: float | None
    actual_turns: int | None
    tool_call_count: int | None
    kb_query_calls: int
    error: str | None
    predicate_type: str | None
    turns_payload: list[dict[str, Any]] | None


def _score(scores: dict[str, Any], name: str) -> float | None:
    v = (scores.get(name) or {}).get("value")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def fetch_cells() -> list[Cell]:
    sql = """
    SELECT
      m.litellm_id     AS model,
      t.slug           AS task,
      t.success_predicate AS predicate,
      r.seed           AS seed,
      r.status         AS status,
      r.actual_turns   AS actual_turns,
      r.tool_call_count AS tool_call_count,
      r.error          AS error,
      r.config         AS config,
      a.turns          AS turns
    FROM experiment_runs r
    JOIN models m USING (model_id)
    JOIN tasks  t ON t.task_id = r.task_id
    LEFT JOIN agent_logs a ON a.run_id = r.run_id
    WHERE r.experiment_id = (SELECT experiment_id FROM experiments WHERE slug = %s)
    ORDER BY 1, 2, 3, 4
    """
    out: list[Cell] = []
    with psycopg.connect(PG_DSN, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(sql, (SLUG,))
        for r in cur.fetchall():
            agent = (r.get("turns") or {}) if isinstance(r.get("turns"), dict) else {}
            scores = agent.get("score_breakdown") or {}
            turn_entries = agent.get("turns") if isinstance(agent.get("turns"), list) else None
            config = r.get("config") or {}
            cond = config.get("name") if isinstance(config, dict) else None
            if cond not in CONDITIONS:
                cond = "with-kb"  # default if unset

            actual_turns = r.get("actual_turns")
            if actual_turns is None and turn_entries:
                actual_turns = len(turn_entries)
            tool_call_count = r.get("tool_call_count")
            if tool_call_count is None and turn_entries:
                tool_call_count = sum(len(t.get("tools") or []) for t in turn_entries)
            kb_calls = 0
            if turn_entries:
                for t in turn_entries:
                    for tc in t.get("tools") or []:
                        if (tc.get("tool") or tc.get("name")) == "kb_query":
                            kb_calls += 1

            pred = r.get("predicate") or {}
            predicate_type = pred.get("type") if isinstance(pred, dict) else None

            out.append(
                Cell(
                    model=r["model"],
                    task=r["task"],
                    condition=cond,
                    seed=int(r["seed"]),
                    status=r["status"],
                    end_state=_score(scores, "end_state"),
                    tool_correctness=_score(scores, "tool_correctness"),
                    budget_respected=_score(scores, "budget_respected"),
                    recall_at_k=_score(scores, "recall_at_k"),
                    mrr=_score(scores, "mrr"),
                    ndcg=_score(scores, "ndcg"),
                    attribution=_score(scores, "attribution"),
                    faithfulness=_score(scores, "faithfulness"),
                    trajectory_judge=_score(scores, "trajectory_judge"),
                    actual_turns=actual_turns,
                    tool_call_count=tool_call_count,
                    kb_query_calls=kb_calls,
                    error=r.get("error"),
                    predicate_type=predicate_type,
                    turns_payload=turn_entries,
                )
            )
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs: list[float], n_resamples: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    import random

    rng = random.Random(0)
    n = len(xs)
    means: list[float] = []
    for _ in range(n_resamples):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(n_resamples * (alpha / 2))], means[int(n_resamples * (1 - alpha / 2))])


def model_class(m: str) -> str:
    return "local" if m in LOCAL_MODELS else "cloud" if m in CLOUD_MODELS else "?"


# ----------------------------------------------------------------------
# CSV writers
# ----------------------------------------------------------------------


def write_per_cell_runs(cells: list[Cell], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model", "task", "condition", "seed", "status",
                "end_state", "tool_correctness", "budget_respected",
                "recall_at_k", "mrr", "ndcg", "attribution",
                "faithfulness", "trajectory_judge",
                "actual_turns", "tool_call_count", "kb_query_calls",
                "predicate_type", "error",
            ]
        )
        for c in cells:
            w.writerow(
                [
                    c.model, c.task, c.condition, c.seed, c.status,
                    c.end_state, c.tool_correctness, c.budget_respected,
                    c.recall_at_k, c.mrr, c.ndcg, c.attribution,
                    c.faithfulness, c.trajectory_judge,
                    c.actual_turns, c.tool_call_count, c.kb_query_calls,
                    c.predicate_type, (c.error or "")[:200],
                ]
            )


def write_per_model_condition(cells: list[Cell], path: Path) -> None:
    """Mean of end_state, tool_correctness, faithfulness etc. per
    (model, condition).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[Any]] = []
    for m in ALL_MODELS:
        for cond in CONDITIONS:
            mc_cells = [c for c in cells if c.model == m and c.condition == cond]
            es = [float(c.end_state) for c in mc_cells if c.end_state is not None]
            tc = [float(c.tool_correctness) for c in mc_cells if c.tool_correctness is not None]
            faith = [float(c.faithfulness) for c in mc_cells if c.faithfulness is not None]
            rec = [float(c.recall_at_k) for c in mc_cells if c.recall_at_k is not None]
            attr = [float(c.attribution) for c in mc_cells if c.attribution is not None]
            kb_calls = [c.kb_query_calls for c in mc_cells]
            rows.append(
                [
                    m, cond, len(mc_cells),
                    f"{mean(es):.3f}", len(es),
                    f"{mean(tc):.3f}", len(tc),
                    f"{mean(faith):.3f}", len(faith),
                    f"{mean(rec):.3f}", len(rec),
                    f"{mean(attr):.3f}", len(attr),
                    f"{mean([float(x) for x in kb_calls]):.2f}",
                ]
            )
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model", "condition", "n_cells",
                "end_state_mean", "n_end_state",
                "tool_correctness_mean", "n_tool_correctness",
                "faithfulness_mean", "n_faithfulness",
                "recall_at_k_mean", "n_recall",
                "attribution_mean", "n_attribution",
                "mean_kb_query_calls",
            ]
        )
        for r in rows:
            w.writerow(r)


def write_faithfulness_slice(cells: list[Cell], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [c for c in cells if c.task == FAITHFULNESS_TASK]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "seed", "faithfulness", "end_state", "kb_query_calls"])
        for c in rows:
            w.writerow([c.model, c.condition, c.seed, c.faithfulness, c.end_state, c.kb_query_calls])


def write_kb_query_invocations(cells: list[Cell], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mean(kb_query_calls) per (model, task) over the WITH condition
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "condition", "n_cells", "mean_kb_query_calls", "min_kb_query_calls"])
        for m in ALL_MODELS:
            tasks = sorted({c.task for c in cells if c.model == m})
            for t in tasks:
                for cond in CONDITIONS:
                    mt_cells = [c for c in cells if c.model == m and c.task == t and c.condition == cond]
                    if not mt_cells:
                        continue
                    calls = [c.kb_query_calls for c in mt_cells]
                    w.writerow(
                        [
                            m, t, cond, len(mt_cells),
                            f"{mean([float(x) for x in calls]):.2f}",
                            min(calls) if calls else 0,
                        ]
                    )


# ----------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------


def compute_h1(cells: list[Cell]) -> dict[str, Any]:
    """H1 -- delta_local - delta_cloud >= 0.10 on end_state."""
    def class_delta(class_models: list[str]) -> tuple[float, float, float, int]:
        with_es = [
            float(c.end_state) for c in cells
            if c.model in class_models and c.condition == "with-kb" and c.end_state is not None
        ]
        without_es = [
            float(c.end_state) for c in cells
            if c.model in class_models and c.condition == "without-kb" and c.end_state is not None
        ]
        w = mean(with_es)
        wo = mean(without_es)
        n = len(with_es) + len(without_es)
        return w, wo, w - wo, n

    w_local, wo_local, delta_local, n_local = class_delta(LOCAL_MODELS)
    w_cloud, wo_cloud, delta_cloud, n_cloud = class_delta(CLOUD_MODELS)
    diff = delta_local - delta_cloud if not (math.isnan(delta_local) or math.isnan(delta_cloud)) else float("nan")
    verdict = "CONFIRMED" if not math.isnan(diff) and diff >= 0.10 else "REFUTED"
    return {
        "verdict": verdict,
        "delta_local": delta_local,
        "delta_cloud": delta_cloud,
        "delta_local_minus_cloud": diff,
        "with_local": w_local, "without_local": wo_local,
        "with_cloud": w_cloud, "without_cloud": wo_cloud,
        "n_local_runs": n_local, "n_cloud_runs": n_cloud,
    }


def compute_h2(cells: list[Cell]) -> dict[str, Any]:
    """H2 — every (model, task) cell in with-kb condition has mean kb_query calls ≥ 1.0."""
    failures: list[tuple[str, str, float]] = []
    n_checked = 0
    for m in ALL_MODELS:
        tasks = sorted({c.task for c in cells if c.model == m})
        for t in tasks:
            mt = [c for c in cells if c.model == m and c.task == t and c.condition == "with-kb"]
            if not mt:
                continue
            n_checked += 1
            mc = mean([float(c.kb_query_calls) for c in mt])
            if mc < 1.0:
                failures.append((m, t, mc))
    verdict = "CONFIRMED" if n_checked > 0 and not failures else "REFUTED"
    return {"verdict": verdict, "n_cells_checked": n_checked, "failures": failures}


def compute_h3(cells: list[Cell]) -> dict[str, Any]:
    """H3 — faithfulness with-kb - without-kb ≥ 0.10 on task 6."""
    with_f = [
        float(c.faithfulness) for c in cells
        if c.task == FAITHFULNESS_TASK and c.condition == "with-kb" and c.faithfulness is not None
    ]
    without_f = [
        float(c.faithfulness) for c in cells
        if c.task == FAITHFULNESS_TASK and c.condition == "without-kb" and c.faithfulness is not None
    ]
    m_with = mean(with_f)
    m_without = mean(without_f)
    delta = m_with - m_without if not (math.isnan(m_with) or math.isnan(m_without)) else float("nan")
    if math.isnan(delta) or not with_f or not without_f:
        verdict = "UNDEFINED"
    else:
        verdict = "CONFIRMED" if delta >= 0.10 else "REFUTED"
    return {
        "verdict": verdict, "delta": delta,
        "with_mean": m_with, "without_mean": m_without,
        "n_with": len(with_f), "n_without": len(without_f),
    }


def compute_h4(cells: list[Cell]) -> dict[str, Any]:
    """H4 — ∃ (model, task) cell in without-kb with appropriate predicate type
    and mean(end_state) ≤ 0.25.
    """
    candidates: list[tuple[str, str, float, int]] = []
    eligible_predicates = {"retrieval_recall", "workspace_file_contains"}
    for m in ALL_MODELS:
        tasks = sorted({c.task for c in cells if c.model == m})
        for t in tasks:
            mt = [
                c for c in cells
                if c.model == m and c.task == t and c.condition == "without-kb"
                and c.predicate_type in eligible_predicates and c.end_state is not None
            ]
            if not mt:
                continue
            es = mean([float(c.end_state) for c in mt if c.end_state is not None])
            if es <= 0.25:
                candidates.append((m, t, es, len(mt)))
    verdict = "CONFIRMED" if candidates else "REFUTED"
    return {"verdict": verdict, "failing_cells": candidates}


def write_verdicts_md(
    cells: list[Cell], h1: dict[str, Any], h2: dict[str, Any], h3: dict[str, Any], h4: dict[str, Any], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    n = len(cells)
    n_done = sum(1 for c in cells if c.status == "done")
    n_err = sum(1 for c in cells if c.status == "error")
    lines.append(f"# EXP-003b verdicts — {n} cells ({n_done} done, {n_err} error)\n")

    # H1
    lines.append("## H1 — Locals gain more from kb_query than cloud (delta_local - delta_cloud >= 0.10)\n")
    lines.append(f"- delta_local  (local with - local without):   **{h1['delta_local']:.3f}**  "
                 f"(with={h1['with_local']:.3f}, without={h1['without_local']:.3f}, n={h1['n_local_runs']})")
    lines.append(f"- delta_cloud  (cloud with - cloud without):   **{h1['delta_cloud']:.3f}**  "
                 f"(with={h1['with_cloud']:.3f}, without={h1['without_cloud']:.3f}, n={h1['n_cloud_runs']})")
    lines.append(f"- difference: **{h1['delta_local_minus_cloud']:.3f}**  (threshold 0.100)")
    lines.append(f"- **H1: {h1['verdict']}**\n")

    # H2
    lines.append("## H2 — Models actually call kb_query when available (mean >= 1.0 per (model, task) with-kb cell)\n")
    lines.append(f"- (model, task) cells checked: {h2['n_cells_checked']}")
    if h2["failures"]:
        lines.append("- FAILING cells (mean kb_query calls < 1.0):")
        for m, t, c in h2["failures"]:
            lines.append(f"  - {m} / {t}: mean={c:.2f}")
    else:
        lines.append("- no failing cells")
    lines.append(f"- **H2: {h2['verdict']}**\n")

    # H3
    lines.append(f"## H3 — Faithfulness improves with kb_query on {FAITHFULNESS_TASK} (delta >= 0.10)\n")
    lines.append(f"- with-kb mean faithfulness:   **{h3['with_mean']:.3f}** (n={h3['n_with']})")
    lines.append(f"- without-kb mean faithfulness: **{h3['without_mean']:.3f}** (n={h3['n_without']})")
    lines.append(f"- delta: **{h3['delta']:.3f}**  (threshold 0.100)")
    lines.append(f"- **H3: {h3['verdict']}**\n")

    # H4
    lines.append("## H4 — At least one (model, task) cell in without-kb with mean(end_state) <= 0.25 on a KB task\n")
    if h4["failing_cells"]:
        lines.append("- Failing (model, task) cells in without-kb:")
        for m, t, es, nc in h4["failing_cells"]:
            lines.append(f"  - {m} / {t}: mean end_state = {es:.3f} (n={nc})")
    else:
        lines.append("- no without-kb cells with end_state <= 0.25 (no catastrophic bluff failure)")
    lines.append(f"- **H4: {h4['verdict']}**\n")

    path.write_text("\n".join(lines) + "\n")


def write_summary_md(
    cells: list[Cell], h1: dict[str, Any], h2: dict[str, Any], h3: dict[str, Any], h4: dict[str, Any], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# EXP-003b — SUMMARY\n")
    n = len(cells)
    n_done = sum(1 for c in cells if c.status == "done")
    n_err = sum(1 for c in cells if c.status == "error")
    lines.append(f"Cells: {n} ({n_done} done, {n_err} error)\n")

    lines.append("\n## Per-(model, condition) end_state means\n")
    lines.append("| model | with-kb | without-kb | delta |")
    lines.append("|---|---|---|---|")
    for m in ALL_MODELS:
        w_vals = [float(c.end_state) for c in cells if c.model == m and c.condition == "with-kb" and c.end_state is not None]
        wo_vals = [float(c.end_state) for c in cells if c.model == m and c.condition == "without-kb" and c.end_state is not None]
        wm = mean(w_vals)
        wo = mean(wo_vals)
        d = wm - wo if not (math.isnan(wm) or math.isnan(wo)) else float("nan")
        lines.append(f"| {m} | {wm:.3f} (n={len(w_vals)}) | {wo:.3f} (n={len(wo_vals)}) | {d:+.3f} |")

    lines.append("\n## Verdicts\n")
    lines.append(f"- **H1** (locals gain more): **{h1['verdict']}** (delta_local - delta_cloud = {h1['delta_local_minus_cloud']:+.3f})")
    lines.append(f"- **H2** (models call kb_query): **{h2['verdict']}** ({len(h2['failures'])} failing cells)")
    lines.append(f"- **H3** (faithfulness improves with KB): **{h3['verdict']}** (delta = {h3['delta']:+.3f})")
    lines.append(f"- **H4** (catastrophic without-KB on KB task): **{h4['verdict']}** ({len(h4['failing_cells'])} failing cells)")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    cells = fetch_cells()
    if not cells:
        print("ERROR: no cells found for EXP-003b", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_per_cell_runs(cells, OUT_DIR / "per_cell_runs.csv")
    write_per_model_condition(cells, OUT_DIR / "per_model_condition.csv")
    write_faithfulness_slice(cells, OUT_DIR / "faithfulness_slice.csv")
    write_kb_query_invocations(cells, OUT_DIR / "kb_query_invocations.csv")

    h1 = compute_h1(cells)
    h2 = compute_h2(cells)
    h3 = compute_h3(cells)
    h4 = compute_h4(cells)

    write_verdicts_md(cells, h1, h2, h3, h4, OUT_DIR / "verdicts.md")
    write_summary_md(cells, h1, h2, h3, h4, OUT_DIR / "SUMMARY.md")

    n_total = len(cells)
    n_done = sum(1 for c in cells if c.status == "done")
    print(
        f"EXP-003b analyzed: {n_done}/{n_total} done  "
        f"H1={h1['verdict']} H2={h2['verdict']} H3={h3['verdict']} H4={h4['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
