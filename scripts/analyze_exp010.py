"""EXP-010 — BRUTAL-BENCH-001: validation + first ranking on pbs-agent-brutal-v0.1.

Four pre-registered hypotheses (docs/exp/EXP-010-brutal-bench.md):

  H1 (headroom):   gemma4 pass@1 <= 0.85 CONFIRMED; 0.85-0.92 INCONCLUSIVE;
                   > 0.92 REFUTED (tier failed).
  H2 (ranking):    gemma4 > qwen3-coder > devstral, strict; any pair within
                   1 task (1/24) -> INCONCLUSIVE for that pair.
  H3 (solvability): >= 22/24 tasks passed by >= 1 model CONFIRMED;
                   20-21 INCONCLUSIVE pending audit; < 20 REFUTED.
  H4 (prediction): devstral's lowest category is longhaul (ties count) AND
                   qwen3-coder's debug rate < its own overall mean.

Writes analysis/EXP-010/{SUMMARY.md,per_task_pass_map.csv} including the
defect-audit list (all-models-fail tasks).

Cost: free (purely a DB read).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

PG_DSN = "dbname=lab host=/var/run/postgresql"
DEFAULT_SLUG = "BRUTAL-BENCH-001"
EXP_DIR_NAME = "EXP-010"

MODEL_A = "gemma4-12b"
MODEL_B = "qwen3-coder-30b"
MODEL_C = "devstral-24b"

# Pre-registered thresholds (do NOT change after sweep starts).
H1_CONFIRM = 0.85
H1_REFUTE = 0.92
H2_TIE_BAND = 1 / 24
H3_CONFIRM = 22
H3_REFUTE_BELOW = 20


def load(slug: str) -> list[dict[str, object]]:
    sql = """
        select m.litellm_id as model, t.slug as task,
               coalesce(t.category, '?') as category, er.seed, er.status,
               (al.turns->'score_breakdown'->'end_state'->>'value')::float
                   as score
        from experiment_runs er
        join experiments e on e.experiment_id = er.experiment_id
        join models m on m.model_id = er.model_id
        join tasks t on t.task_id = er.task_id
        left join agent_logs al on al.run_id = er.run_id
        where e.slug = %s
        order by t.slug, m.litellm_id
    """
    with psycopg.connect(PG_DSN, row_factory=dict_row) as conn:
        return list(conn.execute(sql, (slug,)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument("--out", default=f"analysis/{EXP_DIR_NAME}")
    args = ap.parse_args()

    rows = [r for r in load(args.slug) if r["score"] is not None]
    if not rows:
        raise SystemExit(f"no scored cells for {args.slug}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted({str(r["model"]) for r in rows})
    tasks = sorted({str(r["task"]) for r in rows})
    cat_of = {str(r["task"]): str(r["category"]) for r in rows}
    score: dict[tuple[str, str], float] = {
        (str(r["model"]), str(r["task"])): float(r["score"])
        for r in rows  # type: ignore[arg-type]
    }

    overall = {m: sum(score.get((m, t), 0.0) for t in tasks) / len(tasks) for m in models}
    cats = sorted({cat_of[t] for t in tasks})
    per_cat: dict[str, dict[str, float]] = defaultdict(dict)
    for m in models:
        for cat in cats:
            ct = [t for t in tasks if cat_of[t] == cat]
            per_cat[m][cat] = sum(score.get((m, t), 0.0) for t in ct) / len(ct)

    solved = [t for t in tasks if any(score.get((m, t), 0.0) >= 1.0 for m in models)]
    unsolved = [t for t in tasks if t not in solved]

    # --- verdicts -----------------------------------------------------------
    verdicts: dict[str, str] = {}
    if all(m in overall for m in (MODEL_A, MODEL_B, MODEL_C)):
        a, b, c = overall[MODEL_A], overall[MODEL_B], overall[MODEL_C]
        verdicts["H1"] = (
            f"CONFIRMED ({a:.3f} <= {H1_CONFIRM})"
            if a <= H1_CONFIRM
            else f"REFUTED ({a:.3f} > {H1_REFUTE}; tier failed)"
            if a > H1_REFUTE
            else f"INCONCLUSIVE ({a:.3f} in ({H1_CONFIRM}, {H1_REFUTE}])"
        )
        pair_ab = (
            "strict" if a - b > H2_TIE_BAND else "tie" if abs(a - b) <= H2_TIE_BAND else "reversed"
        )
        pair_bc = (
            "strict" if b - c > H2_TIE_BAND else "tie" if abs(b - c) <= H2_TIE_BAND else "reversed"
        )
        verdicts["H2"] = (
            "CONFIRMED"
            if (pair_ab, pair_bc) == ("strict", "strict")
            else "REFUTED"
            if "reversed" in (pair_ab, pair_bc)
            else f"INCONCLUSIVE (gemma4-vs-qwen3 {pair_ab}, qwen3-vs-devstral {pair_bc})"
        )
        n_solved = len(solved)
        verdicts["H3"] = (
            f"CONFIRMED ({n_solved}/{len(tasks)} solvable)"
            if n_solved >= H3_CONFIRM
            else f"REFUTED ({n_solved}/{len(tasks)}; suite revision required)"
            if n_solved < H3_REFUTE_BELOW
            else f"INCONCLUSIVE ({n_solved}/{len(tasks)}; audit pending)"
        )
        dev_min = min(per_cat[MODEL_C].values())
        dev_longhaul_lowest = per_cat[MODEL_C].get("longhaul", 1.0) == dev_min
        qwen_debug_below = per_cat[MODEL_B].get("debug", 1.0) < overall[MODEL_B]
        verdicts["H4"] = (
            "CONFIRMED"
            if dev_longhaul_lowest and qwen_debug_below
            else "REFUTED"
            + (" (devstral lowest != longhaul)" if not dev_longhaul_lowest else "")
            + (" (qwen3 debug >= own mean)" if not qwen_debug_below else "")
        )
    else:
        verdicts["note"] = "expected models absent; no verdicts"

    # --- outputs ------------------------------------------------------------
    with (out_dir / "per_task_pass_map.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "category", *models, "any_pass"])
        for t in tasks:
            marks = ["1" if score.get((m, t), 0.0) >= 1.0 else "0" for m in models]
            w.writerow([t, cat_of[t], *marks, "1" if t in solved else "0"])

    lines = [f"# {EXP_DIR_NAME} / {args.slug} — summary", ""]
    lines.append("| model | overall | " + " | ".join(cats) + " |")
    lines.append("| --- | --- |" + " --- |" * len(cats))
    for m in models:
        lines.append(
            f"| {m} | {overall[m]:.3f} | " + " | ".join(f"{per_cat[m][c]:.3f}" for c in cats) + " |"
        )
    lines.append("")
    lines.append("## Verdicts")
    lines.append("")
    for h, v in verdicts.items():
        lines.append(f"- **{h}**: {v}")
    lines.append("")
    lines.append(f"## Defect-audit list ({len(unsolved)} all-models-fail tasks)")
    lines.append("")
    for t in unsolved:
        lines.append(f"- {t} ({cat_of[t]}) — audit trajectory before trusting")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
