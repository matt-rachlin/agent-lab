"""EXP-001 hypothesis verdicts.

Pre-registered four hypotheses. This script computes the verdicts AFTER the sweep
+ deterministic + judge evaluators have all populated. Outputs a markdown report
ready to paste into F-003.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
from scipy import stats

PG = "dbname=lab host=/var/run/postgresql"
SLUG = "EXP-001"

LOCAL_MODELS = ["qwen3-14b-q4", "llama3.1-8b-q4", "gemma3-12b-q4", "phi4"]
CLOUD_MODELS = ["gpt-oss-20b-cloud", "gpt-oss-120b-cloud"]
ALL_MODELS = LOCAL_MODELS + CLOUD_MODELS
FRONTIER = "gpt-oss-120b-cloud"


@dataclass
class CellAgg:
    model: str
    task: str
    category: str
    pass_at_1: float
    pass_pow_8: float
    n: int


def fetch_cells(con: duckdb.DuckDBPyConnection) -> list[CellAgg]:
    """Per (model, task) aggregates: pass@1 (mean over seeds), pass^8 (all-seeds-pass).

    We use `exact_match` for math/knowledge and `regex_match` or `exact_match` for fmt;
    falling back to the deterministic evaluator whose score is recorded.
    """
    sql = f"""
    WITH r AS (
        SELECT
            r.run_id,
            m.litellm_id AS model,
            t.slug AS task,
            r.seed,
            CASE
                WHEN t.slug LIKE 'math-%' THEN 'math-reasoning'
                WHEN t.slug LIKE 'fmt-%'  THEN 'format-following'
                WHEN t.slug LIKE 'know-%' THEN 'knowledge-recall'
                ELSE 'other'
            END AS category
        FROM postgres_scan('{PG}', 'public', 'experiment_runs') r
        JOIN postgres_scan('{PG}', 'public', 'models') m USING (model_id)
        JOIN postgres_scan('{PG}', 'public', 'tasks') t ON t.task_id = r.task_id
        WHERE r.experiment_id = (
            SELECT experiment_id FROM postgres_scan('{PG}', 'public', 'experiments')
            WHERE slug = '{SLUG}'
        )
          AND r.status = 'done'
    ),
    e AS (
        SELECT
            e.run_id,
            COALESCE(
                MAX(CASE WHEN ev.name='exact_match' THEN e.score END),
                MAX(CASE WHEN ev.name='regex_match' THEN e.score END),
                MAX(CASE WHEN ev.name='not_empty'   THEN e.score END)
            ) AS score
        FROM postgres_scan('{PG}', 'public', 'eval_results') e
        JOIN postgres_scan('{PG}', 'public', 'evaluators') ev
          ON ev.evaluator_id = e.evaluator_id
        GROUP BY 1
    ),
    joined AS (
        SELECT r.model, r.task, r.category, r.seed, COALESCE(e.score, 0.0) AS score
        FROM r LEFT JOIN e USING (run_id)
    )
    SELECT
        model, task, category,
        AVG(score) AS pass_at_1,
        AVG(CASE WHEN score >= 1.0 THEN 1.0 ELSE 0.0 END) ^
            (CAST(COUNT(*) AS DOUBLE) / 1.0) AS pass_pow_n_naive,  -- not used
        -- pass^8 = product over seeds of seed_pass, when N=8
        EXP(SUM(LN(GREATEST(score, 1e-10)))) AS pass_pow_8,
        COUNT(*) AS n
    FROM joined
    GROUP BY 1, 2, 3
    ORDER BY 1, 2
    """
    rows = con.execute(sql).fetchall()
    return [
        CellAgg(model=r[0], task=r[1], category=r[2], pass_at_1=r[3], pass_pow_8=r[5], n=r[6])
        for r in rows
    ]


def mean_pass1_by_category(cells: list[CellAgg], model: str, category: str) -> tuple[float, int]:
    sub = [c.pass_at_1 for c in cells if c.model == model and c.category == category]
    if not sub:
        return float("nan"), 0
    return sum(sub) / len(sub), len(sub)


def welch_t(a_vals: list[float], b_vals: list[float]) -> tuple[float, float]:
    """Welch's t-test on two independent samples. Returns (t, p)."""
    if len(a_vals) < 2 or len(b_vals) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
    return float(t), float(p)


def per_task_means(cells: list[CellAgg], model: str, category: str) -> list[float]:
    return [c.pass_at_1 for c in cells if c.model == model and c.category == category]


def reliability_ratio(cells: list[CellAgg], model: str) -> float:
    sub_p1 = [c.pass_at_1 for c in cells if c.model == model]
    sub_p8 = [c.pass_pow_8 for c in cells if c.model == model]
    if not sub_p1 or not sub_p8 or sum(sub_p1) == 0:
        return float("nan")
    return (sum(sub_p8) / len(sub_p8)) / (sum(sub_p1) / len(sub_p1))


def verdict(label: str, value: float, op: str, threshold: float) -> str:
    if value != value:  # NaN
        return f"**{label}**: insufficient data"
    ok = {"≥": value >= threshold, "≤": value <= threshold}[op]
    return f"**{label}**: {'CONFIRMED' if ok else 'REFUTED'} (observed: {value:+.3f}, rule: {op} {threshold:+.2f})"


def main() -> None:
    con = duckdb.connect(":memory:")
    con.execute("INSTALL postgres_scanner; LOAD postgres_scanner;")
    cells = fetch_cells(con)
    if not cells:
        print("ERROR: no cells found for EXP-001")
        sys.exit(2)

    out: list[str] = []
    out.append(f"# EXP-001 verdicts — {len(cells)} cells, computed automatically\n")

    # ----- H1: cloud beats local on math-reasoning by >=15 pp -----
    p1_frontier = mean_pass1_by_category(cells, FRONTIER, "math-reasoning")[0]
    best_local = max(
        (mean_pass1_by_category(cells, L, "math-reasoning")[0] for L in LOCAL_MODELS),
        default=float("nan"),
    )
    h1_delta = p1_frontier - best_local
    f_pertask = per_task_means(cells, FRONTIER, "math-reasoning")
    l_pertask_all = [
        v for L in LOCAL_MODELS for v in per_task_means(cells, L, "math-reasoning")
    ]
    _, h1_p = welch_t(f_pertask, l_pertask_all)
    out.append(f"## H1 — Reasoning gap on math\n")
    out.append(f"- {FRONTIER} mean pass@1 on math-reasoning: **{p1_frontier:.3f}**")
    out.append(f"- best local model mean pass@1 on math-reasoning: **{best_local:.3f}**")
    out.append(f"- delta: **{h1_delta:+.3f}** (rule: ≥ +0.15)")
    out.append(f"- Welch's t-test p-value (frontier vs all-locals per-task means): **{h1_p:.4f}**")
    out.append(verdict("H1", h1_delta, "≥", 0.15) + "\n")

    # ----- H2: knowledge near-parity ≤10pp -----
    p1_frontier_k = mean_pass1_by_category(cells, FRONTIER, "knowledge-recall")[0]
    best_local_k = max(
        (mean_pass1_by_category(cells, L, "knowledge-recall")[0] for L in LOCAL_MODELS),
        default=float("nan"),
    )
    h2_delta = p1_frontier_k - best_local_k
    out.append(f"## H2 — Knowledge near-parity\n")
    out.append(f"- {FRONTIER} mean pass@1 on knowledge-recall: **{p1_frontier_k:.3f}**")
    out.append(f"- best local model mean pass@1 on knowledge-recall: **{best_local_k:.3f}**")
    out.append(f"- delta: **{h2_delta:+.3f}** (rule: ≤ +0.10)")
    out.append(verdict("H2", h2_delta, "≤", 0.10) + "\n")

    # ----- H3: qwen3 reasoning beats gemma3/llama3.1 on fmt by ≥20pp -----
    p1_qwen_f = mean_pass1_by_category(cells, "qwen3-14b-q4", "format-following")[0]
    p1_gemma_f = mean_pass1_by_category(cells, "gemma3-12b-q4", "format-following")[0]
    p1_llama_f = mean_pass1_by_category(cells, "llama3.1-8b-q4", "format-following")[0]
    qwen_pertask = per_task_means(cells, "qwen3-14b-q4", "format-following")
    gemma_pertask = per_task_means(cells, "gemma3-12b-q4", "format-following")
    llama_pertask = per_task_means(cells, "llama3.1-8b-q4", "format-following")
    _, p_v_gemma = welch_t(qwen_pertask, gemma_pertask)
    _, p_v_llama = welch_t(qwen_pertask, llama_pertask)
    out.append(f"## H3 — Reasoning-mode advantage on format-following\n")
    out.append(f"- qwen3-14b-q4 mean pass@1: **{p1_qwen_f:.3f}**")
    out.append(f"- gemma3-12b-q4 mean pass@1: **{p1_gemma_f:.3f}** (delta vs qwen3 = {p1_qwen_f - p1_gemma_f:+.3f}, p={p_v_gemma:.4f})")
    out.append(f"- llama3.1-8b-q4 mean pass@1: **{p1_llama_f:.3f}** (delta vs qwen3 = {p1_qwen_f - p1_llama_f:+.3f}, p={p_v_llama:.4f})")
    h3_min_delta = min(p1_qwen_f - p1_gemma_f, p1_qwen_f - p1_llama_f)
    out.append(verdict("H3", h3_min_delta, "≥", 0.20) + "\n")

    # ----- H4: reliability ratio -----
    rr = {m: reliability_ratio(cells, m) for m in ALL_MODELS}
    out.append(f"## H4 — Reliability cliff\n")
    out.append(f"| model | reliability ratio (mean p^8 / mean p@1) |")
    out.append(f"|---|---|")
    for m in ALL_MODELS:
        out.append(f"| {m} | {rr[m]:.3f} |")
    rr_frontier = rr[FRONTIER]
    min_local_rr = min((rr[L] for L in LOCAL_MODELS if rr[L] == rr[L]), default=float("nan"))
    out.append(f"- {FRONTIER} reliability ratio: **{rr_frontier:.3f}** (rule: ≥ 0.75)")
    out.append(f"- minimum local reliability ratio: **{min_local_rr:.3f}** (rule: ≤ 0.50)")
    h4_part_a = rr_frontier >= 0.75
    h4_part_b = min_local_rr <= 0.50
    h4_ok = h4_part_a and h4_part_b
    out.append(f"**H4**: {'CONFIRMED' if h4_ok else 'REFUTED'} (frontier {'≥0.75 ✓' if h4_part_a else '<0.75 ✗'}, some local {'≤0.50 ✓' if h4_part_b else '>0.50 ✗'})\n")

    # ----- Cell counts -----
    n_by_model = {m: sum(1 for c in cells if c.model == m) for m in ALL_MODELS}
    n_total = sum(n_by_model.values())
    out.append(f"## Sample sizes (must be 24 cells/model for full coverage)\n")
    out.append(f"| model | cells |")
    out.append(f"|---|---|")
    for m, n in n_by_model.items():
        out.append(f"| {m} | {n}/24 |")
    out.append(f"\nTotal cells evaluated: **{n_total}** (target 144 = 6 models × 24 tasks)\n")

    report = "\n".join(out)
    out_path = Path("/data/lab/code/docs/findings/F-003-EXP-001-verdicts.tmp.md")
    out_path.write_text(report)
    print(report)
    print(f"\n--- written to {out_path} ---")


if __name__ == "__main__":
    main()
