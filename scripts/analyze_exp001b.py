"""EXP-001b hypothesis verdicts.

Compares two new qwen3-14b-q4 configs against the EXP-001 baseline:
  EXP-001 baseline = qwen3-14b-q4, reasoning on (default), max_tokens=1024
  B.1 = reasoning on,  max_tokens=2048
  B.2 = reasoning off (think=false), max_tokens=1024

Configs are discriminated by config_hash. Hashes are read from the sweep YAMLs
to avoid drift.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
from scipy import stats

from lab.sweep.config import config_hash, load_sweep

PG = "dbname=lab host=/var/run/postgresql"
GEMMA_FMT_BASELINE = 0.875  # F-003 EXP-001
REPO = Path("/data/lab/code")


@dataclass
class Cell:
    label: str
    task: str
    category: str
    pass_at_1: float
    empty_count: int
    total_count: int


def discover_hashes() -> dict[str, str]:
    """Return label → config_hash for the three configs we compare."""
    spec_b = load_sweep(REPO / "conf/sweep/EXP-001b.yaml")
    spec_a = load_sweep(REPO / "conf/sweep/EXP-001.yaml")
    out: dict[str, str] = {}
    for c in spec_b.configs:
        if c.name == "greedy-2048-thinking":
            out["B.1"] = config_hash(c)
        elif c.name == "greedy-1024-no-think":
            out["B.2"] = config_hash(c)
    for c in spec_a.configs:
        if c.name == "greedy-1024":
            out["baseline"] = config_hash(c)
    return out


def fetch_cells(con: duckdb.DuckDBPyConnection, hashes: dict[str, str]) -> list[Cell]:
    labels_sql = ", ".join(f"('{lbl}','{h}')" for lbl, h in hashes.items())
    sql = f"""
    WITH labelled AS (
        SELECT v.label, v.cfg_hash
        FROM (VALUES {labels_sql}) AS v(label, cfg_hash)
    ),
    r AS (
        SELECT
            r.run_id,
            l.label,
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
        JOIN labelled l ON l.cfg_hash = r.config_hash
        WHERE r.status = 'done' AND m.litellm_id = 'qwen3-14b-q4'
    ),
    e AS (
        SELECT
            e.run_id,
            COALESCE(
                MAX(CASE WHEN ev.name='exact_match' THEN e.score END),
                MAX(CASE WHEN ev.name='regex_match' THEN e.score END),
                MAX(CASE WHEN ev.name='not_empty'   THEN e.score END)
            ) AS score,
            MAX(CASE WHEN ev.name='not_empty' THEN e.score END) AS not_empty_score
        FROM postgres_scan('{PG}', 'public', 'eval_results') e
        JOIN postgres_scan('{PG}', 'public', 'evaluators') ev ON ev.evaluator_id = e.evaluator_id
        GROUP BY 1
    )
    SELECT
        r.label, r.task, r.category,
        AVG(COALESCE(e.score, 0.0)) AS pass_at_1,
        SUM(CASE WHEN COALESCE(e.not_empty_score, 0.0) < 1.0 THEN 1 ELSE 0 END) AS empty_count,
        COUNT(*) AS total
    FROM r LEFT JOIN e USING (run_id)
    GROUP BY 1, 2, 3
    ORDER BY 1, 2
    """
    rows = con.execute(sql).fetchall()
    return [
        Cell(
            label=r[0],
            task=r[1],
            category=r[2],
            pass_at_1=r[3],
            empty_count=int(r[4]),
            total_count=int(r[5]),
        )
        for r in rows
    ]


def by_label(cells: list[Cell], label: str) -> list[Cell]:
    return [c for c in cells if c.label == label]


def empty_rate(cells: list[Cell]) -> float:
    e = sum(c.empty_count for c in cells)
    t = sum(c.total_count for c in cells)
    return e / t if t else float("nan")


def cat_pass1(cells: list[Cell], cat: str) -> tuple[float, list[float]]:
    sub = [c for c in cells if c.category == cat]
    means = [c.pass_at_1 for c in sub]
    return (sum(means) / len(means) if means else float("nan"), means)


def welch_p(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    _, p = stats.ttest_ind(a, b, equal_var=False)
    return float(p)


def verdict(label: str, *parts: tuple[str, bool]) -> str:
    ok = all(p[1] for p in parts)
    detail = " AND ".join(f"{p[0]} {'✓' if p[1] else '✗'}" for p in parts)
    return f"**{label}**: {'CONFIRMED' if ok else 'REFUTED'} ({detail})"


def main() -> None:
    hashes = discover_hashes()
    print(f"config hashes: {hashes}\n")
    con = duckdb.connect(":memory:")
    con.execute("INSTALL postgres_scanner; LOAD postgres_scanner;")
    cells = fetch_cells(con, hashes)
    if not cells:
        print("ERROR: no cells")
        sys.exit(2)

    baseline = by_label(cells, "baseline")
    b1 = by_label(cells, "B.1")
    b2 = by_label(cells, "B.2")
    print(f"cell counts: baseline={len(baseline)}  B.1={len(b1)}  B.2={len(b2)}\n")

    out: list[str] = ["# EXP-001b verdicts\n"]
    out.append("## Per-config metrics\n")
    out.append("| config | (model,task) cells | empty rate | fmt p@1 | math p@1 | know p@1 |")
    out.append("|---|---|---|---|---|---|")
    for label, sub in [
        ("baseline (EXP-001: reasoning on, 1024)", baseline),
        ("B.1 (reasoning on, 2048)", b1),
        ("B.2 (think=false, 1024)", b2),
    ]:
        er = empty_rate(sub)
        f_m, _ = cat_pass1(sub, "format-following")
        m_m, _ = cat_pass1(sub, "math-reasoning")
        k_m, _ = cat_pass1(sub, "knowledge-recall")
        out.append(f"| {label} | {len(sub)} | {er:.3f} | {f_m:.3f} | {m_m:.3f} | {k_m:.3f} |")

    # H1
    b1_er = empty_rate(b1)
    b1_fmt, _ = cat_pass1(b1, "format-following")
    bl_fmt, _ = cat_pass1(baseline, "format-following")
    out.append("\n## H1 — budget alone fixes it (B.1 vs baseline)\n")
    out.append(f"- B.1 empty_rate = **{b1_er:.3f}** (rule: ≤0.10)")
    out.append(
        f"- B.1 fmt p@1 = **{b1_fmt:.3f}**, baseline fmt p@1 = **{bl_fmt:.3f}**, delta = **{b1_fmt - bl_fmt:+.3f}** (rule: ≥+0.20)"
    )
    out.append(
        verdict("H1", ("empty≤0.10", b1_er <= 0.10), ("Δfmt≥+0.20", (b1_fmt - bl_fmt) >= 0.20))
    )

    # H2
    b2_er = empty_rate(b2)
    b2_fmt, _ = cat_pass1(b2, "format-following")
    out.append("\n## H2 — disabling reasoning alone fixes it (B.2 vs gemma3 0.875)\n")
    out.append(f"- B.2 empty_rate = **{b2_er:.3f}** (rule: ≤0.05)")
    out.append(
        f"- B.2 fmt p@1 = **{b2_fmt:.3f}**, gemma3 (EXP-001) = **{GEMMA_FMT_BASELINE}**, |delta| = **{abs(b2_fmt - GEMMA_FMT_BASELINE):.3f}** (rule: ≤0.05)"
    )
    out.append(
        verdict(
            "H2",
            ("empty≤0.05", b2_er <= 0.05),
            ("|Δfmt vs gemma3|≤0.05", abs(b2_fmt - GEMMA_FMT_BASELINE) <= 0.05),
        )
    )

    # H3
    b2_math, b2_pt = cat_pass1(b2, "math-reasoning")
    bl_math, bl_pt = cat_pass1(baseline, "math-reasoning")
    drop = bl_math - b2_math
    p = welch_p(bl_pt, b2_pt)
    out.append("\n## H3 — reasoning earns its keep on math (B.2 < baseline by ≥10pp)\n")
    out.append(
        f"- baseline math p@1 = **{bl_math:.3f}**, B.2 math p@1 = **{b2_math:.3f}**, drop = **{drop:+.3f}** (rule: ≥+0.10)"
    )
    out.append(f"- Welch's p (baseline vs B.2 per-task means) = **{p:.4f}**")
    out.append(verdict("H3", ("drop≥+0.10", drop >= 0.10)))

    report = "\n".join(out)
    print(report)


if __name__ == "__main__":
    main()
