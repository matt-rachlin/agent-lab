"""EXP-002 hypothesis verdicts + per-(model, scorer) tables.

Pre-registered four hypotheses (docs/exp/EXP-002.md). This script reads
the per-cell scorer breakdown from agent_logs.turns->'score_breakdown',
joins to experiment_runs / models / tasks, and emits the verdict for each
hypothesis strictly from the pre-registered decision rule. It also writes:

  - analysis/EXP-002/SUMMARY.md (top-line verdicts + headline numbers)
  - analysis/EXP-002/verdicts.md (per-hypothesis verdicts + supporting tables)
  - analysis/EXP-002/per_model_scorer.csv
  - analysis/EXP-002/per_model_task_passes.csv
  - analysis/EXP-002/per_model_task_turns_tools.csv
  - analysis/EXP-002/per_cell_runs.csv
  - analysis/EXP-002/per_model_termination.csv
  - analysis/EXP-002/per_tool_success.csv
  - analysis/EXP-002/trajectory_judge.csv

Tables read (lab DB):
  - experiments
  - experiment_runs
  - models
  - tasks
  - agent_logs (turns JSONB: {turns:[...], score_breakdown:{name:{value,...}}})
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
SLUG = "EXP-002"

LOCAL_MODELS = ["qwen3-14b-q4", "llama3.1-8b-q4"]
CLOUD_MODELS = ["gpt-oss-20b-cloud", "glm-5.1-cloud", "gpt-oss-120b-cloud"]
ALL_MODELS = LOCAL_MODELS + CLOUD_MODELS

# Per-(model, task, seed) cell.
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
    latency_ms: int | None
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None
    actual_turns: int | None
    tool_call_count: int | None
    terminated_reason: str | None
    error: str | None
    turns_payload: list[dict[str, Any]] | None  # full per-turn entries


def fetch_cells() -> list[Cell]:
    sql = """
    SELECT
      m.litellm_id AS model,
      t.slug       AS task,
      r.seed       AS seed,
      r.status     AS status,
      r.tokens_in  AS tokens_in,
      r.tokens_out AS tokens_out,
      r.latency_ms AS latency_ms,
      r.cost_usd   AS cost_usd,
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
        cur.execute(sql, (SLUG,))
        for r in cur.fetchall():
            agent = (r.get("turns") or {}) if isinstance(r.get("turns"), dict) else {}
            scores = agent.get("score_breakdown") or {}

            def _score(name: str, _scores: dict[str, Any] = scores) -> float | None:
                v = (_scores.get(name) or {}).get("value")
                try:
                    return float(v) if v is not None else None
                except (TypeError, ValueError):
                    return None

            turn_entries = agent.get("turns") if isinstance(agent.get("turns"), list) else None
            # Prefer columns from experiment_runs (canonical), fall back to agent_logs payload.
            actual_turns = r.get("actual_turns")
            if actual_turns is None:
                actual_turns = agent.get("actual_turns")
            tool_call_count = r.get("tool_call_count")
            if tool_call_count is None:
                tool_call_count = agent.get("tool_call_count")
            # If experiment_runs.tokens_in/out are empty, sum per-turn tokens from agent_logs.
            tokens_in = r.get("tokens_in")
            tokens_out = r.get("tokens_out")
            if (tokens_in is None or tokens_out is None) and turn_entries:
                ti_sum = sum(int(t.get("tokens_in") or 0) for t in turn_entries)
                to_sum = sum(int(t.get("tokens_out") or 0) for t in turn_entries)
                tokens_in = tokens_in if tokens_in is not None else ti_sum
                tokens_out = tokens_out if tokens_out is not None else to_sum
            # If actual_turns missing from both sources, derive from turn entries.
            if actual_turns is None and turn_entries:
                actual_turns = len(turn_entries)
            # If tool_call_count missing, derive.
            if tool_call_count is None and turn_entries:
                tool_call_count = sum(len(t.get("tools") or []) for t in turn_entries)
            cost = r.get("cost_usd")
            cost_f: float | None = None
            try:
                cost_f = float(cost) if cost is not None else None
            except (TypeError, ValueError):
                cost_f = None
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
                    latency_ms=r.get("latency_ms"),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=cost_f,
                    actual_turns=actual_turns,
                    tool_call_count=tool_call_count,
                    terminated_reason=agent.get("terminated_reason"),
                    error=r.get("error"),
                    turns_payload=turn_entries,
                )
            )
    return out


def welch_t_p(xs: list[float], ys: list[float]) -> float:
    """Two-sided Welch's t-test p-value (no scipy dependency).

    Uses Welch-Satterthwaite df, then a Student-t survival approximation
    via the incomplete-beta identity.
    """
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return float("nan")
    mx = sum(xs) / nx
    my = sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1)
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1)
    if vx == 0.0 and vy == 0.0:
        return 1.0 if mx == my else 0.0
    se = math.sqrt(vx / nx + vy / ny)
    if se == 0.0:
        return 1.0 if mx == my else 0.0
    t = (mx - my) / se
    num = (vx / nx + vy / ny) ** 2
    den = (vx**2) / (nx**2 * (nx - 1)) + (vy**2) / (ny**2 * (ny - 1))
    if den == 0:
        return float("nan")
    df = num / den
    # Student-t two-sided p via regularized incomplete beta:
    #   p = I_{df/(df+t^2)}(df/2, 1/2)
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    return _betainc_reg(x, a, b)


def _betainc_reg(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a,b) via Lentz continued fraction.

    Numerical Recipes §6.4. Accurate enough for reporting p-values.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    # Choose the converging branch.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float) -> float:
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3.0e-7:
            break
    return h


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(
    xs: list[float], n_resamples: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Returns (lo, hi)."""
    if not xs:
        return (float("nan"), float("nan"))
    import random

    rng = random.Random(0)  # deterministic for the report
    n = len(xs)
    means: list[float] = []
    for _ in range(n_resamples):
        s = sum(xs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(n_resamples * (alpha / 2))]
    hi = means[int(n_resamples * (1 - alpha / 2))]
    return (lo, hi)


def per_model_scorer_mean(
    cells: list[Cell], model: str, scorer: str
) -> tuple[float, int, float, float]:
    """Mean + n + bootstrap CI for one (model, scorer) over cells with non-None scores."""
    vals = []
    for c in cells:
        if c.model != model:
            continue
        v = getattr(c, scorer)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return (float("nan"), 0, float("nan"), float("nan"))
    m = mean(vals)
    lo, hi = bootstrap_ci(vals)
    return (m, len(vals), lo, hi)


def per_model_task_pass_at_1(
    cells: list[Cell], model: str, scorer: str
) -> dict[str, float]:
    """pass@1 = mean of scorer across seeds for each task."""
    bucket: dict[str, list[float]] = {}
    for c in cells:
        if c.model != model:
            continue
        v = getattr(c, scorer)
        if v is None:
            continue
        bucket.setdefault(c.task, []).append(float(v))
    return {t: sum(vs) / len(vs) for t, vs in bucket.items() if vs}


def per_model_task_pass_pow_n(
    cells: list[Cell], model: str, scorer: str
) -> dict[str, float]:
    """pass^N = fraction of N-seed cells where ALL seeds scored >= 1.0.

    Per pre-reg: pass^8 = 1.0 iff every one of the 8 seeds scored 1.0.
    """
    bucket: dict[str, list[float]] = {}
    for c in cells:
        if c.model != model:
            continue
        v = getattr(c, scorer)
        if v is None:
            continue
        bucket.setdefault(c.task, []).append(float(v))
    out: dict[str, float] = {}
    for task, vals in bucket.items():
        if not vals:
            continue
        out[task] = 1.0 if all(v >= 1.0 for v in vals) else 0.0
    return out


def reliability_ratio(cells: list[Cell], model: str, scorer: str) -> tuple[float, int]:
    """mean over tasks of pass^8(L,t) / pass^1(L,t), excluding tasks with pass^1==0.

    Returns (ratio, n_tasks_used).
    """
    p1 = per_model_task_pass_at_1(cells, model, scorer)
    p8 = per_model_task_pass_pow_n(cells, model, scorer)
    ratios: list[float] = []
    for task, p1v in p1.items():
        if p1v <= 1e-9:
            continue
        p8v = p8.get(task, 0.0)
        ratios.append(p8v / p1v)
    return (mean(ratios), len(ratios))


# ----------------------------------------------------------------------------
# H4 — per-turn cost / latency ratio
# ----------------------------------------------------------------------------

_MODEL_WEIGHTS: dict[str, float] = {
    "gpt-oss-20b-cloud": 1.0,
    "gpt-oss-120b-cloud": 6.0,
    "glm-5.1-cloud": 3.0,  # rough mid-tier proxy; not in lab.observability.quota._MODEL_WEIGHTS
    "qwen3-14b-q4": 0.5,
    "llama3.1-8b-q4": 0.3,
}


def per_turn_cost_and_latency(cells: list[Cell], model: str) -> tuple[float, float, int]:
    """Return (mean_cost_weight_per_turn, mean_latency_ms_per_turn, n_cells_used).

    Per pre-reg: cost is proxied by `lab.observability.quota._MODEL_WEIGHTS` when metered
    cost is $0. We use per-cell average turn cost = model_weight per turn
    (independent of token volume — it's a flat proxy in pre-reg).
    Latency per turn = total cell latency / actual_turns.
    """
    cost_per_turn: list[float] = []
    lat_per_turn: list[float] = []
    w = _MODEL_WEIGHTS.get(model, 1.0)
    for c in cells:
        if c.model != model or c.status != "done":
            continue
        turns = c.actual_turns or 0
        if turns <= 0 or c.latency_ms is None:
            continue
        cost_per_turn.append(w)  # one model_weight unit per turn
        lat_per_turn.append(c.latency_ms / turns)
    return (mean(cost_per_turn), mean(lat_per_turn), len(cost_per_turn))


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


def fmt_pct(v: float) -> str:
    return f"{v:.3f}" if not math.isnan(v) else "—"


def main() -> None:
    cells = fetch_cells()
    if not cells:
        print("ERROR: no cells found for EXP-002")
        sys.exit(2)

    n_total = len(cells)
    n_done_total = sum(1 for c in cells if c.status == "done")
    n_done = n_done_total
    n_err = sum(1 for c in cells if c.status == "error")
    n_err_total = n_err

    out: list[str] = []
    out.append(f"# EXP-002 verdicts — {n_total} cells ({n_done} done, {n_err} error)\n")

    # ---------- Per-(model, scorer) table ----------
    out.append("## Per-(model, scorer) means with bootstrap 95% CI\n")
    out.append("| model | scorer | mean | n | 95% CI |")
    out.append("|---|---|---|---|---|")
    for m in ALL_MODELS:
        for s in ("end_state", "tool_correctness", "budget_respected", "trajectory_judge"):
            mean_v, n, lo, hi = per_model_scorer_mean(cells, m, s)
            if n == 0:
                continue
            out.append(f"| {m} | {s} | {mean_v:.3f} | {n} | [{lo:.3f}, {hi:.3f}] |")
    out.append("")

    # ---------- pass@1 / pass^8 per (model, task) — end_state ----------
    out.append("## end_state pass@1 / pass^8 per (model, task)\n")
    out.append("| model | task | pass@1 | pass^8 |")
    out.append("|---|---|---|---|")
    for m in ALL_MODELS:
        p1 = per_model_task_pass_at_1(cells, m, "end_state")
        p8 = per_model_task_pass_pow_n(cells, m, "end_state")
        for t in sorted(p1):
            out.append(f"| {m} | {t} | {p1[t]:.3f} | {p8.get(t, 0.0):.3f} |")
    out.append("")

    # ---------- H1 ----------
    h1_vals: list[float] = []
    for c in cells:
        if c.model in CLOUD_MODELS and c.tool_correctness is not None:
            h1_vals.append(float(c.tool_correctness))
    h1_mean = mean(h1_vals)
    h1_lo, h1_hi = bootstrap_ci(h1_vals)
    h1_pass = h1_mean >= 0.60
    out.append("## H1 — Cloud tool-call accuracy ≥ 0.60\n")
    out.append(f"- cells used: {len(h1_vals)} (cloud × tasks × seeds)")
    out.append(f"- mean tool_correctness: **{h1_mean:.3f}** (95% CI [{h1_lo:.3f}, {h1_hi:.3f}])")
    out.append("- rule: ≥ 0.60")
    out.append(f"- **H1: {'CONFIRMED' if h1_pass else 'REFUTED'}**\n")

    # ---------- H2 ----------
    h2_vals: list[float] = []
    for c in cells:
        if c.model in LOCAL_MODELS and c.tool_correctness is not None:
            h2_vals.append(float(c.tool_correctness))
    h2_mean = mean(h2_vals)
    h2_lo, h2_hi = bootstrap_ci(h2_vals)
    h2_pass = h2_mean >= 0.40
    out.append("## H2 — Local tool-call accuracy ≥ 0.40\n")
    out.append(f"- cells used: {len(h2_vals)} (local × tasks × seeds)")
    out.append(f"- mean tool_correctness: **{h2_mean:.3f}** (95% CI [{h2_lo:.3f}, {h2_hi:.3f}])")
    out.append("- rule: ≥ 0.40")
    out.append(f"- **H2: {'CONFIRMED' if h2_pass else 'REFUTED'}**\n")

    # ---------- H3 ----------
    out.append("## H3 — Multi-turn reliability cliff (∃ local L with mean pass^8/pass^1 < 0.70 on end_state)\n")
    out.append("| local model | reliability_ratio | n_tasks_with_p1>0 | verdict |")
    out.append("|---|---|---|---|")
    h3_confirmed = False
    h3_undefined_all = True
    for L in LOCAL_MODELS:
        rr, n_tasks = reliability_ratio(cells, L, "end_state")
        if n_tasks < 6:
            verdict = "undefined (n_tasks<6)"
        elif rr < 0.70:
            verdict = "< 0.70 ✓ (cliff)"
            h3_confirmed = True
            h3_undefined_all = False
        else:
            verdict = "≥ 0.70 ✗"
            h3_undefined_all = False
        out.append(f"| {L} | {fmt_pct(rr)} | {n_tasks} | {verdict} |")
    if h3_undefined_all:
        h3_status = "UNDEFINED"
    elif h3_confirmed:
        h3_status = "CONFIRMED"
    else:
        h3_status = "REFUTED"
    out.append(f"\n- **H3: {h3_status}**\n")

    # ---------- H4 ----------
    out.append("## H4 — cost/turn ratio ≥ 1.5 × latency/turn ratio (gpt-oss-120b vs gpt-oss-20b)\n")
    c20, l20, n20 = per_turn_cost_and_latency(cells, "gpt-oss-20b-cloud")
    c120, l120, n120 = per_turn_cost_and_latency(cells, "gpt-oss-120b-cloud")
    cost_ratio = (c120 / c20) if (c20 and not math.isnan(c20) and c20 != 0) else float("nan")
    lat_ratio = (l120 / l20) if (l20 and not math.isnan(l20) and l20 != 0) else float("nan")
    rule_rhs = 1.5 * lat_ratio if not math.isnan(lat_ratio) else float("nan")
    h4_pass = (
        not math.isnan(cost_ratio)
        and not math.isnan(rule_rhs)
        and cost_ratio >= rule_rhs
    )
    out.append(f"- gpt-oss-20b-cloud: cost/turn weight = {c20:.3f}, latency/turn = {l20:.1f} ms (n={n20})")
    out.append(f"- gpt-oss-120b-cloud: cost/turn weight = {c120:.3f}, latency/turn = {l120:.1f} ms (n={n120})")
    out.append(f"- cost_ratio = {fmt_pct(cost_ratio)}, latency_ratio = {fmt_pct(lat_ratio)}, 1.5×latency_ratio = {fmt_pct(rule_rhs)}")
    out.append("- rule: cost_ratio ≥ 1.5 × latency_ratio")
    out.append(f"- **H4: {'CONFIRMED' if h4_pass else 'REFUTED'}**\n")

    # ---------- Per-tool success rate ----------
    out.append("## Per-tool success rate (across all cells)\n")
    tool_attempts: dict[str, int] = {}
    tool_errors: dict[str, int] = {}
    for c in cells:
        if not c.turns_payload:
            continue
        for t in c.turns_payload:
            for tc in (t.get("tools") or []):
                name = tc.get("tool") or "<unknown>"
                tool_attempts[name] = tool_attempts.get(name, 0) + 1
                if tc.get("error"):
                    tool_errors[name] = tool_errors.get(name, 0) + 1
    if tool_attempts:
        out.append("| tool | attempts | errors | success rate |")
        out.append("|---|---|---|---|")
        for name in sorted(tool_attempts):
            a = tool_attempts[name]
            e = tool_errors.get(name, 0)
            rate = (a - e) / a if a else float("nan")
            out.append(f"| {name} | {a} | {e} | {rate:.3f} |")
    else:
        out.append("(no tool calls recorded)")
    out.append("")

    # ---------- Per-model trajectory patterns ----------
    out.append("## Per-model trajectory patterns\n")
    out.append("| model | done | error | budget_exhausted | max_turns_reached | litellm_error | model_finished | over-budget | hallucinated tools | never invoked |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in ALL_MODELS:
        reasons: dict[str, int] = {}
        done = 0
        err = 0
        over_budget = 0
        halluc = 0
        never_invoked = 0
        for c in cells:
            if c.model != m:
                continue
            if c.status == "done":
                done += 1
            else:
                err += 1
            r = c.terminated_reason or ""
            if r:
                reasons[r] = reasons.get(r, 0) + 1
            # over-budget: budget_respected score == 0
            if c.budget_respected is not None and c.budget_respected < 1.0:
                over_budget += 1
            # never-invoked: zero tool calls but turns happened
            if (c.tool_call_count == 0) and (c.actual_turns or 0) > 0:
                never_invoked += 1
            # hallucinated tools: turn-level "unknown tool" errors
            for t in (c.turns_payload or []):
                for tc in (t.get("tools") or []):
                    if tc.get("error") and "unknown tool" in str(tc.get("error", "")):
                        halluc += 1
        out.append(
            f"| {m} | {done} | {err} | {reasons.get('budget_exhausted',0)} | {reasons.get('max_turns_reached',0)} | "
            f"{reasons.get('litellm_error',0)} | {reasons.get('model_finished',0)} | {over_budget} | {halluc} | {never_invoked} |"
        )
    out.append("")

    # ---------- Cell-count coverage ----------
    out.append("## Cell coverage (expected 96/model = 12 tasks × 8 seeds)\n")
    out.append("| model | done | error | total |")
    out.append("|---|---|---|---|")
    for m in ALL_MODELS:
        nd = sum(1 for c in cells if c.model == m and c.status == "done")
        ne = sum(1 for c in cells if c.model == m and c.status == "error")
        out.append(f"| {m} | {nd} | {ne} | {nd+ne}/96 |")
    out.append("")

    # ---------- NOANSWER / no-data scorer rate ----------
    out.append("## Scorer coverage (rows with non-null value)\n")
    out.append("| scorer | non-null cells | null/missing |")
    out.append("|---|---|---|")
    for s in ("end_state", "tool_correctness", "budget_respected", "trajectory_judge"):
        n_nn = sum(1 for c in cells if getattr(c, s) is not None)
        n_null = n_total - n_nn
        out.append(f"| {s} | {n_nn} | {n_null} |")
    out.append("")

    report = "\n".join(out)
    tmp_path = Path("/data/lab/code/docs/findings/F-005-EXP-002-verdicts.tmp.md")
    tmp_path.write_text(report)

    analysis_dir = Path("/data/lab/code/analysis/EXP-002")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "verdicts.md").write_text(report)

    # ---------- CSV: per_cell_runs.csv ----------
    per_cell = analysis_dir / "per_cell_runs.csv"
    with per_cell.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "model", "task", "seed", "status",
            "end_state", "tool_correctness", "budget_respected", "trajectory_judge",
            "latency_ms", "tokens_in", "tokens_out", "cost_usd",
            "actual_turns", "tool_call_count", "terminated_reason", "error",
        ])
        for c in cells:
            w.writerow([
                c.model, c.task, c.seed, c.status,
                c.end_state, c.tool_correctness, c.budget_respected, c.trajectory_judge,
                c.latency_ms, c.tokens_in, c.tokens_out, c.cost_usd,
                c.actual_turns, c.tool_call_count, c.terminated_reason,
                (c.error or "")[:200],
            ])

    # ---------- CSV: per_model_scorer.csv ----------
    with (analysis_dir / "per_model_scorer.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "scorer", "n", "mean", "ci_lo", "ci_hi"])
        for m in ALL_MODELS:
            for s in ("end_state", "tool_correctness", "budget_respected", "trajectory_judge"):
                mean_v, n, lo, hi = per_model_scorer_mean(cells, m, s)
                if n == 0:
                    continue
                w.writerow([m, s, n, f"{mean_v:.6f}", f"{lo:.6f}", f"{hi:.6f}"])

    # ---------- CSV: per_model_task_passes.csv ----------
    with (analysis_dir / "per_model_task_passes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "task", "scorer", "pass_at_1", "pass_pow_8"])
        for m in ALL_MODELS:
            for s in ("end_state", "tool_correctness", "budget_respected"):
                p1 = per_model_task_pass_at_1(cells, m, s)
                p8 = per_model_task_pass_pow_n(cells, m, s)
                for t in sorted(p1):
                    w.writerow([m, t, s, f"{p1[t]:.6f}", f"{p8.get(t, 0.0):.6f}"])

    # ---------- CSV: per_model_task_turns_tools.csv ----------
    with (analysis_dir / "per_model_task_turns_tools.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "model", "task", "n_cells",
            "mean_turns", "mean_tool_calls",
            "mean_latency_ms", "mean_tokens_in", "mean_tokens_out", "mean_cost_usd",
        ])
        # group by (model, task)
        groups: dict[tuple[str, str], list[Cell]] = {}
        for c in cells:
            groups.setdefault((c.model, c.task), []).append(c)
        for (m, t), cs in sorted(groups.items()):
            done = [c for c in cs if c.status == "done"]
            n = len(done)
            if n == 0:
                w.writerow([m, t, 0, "", "", "", "", "", ""])
                continue
            turns = [c.actual_turns for c in done if c.actual_turns is not None]
            tcs = [c.tool_call_count for c in done if c.tool_call_count is not None]
            lat = [c.latency_ms for c in done if c.latency_ms is not None]
            ti = [c.tokens_in for c in done if c.tokens_in is not None]
            to_ = [c.tokens_out for c in done if c.tokens_out is not None]
            costs = [c.cost_usd for c in done if c.cost_usd is not None]
            w.writerow([
                m, t, n,
                f"{mean([float(x) for x in turns]):.3f}" if turns else "",
                f"{mean([float(x) for x in tcs]):.3f}" if tcs else "",
                f"{mean([float(x) for x in lat]):.1f}" if lat else "",
                f"{mean([float(x) for x in ti]):.1f}" if ti else "",
                f"{mean([float(x) for x in to_]):.1f}" if to_ else "",
                f"{mean(costs):.6f}" if costs else "",
            ])

    # ---------- CSV: per_model_termination.csv ----------
    with (analysis_dir / "per_model_termination.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "model", "n_done", "n_error",
            "model_finished", "budget_exhausted", "max_turns_reached",
            "litellm_error", "other_unknown",
        ])
        for m in ALL_MODELS:
            n_done = sum(1 for c in cells if c.model == m and c.status == "done")
            n_err = sum(1 for c in cells if c.model == m and c.status == "error")
            counts: dict[str, int] = {}
            for c in cells:
                if c.model != m:
                    continue
                r = c.terminated_reason or "unknown"
                counts[r] = counts.get(r, 0) + 1
            other_unknown = sum(
                v for k, v in counts.items()
                if k not in {"model_finished", "budget_exhausted", "max_turns_reached", "litellm_error"}
            )
            w.writerow([
                m, n_done, n_err,
                counts.get("model_finished", 0),
                counts.get("budget_exhausted", 0),
                counts.get("max_turns_reached", 0),
                counts.get("litellm_error", 0),
                other_unknown,
            ])

    # ---------- CSV: per_tool_success.csv (overall + per model) ----------
    with (analysis_dir / "per_tool_success.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "tool", "attempts", "errors", "success_rate"])
        # overall
        overall_a: dict[str, int] = {}
        overall_e: dict[str, int] = {}
        per_model_tool_attempts: dict[str, dict[str, int]] = {}
        per_model_tool_errors: dict[str, dict[str, int]] = {}
        for c in cells:
            if not c.turns_payload:
                continue
            per_model_tool_attempts.setdefault(c.model, {})
            per_model_tool_errors.setdefault(c.model, {})
            for tn in c.turns_payload:
                for tc in (tn.get("tools") or []):
                    name = tc.get("tool") or "<unknown>"
                    overall_a[name] = overall_a.get(name, 0) + 1
                    per_model_tool_attempts[c.model][name] = (
                        per_model_tool_attempts[c.model].get(name, 0) + 1
                    )
                    if tc.get("error"):
                        overall_e[name] = overall_e.get(name, 0) + 1
                        per_model_tool_errors[c.model][name] = (
                            per_model_tool_errors[c.model].get(name, 0) + 1
                        )
        for name in sorted(overall_a):
            a = overall_a[name]
            e = overall_e.get(name, 0)
            w.writerow(["<all>", name, a, e, f"{(a-e)/a:.6f}" if a else ""])
        for m in ALL_MODELS:
            attempts = per_model_tool_attempts.get(m, {})
            errs = per_model_tool_errors.get(m, {})
            for name in sorted(attempts):
                a = attempts[name]
                e = errs.get(name, 0)
                w.writerow([m, name, a, e, f"{(a-e)/a:.6f}" if a else ""])

    # ---------- CSV: trajectory_judge.csv (only judge-enabled task) ----------
    with (analysis_dir / "trajectory_judge.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "task", "n_cells", "mean_judge", "ci_lo", "ci_hi"])
        # group cells with non-null trajectory_judge by (model, task)
        groups2: dict[tuple[str, str], list[float]] = {}
        for c in cells:
            if c.trajectory_judge is None:
                continue
            groups2.setdefault((c.model, c.task), []).append(float(c.trajectory_judge))
        for (m, t), vs in sorted(groups2.items()):
            lo, hi = bootstrap_ci(vs)
            w.writerow([m, t, len(vs), f"{mean(vs):.6f}", f"{lo:.6f}", f"{hi:.6f}"])

    # ---------- Welch p-values for hypothesis context ----------
    cloud_tool = [
        float(c.tool_correctness) for c in cells
        if c.model in CLOUD_MODELS and c.tool_correctness is not None
    ]
    local_tool = [
        float(c.tool_correctness) for c in cells
        if c.model in LOCAL_MODELS and c.tool_correctness is not None
    ]
    welch_cloud_vs_local = welch_t_p(cloud_tool, local_tool)
    cloud_mean = mean(cloud_tool)
    local_mean = mean(local_tool)

    # ---------- SUMMARY.md ----------
    summary: list[str] = []
    summary.append("# EXP-002 Summary — 12 GB Agent v0.2, first tool-use characterization\n")
    summary.append(f"- Cells: {n_total} ({n_done_total} done, {n_err_total} error)")
    summary.append("- Models: 5 (2 local, 3 cloud) | Tasks: 12 (PBS-Agent v0.1) | Seeds: 8 | Config: greedy-1024")
    summary.append("")
    summary.append("## Per-hypothesis verdicts\n")
    summary.append("| H | Rule | Value | Verdict |")
    summary.append("|---|---|---|---|")
    h1_pass = cloud_mean >= 0.60
    h2_pass = local_mean >= 0.40
    summary.append(
        f"| H1 | cloud mean tool_correctness ≥ 0.60 | **{cloud_mean:.3f}** | "
        f"**{'CONFIRMED' if h1_pass else 'REFUTED'}** |"
    )
    summary.append(
        f"| H2 | local mean tool_correctness ≥ 0.40 | **{local_mean:.3f}** | "
        f"**{'CONFIRMED' if h2_pass else 'REFUTED'}** |"
    )
    # H3 verdict
    h3_label = "UNDEFINED"
    h3_lines: list[str] = []
    for L in LOCAL_MODELS:
        rr, n_tasks = reliability_ratio(cells, L, "end_state")
        h3_lines.append(f"{L}: ratio={rr:.3f}, n_tasks={n_tasks}")
        if n_tasks >= 6:
            if rr < 0.70:
                h3_label = "CONFIRMED"
            elif h3_label == "UNDEFINED":
                h3_label = "REFUTED"
    summary.append(
        f"| H3 | ∃ local L: mean(pass⁸/pass¹) < 0.70 on end_state | "
        f"{'; '.join(h3_lines)} | **{h3_label}** |"
    )
    c20, l20, n20 = per_turn_cost_and_latency(cells, "gpt-oss-20b-cloud")
    c120, l120, n120 = per_turn_cost_and_latency(cells, "gpt-oss-120b-cloud")
    cost_ratio = (c120 / c20) if (c20 and not math.isnan(c20) and c20 != 0) else float("nan")
    lat_ratio = (l120 / l20) if (l20 and not math.isnan(l20) and l20 != 0) else float("nan")
    rule_rhs = 1.5 * lat_ratio if not math.isnan(lat_ratio) else float("nan")
    h4_pass = (
        not math.isnan(cost_ratio)
        and not math.isnan(rule_rhs)
        and cost_ratio >= rule_rhs
    )
    summary.append(
        f"| H4 | cost_ratio(120b/20b) ≥ 1.5×latency_ratio | "
        f"cost={cost_ratio:.3f}, 1.5×lat={rule_rhs:.3f} | **{'CONFIRMED' if h4_pass else 'REFUTED'}** |"
    )
    summary.append("")
    summary.append("## Headline numbers — per-model means (deterministic scorers)\n")
    summary.append("| model | end_state | tool_correctness | budget_respected | mean_turns | mean_tool_calls |")
    summary.append("|---|---|---|---|---|---|")
    for m in ALL_MODELS:
        es_m, _, _, _ = per_model_scorer_mean(cells, m, "end_state")
        tc_m, _, _, _ = per_model_scorer_mean(cells, m, "tool_correctness")
        br_m, _, _, _ = per_model_scorer_mean(cells, m, "budget_respected")
        turns = [
            float(c.actual_turns) for c in cells
            if c.model == m and c.status == "done" and c.actual_turns is not None
        ]
        tcs = [
            float(c.tool_call_count) for c in cells
            if c.model == m and c.status == "done" and c.tool_call_count is not None
        ]
        summary.append(
            f"| {m} | {es_m:.3f} | {tc_m:.3f} | {br_m:.3f} | "
            f"{mean(turns):.2f} | {mean(tcs):.2f} |"
        )
    summary.append("")
    summary.append("## Auxiliary stats\n")
    summary.append(
        f"- Cloud vs local mean tool_correctness gap: "
        f"{cloud_mean - local_mean:+.3f} (Welch p={welch_cloud_vs_local:.3f}, "
        f"n_cloud={len(cloud_tool)}, n_local={len(local_tool)})"
    )
    # Scorer non-null coverage
    summary.append("")
    summary.append("## Scorer coverage\n")
    summary.append("| scorer | non-null cells | null/missing | pct_non_null |")
    summary.append("|---|---|---|---|")
    for s in ("end_state", "tool_correctness", "budget_respected", "trajectory_judge"):
        n_nn = sum(1 for c in cells if getattr(c, s) is not None)
        pct = (n_nn / n_total) * 100 if n_total else 0.0
        summary.append(f"| {s} | {n_nn} | {n_total - n_nn} | {pct:.1f}% |")
    summary.append("")
    summary.append("Generated by `scripts/analyze_exp002.py`. Detail in `verdicts.md` and CSVs alongside.")

    (analysis_dir / "SUMMARY.md").write_text("\n".join(summary) + "\n")

    print(report)
    print(f"\n--- detail written to {tmp_path}")
    print(f"--- analysis dir: {analysis_dir} ---")


if __name__ == "__main__":
    main()
