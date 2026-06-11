"""EXP-011 — HARD-BENCH-CLOUD-001: frontier cloud anchor on the hard suite.

Three pre-registered hypotheses (docs/exp/EXP-011-hard-bench-cloud-anchor.md):

  H1 (frontier ceiling):  max cloud pass@1 >= 0.938 CONFIRMED;
                          < 0.906 REFUTED; between INCONCLUSIVE.
  H2 (within-family scale): qwen3-coder-480b >= 0.874 CONFIRMED;
                          <= 0.812 REFUTED; between INCONCLUSIVE.
  H3 (failure-mode absence): zero text-emitted-call and zero narration
                          episodes across all cloud trajectories; any
                          single episode REFUTES.

H3 scans every cell's trace.jsonl from MinIO (trace_path on the run row):
- narration episode  = zero structured tool calls across the episode
  (also visible in agent_logs turn records as tool_calls_requested == 0
  on every turn — both signals are checked and reported).
- text-emitted calls = an assistant message with no structured tool_calls
  whose content matches a JSON object carrying "name" + "arguments"/
  "parameters" against a known tool name (the F-012 pattern).

Writes analysis/EXP-011/{SUMMARY.md,per_cell.csv,h3_flags.csv}.

Cost: free (DB read + MinIO reads).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from lab.core.minio_io import make_minio_client

PG_DSN = "dbname=lab host=/var/run/postgresql"
DEFAULT_SLUG = "HARD-BENCH-CLOUD-001"
EXP_DIR_NAME = "EXP-011"

MODEL_GLM = "glm-5.1-cloud"
MODEL_Q480 = "qwen3-coder-480b-cloud"

# Pre-registered thresholds (do NOT change after sweep starts).
H1_CONFIRM = 0.938
H1_REFUTE_BELOW = 0.906
H2_CONFIRM = 0.874
H2_REFUTE_AT = 0.812

# Local comparators, HARD-BENCH-002 seed-1 (not re-run here).
LOCAL_BASELINES = {
    "gemma4-12b": 0.938,
    "qwen3-coder-30b": 0.812,
    "devstral-24b": 0.531,
}

TOOL_NAMES = {
    "fs_read",
    "fs_write",
    "fs_grep",
    "shell_exec",
    "python_eval",
    "http_fetch",
    "kb_query",
}
JSON_OBJ_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}")


def looks_like_text_tool_call(content: str) -> bool:
    for m in JSON_OBJ_RE.finditer(content):
        try:
            obj = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        if (
            isinstance(obj, dict)
            and obj.get("name") in TOOL_NAMES
            and ("arguments" in obj or "parameters" in obj)
        ):
            return True
    return False


def scan_trace(blob: bytes) -> tuple[int, int]:
    """Return (n_structured_tool_calls, n_text_emitted_call_turns).

    Trajectory JSONL is typed records: header / messages (initial prompt) /
    turn / footer. Turn records carry tool_calls_requested and a
    content_preview (truncated for long replies — the F-012 text-call
    pattern appears at the start of content, so the preview suffices).
    """
    structured = 0
    text_emitted = 0
    for line in blob.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "turn":
            continue
        requested = int(rec.get("tool_calls_requested") or 0)
        structured += requested
        preview = rec.get("content_preview")
        if (
            requested == 0
            and isinstance(preview, str)
            and preview.strip()
            and looks_like_text_tool_call(preview)
        ):
            text_emitted += 1
    return structured, text_emitted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument("--out", default=f"analysis/{EXP_DIR_NAME}")
    ap.add_argument(
        "--skip-traces", action="store_true", help="skip the MinIO H3 scan (DB-only signals)"
    )
    args = ap.parse_args()

    sql = """
        select m.litellm_id as model, t.slug as task,
               coalesce(t.category, '?') as category, er.seed,
               er.trace_path,
               (al.turns->'score_breakdown'->'end_state'->>'value')::float
                   as score,
               (
                 select coalesce(
                   sum((tu->>'tool_calls_requested')::int), 0)
                 from jsonb_array_elements(al.turns->'turns') tu
               ) as calls_requested
        from experiment_runs er
        join experiments e on e.experiment_id = er.experiment_id
        join models m on m.model_id = er.model_id
        join tasks t on t.task_id = er.task_id
        left join agent_logs al on al.run_id = er.run_id
        where e.slug = %s
        order by m.litellm_id, t.slug
    """
    with psycopg.connect(PG_DSN, row_factory=dict_row) as conn:
        rows = [r for r in conn.execute(sql, (args.slug,)) if r["score"] is not None]
    if not rows:
        raise SystemExit(f"no scored cells for {args.slug}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted({str(r["model"]) for r in rows})
    overall: dict[str, float] = {}
    per_cat: dict[str, dict[str, float]] = defaultdict(dict)
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        overall[m] = sum(float(r["score"]) for r in mine) / len(mine)  # type: ignore[arg-type]
        for cat in sorted({str(r["category"]) for r in mine}):
            cc = [r for r in mine if r["category"] == cat]
            per_cat[m][cat] = sum(float(r["score"]) for r in cc) / len(cc)  # type: ignore[arg-type]

    # --- H3 trajectory scan -------------------------------------------------
    flags: list[dict[str, object]] = []
    narration_total = 0
    text_emit_total = 0
    if not args.skip_traces:
        client = make_minio_client()
        for r in rows:
            tp = str(r["trace_path"] or "")
            if not tp.startswith("s3://"):
                continue
            bucket, key = tp.removeprefix("s3://").split("/", 1)
            try:
                resp = client.get_object(bucket, key)
                blob = resp.read()
                resp.close()
                resp.release_conn()
            except Exception as exc:
                flags.append(
                    {"model": r["model"], "task": r["task"], "flag": f"trace-fetch-failed: {exc}"}
                )
                continue
            structured, text_emitted = scan_trace(blob)
            narration = structured == 0
            if narration:
                narration_total += 1
                flags.append(
                    {
                        "model": r["model"],
                        "task": r["task"],
                        "flag": "narration (0 structured tool calls)",
                    }
                )
            if text_emitted:
                text_emit_total += text_emitted
                flags.append(
                    {
                        "model": r["model"],
                        "task": r["task"],
                        "flag": f"text-emitted calls x{text_emitted}",
                    }
                )
            db_zero = int(r["calls_requested"] or 0) == 0  # type: ignore[arg-type]
            if db_zero != narration:
                flags.append(
                    {
                        "model": r["model"],
                        "task": r["task"],
                        "flag": "signal-mismatch (db vs trace)",
                    }
                )

    # --- verdicts -----------------------------------------------------------
    verdicts: dict[str, str] = {}
    best = max(overall.values())
    verdicts["H1"] = (
        f"CONFIRMED (best cloud {best:.3f} >= {H1_CONFIRM})"
        if best >= H1_CONFIRM
        else f"REFUTED (best cloud {best:.3f} < {H1_REFUTE_BELOW})"
        if best < H1_REFUTE_BELOW
        else f"INCONCLUSIVE (best cloud {best:.3f})"
    )
    if MODEL_Q480 in overall:
        q = overall[MODEL_Q480]
        verdicts["H2"] = (
            f"CONFIRMED ({q:.3f} >= {H2_CONFIRM})"
            if q >= H2_CONFIRM
            else f"REFUTED ({q:.3f} <= {H2_REFUTE_AT})"
            if q <= H2_REFUTE_AT
            else f"INCONCLUSIVE ({q:.3f})"
        )
    verdicts["H3"] = (
        "SKIPPED (--skip-traces)"
        if args.skip_traces
        else "CONFIRMED (0 narration, 0 text-emitted episodes)"
        if narration_total == 0 and text_emit_total == 0
        else f"REFUTED ({narration_total} narration, {text_emit_total} text-emitted)"
    )

    # --- outputs ------------------------------------------------------------
    with (out_dir / "per_cell.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "category", "seed", "score", "calls_requested"])
        for r in rows:
            w.writerow(
                [r["model"], r["task"], r["category"], r["seed"], r["score"], r["calls_requested"]]
            )
    with (out_dir / "h3_flags.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "flag"])
        for fl in flags:
            w.writerow([fl["model"], fl["task"], fl["flag"]])

    cats = sorted({str(r["category"]) for r in rows})
    lines = [f"# {EXP_DIR_NAME} / {args.slug} — summary", ""]
    lines.append("| model | overall | " + " | ".join(cats) + " |")
    lines.append("| --- | --- |" + " --- |" * len(cats))
    for m in models:
        lines.append(
            f"| {m} | {overall[m]:.3f} | "
            + " | ".join(f"{per_cat[m].get(c, 0):.3f}" for c in cats)
            + " |"
        )
    for m, v in LOCAL_BASELINES.items():
        lines.append(
            f"| {m} (HARD-BENCH-002, local) | {v:.3f} | " + " | ".join("-" for _ in cats) + " |"
        )
    lines.append("")
    lines.append("## Verdicts")
    lines.append("")
    for h, v in verdicts.items():
        lines.append(f"- **{h}**: {v}")
    if flags:
        lines.append("")
        lines.append(f"## H3 flags ({len(flags)})")
        for fl in flags[:30]:
            lines.append(f"- {fl['model']} / {fl['task']}: {fl['flag']}")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
