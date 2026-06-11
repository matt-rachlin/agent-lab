"""Confidence-calibration scanner — do agents know when they're wrong?

Protocol: tool_use_system_v3 requires the final (no-tool-call) reply to end
with "confidence: NN" (0-100). This scanner joins each episode's stated
confidence with its machine-verified end_state score and reports, per model:

- coverage: fraction of episodes that emitted a parseable confidence line
- calibration table: confidence buckets (decile) vs empirical pass rate
- Brier score (stated confidence/100 vs binary outcome)
- overconfidence gap: mean(confidence)/100 - pass rate
- discrimination (AUC): does confidence rank passes above failures?

Final-turn text comes from the trajectory's last turn record content_preview
(4 KB cap — the confidence line is terminal, so the tail matters: previews
are prefix-truncated, so ALSO check agent_logs final turn and fall back to
the messages record's last assistant content, which is untruncated).

Graceful on experiments that never used v3: reports coverage 0 and exits 0.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from lab.core.minio_io import make_minio_client

PG_DSN = "dbname=lab host=/var/run/postgresql"
CONF_RE = re.compile(r"confidence:\s*(\d{1,3})\s*$", re.MULTILINE)


def parse_confidence(text: str) -> int | None:
    matches = CONF_RE.findall(text.strip())
    if not matches:
        return None
    val = int(matches[-1])
    return val if 0 <= val <= 100 else None


def final_assistant_text(blob: bytes) -> str:
    """Last assistant content from the trajectory's messages record
    (untruncated), falling back to the last turn's content_preview."""
    import json

    last_messages_text = ""
    last_preview = ""
    for line in blob.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec: dict[str, Any] = json.loads(line)
        except Exception:
            continue
        if rec.get("type") == "messages":
            for msg in rec.get("messages", []):
                if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                    last_messages_text = msg["content"]
        elif rec.get("type") == "turn":
            pv = rec.get("content_preview")
            if isinstance(pv, str) and pv.strip():
                last_preview = pv
    return last_messages_text or last_preview


def auc(pairs: list[tuple[int, float]]) -> float | None:
    """Rank AUC of confidence vs pass/fail; None if one class absent."""
    pos = [c for c, s in pairs if s >= 1.0]
    neg = [c for c, s in pairs if s < 1.0]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out or f"analysis/confidence/{args.slug}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sql = """
        select m.litellm_id as model, t.slug as task, er.seed, er.trace_path,
               (al.turns->'score_breakdown'->'end_state'->>'value')::float as score
        from experiment_runs er
        join experiments e on e.experiment_id = er.experiment_id
        join models m on m.model_id = er.model_id
        join tasks t on t.task_id = er.task_id
        left join agent_logs al on al.run_id = er.run_id
        where e.slug = %s
    """
    with psycopg.connect(PG_DSN, row_factory=dict_row) as conn:
        rows = [r for r in conn.execute(sql, (args.slug,)) if r["score"] is not None]
    if not rows:
        print(f"no scored cells for {args.slug}")
        return

    client = make_minio_client()
    cells: list[dict[str, Any]] = []
    for r in rows:
        tp = str(r["trace_path"] or "")
        conf: int | None = None
        if tp.startswith("s3://"):
            bucket, key = tp.removeprefix("s3://").split("/", 1)
            try:
                resp = client.get_object(bucket, key)
                blob = resp.read()
                resp.close()
                resp.release_conn()
                conf = parse_confidence(final_assistant_text(blob))
            except Exception:
                conf = None
        cells.append(
            {
                "model": r["model"],
                "task": r["task"],
                "seed": r["seed"],
                "score": float(r["score"]),
                "confidence": conf,
            }  # type: ignore[arg-type]
        )

    with (out_dir / "per_cell.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "seed", "score", "confidence"])
        for c in cells:
            w.writerow(
                [
                    c["model"],
                    c["task"],
                    c["seed"],
                    c["score"],
                    "" if c["confidence"] is None else c["confidence"],
                ]
            )

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        by_model[str(c["model"])].append(c)

    lines = [f"# Confidence calibration — {args.slug}", ""]
    lines.append("| model | n | coverage | mean conf | pass rate | overconf gap | Brier | AUC |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for model in sorted(by_model):
        mc = by_model[model]
        with_conf = [c for c in mc if c["confidence"] is not None]
        coverage = len(with_conf) / len(mc)
        if with_conf:
            pairs = [(int(c["confidence"]), float(c["score"])) for c in with_conf]
            mean_conf = sum(p for p, _ in pairs) / len(pairs) / 100
            pass_rate = sum(1 for _, s in pairs if s >= 1.0) / len(pairs)
            brier = sum((p / 100 - (1.0 if s >= 1.0 else 0.0)) ** 2 for p, s in pairs) / len(pairs)
            a = auc(pairs)
            lines.append(
                f"| {model} | {len(mc)} | {coverage:.2f} | {mean_conf:.2f} "
                f"| {pass_rate:.2f} | {mean_conf - pass_rate:+.2f} | {brier:.3f} "
                f"| {'-' if a is None else f'{a:.2f}'} |"
            )
        else:
            lines.append(f"| {model} | {len(mc)} | 0.00 | - | - | - | - | - |")
    lines.append("")
    lines.append("## Buckets (all models pooled, parseable cells)")
    lines.append("")
    lines.append("| confidence bucket | n | empirical pass rate |")
    lines.append("| --- | --- | --- |")
    pooled = [c for c in cells if c["confidence"] is not None]
    for lo in range(0, 100, 10):
        hi = lo + 10
        b = [c for c in pooled if lo <= int(c["confidence"]) < (hi if hi < 100 else 101)]
        if b:
            pr = sum(1 for c in b if float(c["score"]) >= 1.0) / len(b)
            lines.append(f"| {lo}-{hi} | {len(b)} | {pr:.2f} |")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
