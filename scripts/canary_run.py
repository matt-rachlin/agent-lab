#!/usr/bin/env python3
"""Nightly lab canary — does the whole stack still work?

Runs INSIDE the gpu pueue slot (submitted by lab-canary.timer via
canary_submit.sh), so it never overlaps sweeps.

1. Health probes: litellm /v1/models, ollama list, MLflow ping, lab pg.
2. A real mini-sweep: gemma4-12b (champion) on pbs-agent-v0.1 (12 easy
   tasks, historical pass rate 1.000 across all runs), slug CANARY-<date>.
3. Verdict: pass rate must be 1.0 and zero error cells.
4. Writes /data/lab/canary/status.json + history.csv; exits nonzero on
   any failure so the pueue task shows Failed in `mq`.
"""

from __future__ import annotations

import csv
import datetime
import json
import pathlib
import subprocess
import sys
import urllib.request

CANARY_DIR = pathlib.Path("/data/lab/canary")
LAB = "/data/lab/code"
KEY_FILE = "/data/lab/services/litellm-master-key"

# Frozen baseline: gemma4-12b on pbs-agent-v0.1 has scored 1.000 in every
# recorded run (CODER-BENCH-001 3 seeds, earlier agent benches). Change this
# only with a committed rationale.
EXPECTED_PASS_RATE = 1.0
MODEL = "gemma4-12b"
SUITE = "pbs-agent-v0.1"

SWEEP_TEMPLATE = """experiment:
  slug: {slug}
  title: "Nightly stack canary: {model} on {suite}"
  hypothesis: "Canary, not an experiment: stack regression check; expected pass rate 1.0."
tasks:
  suite: {suite}
models:
  - {model}
configs:
  - name: react-4096
    temperature: 0.0
    top_p: 1.0
    max_tokens: 4096
    scaffold: react
seeds: [1]
max_concurrency: 1
request_timeout_sec: 1800
"""


def sh(cmd: str, timeout: int = 7200) -> tuple[int, str]:
    p = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout, check=False
    )
    return p.returncode, (p.stdout + p.stderr)[-2000:]


def health() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    with open(KEY_FILE) as f:
        key = f.read().strip()
    try:
        req = urllib.request.Request(
            "http://localhost:4000/v1/models", headers={"Authorization": f"Bearer {key}"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            checks["litellm"] = len(json.load(r)["data"]) > 10
    except Exception:
        checks["litellm"] = False
    checks["ollama"] = sh("ollama list | grep -q gemma4")[0] == 0
    try:
        with urllib.request.urlopen("http://localhost:5050/health", timeout=15) as r:
            checks["mlflow"] = r.status == 200
    except Exception:
        checks["mlflow"] = False
    checks["postgres"] = sh("psql 'postgresql://m@/lab' -c 'select 1' -t -A")[0] == 0
    return checks


def main() -> None:
    CANARY_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    slug = f"CANARY-{today.replace('-', '')}"
    status: dict[str, object] = {"date": today, "slug": slug}

    hc = health()
    status["health"] = hc
    ok = all(hc.values())

    pass_rate = None
    error_cells = None
    if ok:
        cfg = pathlib.Path(f"/data/lab/tmp/canary-{today}.yaml")
        cfg.write_text(SWEEP_TEMPLATE.format(slug=slug, model=MODEL, suite=SUITE))
        rc, tail = sh(
            f"cd {LAB} && /home/m/.local/bin/uv run lab sweep run {cfg} --allow-slow-models"
        )
        status["sweep_rc"] = rc
        if rc != 0:
            status["sweep_tail"] = tail
            ok = False
        else:
            q = (
                "select count(*) filter (where (al.turns->'score_breakdown'->'end_state'"
                "->>'value')::float >= 1.0), count(*), "
                "count(*) filter (where er.error is not null) "
                "from experiment_runs er "
                "join experiments e on e.experiment_id=er.experiment_id "
                "left join agent_logs al on al.run_id=er.run_id "
                f"where e.slug='{slug}'"
            )
            rc2, out = sh(f"psql 'postgresql://m@/lab' -t -A -F'|' -c \"{q}\"")
            if rc2 == 0 and out.strip():
                passed, total, errors = (int(x) for x in out.strip().split("|"))
                pass_rate = passed / total if total else 0.0
                error_cells = errors
                ok = total == 12 and pass_rate >= EXPECTED_PASS_RATE and errors == 0
            else:
                ok = False

    status["pass_rate"] = pass_rate
    status["error_cells"] = error_cells
    status["ok"] = ok
    (CANARY_DIR / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    hist = CANARY_DIR / "history.csv"
    new = not hist.exists()
    with hist.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "slug", "ok", "pass_rate", "error_cells", "health"])
        w.writerow([today, slug, ok, pass_rate, error_cells, json.dumps(hc)])

    print(json.dumps(status, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
