"""Run the ADR-008 BFCL verifier battery for the scoreboard candidates and, on a
HOLDS verdict, promote their capability + safety baseline runs to `verified`
(battery basis, not the D5 reliability shortcut). Then render the scoreboard.

Usage: cand_battery_promote.py [--dry]
  --dry : skip the GPU battery (stub HOLDS) and ROLL BACK promotions — validates
          plumbing only.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict

import psycopg
from lab.platform.trust import record_transition
from lab.platform.verifier import BatteryConfig, BatteryResult, verdict

from lab.core.settings import get_settings
from lab.eval.bfcl_battery import _tool_choice_for, run_bfcl_battery

DRY = "--dry" in sys.argv

ANCHORS = {
    "tool_tuned": ["qwen3-4b-ft-toolcall-q4-latest", "gemma4-12b"],
    "general": ["qwen3-8b", "llama3.1-8b-q4"],
}
SUBJECTS = {
    "qwen3-14b-q4": ("CAND-BFCL-QWEN3-14B-001", "CAND-SAFETY-QWEN3-14B-001"),
    "gpt-oss-20b-ollama": ("CAND-BFCL-GPTOSS-20B-001", "CAND-SAFETY-GPTOSS-20B-001"),
}
_STUB = BatteryResult(16, True, 5, True, 2, True, {"tool_tuned": 2, "general": 2}, True)


def run_ids(conn: psycopg.Connection, slug: str, model: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT r.run_id FROM experiment_runs r "
            "JOIN experiments e ON e.experiment_id=r.experiment_id "
            "JOIN models m ON m.model_id=r.model_id "
            "WHERE e.slug=%s AND m.litellm_id=%s AND r.status='done' "
            "AND r.trust_level <> 'verified'",
            (slug, model),
        )
        return [row[0] for row in cur.fetchall()]


def main() -> None:
    settings = get_settings()
    for subject, (cap_slug, safety_slug) in SUBJECTS.items():
        print(
            f"\n===== BATTERY: {subject} (tool_choice={_tool_choice_for(subject)}) =====",
            flush=True,
        )
        if DRY:
            br = _STUB
            print("  [dry] skipping GPU battery, using stub HOLDS")
        else:
            br = run_bfcl_battery(
                subject=subject,
                subject_tool_choice=_tool_choice_for(subject),
                anchors_by_class=ANCHORS,
                n_tasks=30,
                seeds=16,
                variants=5,
            )
        v = verdict(br, BatteryConfig())
        print(f"  verdict={v.outcome} reasons={v.reasons}")
        print(f"  battery={asdict(br)}", flush=True)

        # Safety eval_results (constraint_violations) must exist before promotion.
        if not DRY:
            subprocess.run(
                [
                    "uv",
                    "run",
                    "lab",
                    "eval",
                    "apply",
                    safety_slug,
                    "--only",
                    "constraint_violations",
                    "--no-judge",
                ],
                check=False,
            )

        with psycopg.connect(settings.pg_dsn) as conn:
            cap = run_ids(conn, cap_slug, subject)
            safety = run_ids(conn, safety_slug, subject)
            print(f"  promotable runs: cap={len(cap)} safety={len(safety)}")
            if v.outcome == "holds":
                ev = {"verdict": "verified-by-battery", "battery": asdict(br)}
                for rid in cap + safety:
                    record_transition(
                        rid, "verified", actor="system:verifier:battery", evidence=ev, conn=conn
                    )
                print(f"  -> promoted {len(cap) + len(safety)} runs to verified")
            else:
                rep = (cap + safety)[:1]
                for rid in rep:
                    record_transition(
                        rid,
                        "verification_attempted",
                        actor="system:verifier:battery",
                        evidence={
                            "verdict": v.outcome,
                            "reasons": v.reasons,
                            "battery": asdict(br),
                        },
                        conn=conn,
                    )
                print(f"  -> NOT promoted (verdict {v.outcome}); recorded verification_attempted")
            if DRY:
                conn.rollback()
                print("  [dry] rolled back")
            else:
                conn.commit()

    print("\n===== SCOREBOARD =====", flush=True)
    subprocess.run(["uv", "run", "lab", "analyze", "scoreboard"], check=False)


if __name__ == "__main__":
    main()
