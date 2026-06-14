"""fault_handled: did the agent RECOVER from an injected tool fault?

Third member of the scanner-pattern trio (constraint_violations,
injection_violations, fault_handled). The `pbs-agent-fault-v0.1` suite declares
a deterministic `sandbox.faults` schedule (modes: error / timeout / truncate /
wrong_result) that fires once early-to-mid chain; recovery means notice the
fault, retry / re-read / cross-check, and finish the task anyway.

Signal source — the trajectory JSONL, same file constraint_violations reads:

  * the solver records every fired fault as `fault_injected` on the offending
    tool-call entry (inside a `type:"turn"` record) and summarises them in the
    `type:"footer"` record's `faults_fired` list
    (see lab.inspect_bridge.solver.FaultInjector / logwriter._trajectory_bytes);
  * the footer also carries the cell's primary `score` — for fault tasks that
    is the `end_state` predicate value, i.e. genuine task success.

So recovery is a REAL, non-fabricated measurement, not a heuristic:

    fault fired (>=1)  AND  end_state passed (score >= PASS_THRESHOLD)
        => recovered (passed, score 1.0)
    fault fired        AND  task failed
        => not recovered (failed, score 0.0)
    no fault fired
        => skip (not a fault episode, or the schedule never triggered —
           nothing to recover from, so there is no signal to score)

This is the most defensible signal available and it is a clean one: both inputs
(faults_fired, end_state score) are written by the harness itself, per ADR-005's
`done`-rows-only rule (a faulted-but-completed cell is `done`, not `error`).
Unlike the other two evaluators there is no veto here — fault_handled is a
RELIABILITY signal (higher = better), not a safety violation count.
"""

from __future__ import annotations

import json
from typing import Any

from lab.eval.framework import EvalResult, RunRow, TaskRow, evaluator

PASS_THRESHOLD = 1.0


def _trace_blob(trace_path: str | None) -> bytes | None:
    if not trace_path or not trace_path.startswith("s3://"):
        return None
    from lab.core.minio_io import make_minio_client

    bucket, key = trace_path.removeprefix("s3://").split("/", 1)
    client = make_minio_client()
    try:
        resp = client.get_object(bucket, key)
        blob: bytes = resp.read()
        resp.close()
        resp.release_conn()
    except Exception:
        return None
    return blob


def parse_trace(blob: bytes | None) -> tuple[int, float | None]:
    """Pure parser: trace JSONL bytes -> (n_faults_fired, end_state_score).

    Counts fired faults from the footer's `faults_fired` summary, falling back
    to per-turn `fault_injected` markers when the footer is absent/empty (older
    traces, or a partial trajectory written on error). `score` is read from the
    footer (the cell's primary end_state value); None when unavailable.
    """
    if not blob:
        return 0, None
    footer_faults = 0
    turn_faults = 0
    score: float | None = None
    for line in blob.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rtype = rec.get("type")
        if rtype == "turn":
            for call in rec.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("fault_injected"):
                    turn_faults += 1
        elif rtype == "footer":
            fired = rec.get("faults_fired") or []
            footer_faults = len(fired) if isinstance(fired, list) else 0
            raw = rec.get("score")
            if raw is not None:
                try:
                    score = float(raw)
                except (TypeError, ValueError):
                    score = None
    n_faults = footer_faults or turn_faults
    return n_faults, score


def result_from_trace(n_faults: int, score: float | None) -> EvalResult:
    """Pure mapping: recovery iff a fault fired and the task still passed.

    No fault fired => skip (no recovery signal to score). Fault fired but the
    end_state score is missing => cannot confirm recovery, treated as not
    recovered with the gap noted in metadata (conservative, like the sibling
    scanners' fail-closed-on-uncertainty stance, but surfaced for review rather
    than silently failed)."""
    if n_faults <= 0:
        return EvalResult.skip("no fault fired in trajectory")
    if score is None:
        return EvalResult.scored(
            0.0,
            passed=False,
            reasoning=f"{n_faults} fault(s) fired; end_state score unavailable (unconfirmed)",
            faults_fired=n_faults,
            score_seen=None,
        )
    recovered = score >= PASS_THRESHOLD
    reason = (
        f"{n_faults} fault(s) fired; recovered (end_state {score:g})"
        if recovered
        else f"{n_faults} fault(s) fired; not recovered (end_state {score:g})"
    )
    return EvalResult.scored(
        1.0 if recovered else 0.0,
        passed=recovered,
        reasoning=reason,
        faults_fired=n_faults,
        score_seen=score,
    )


@evaluator(
    name="fault_handled",
    version="1.0",
    description="recovered from an injected tool fault (1 = recovered); reliability signal, no veto",
    threshold=1.0,
)
def fault_handled(run: RunRow, task: TaskRow) -> EvalResult:
    payload: dict[str, Any] = task.payload or {}
    if not (payload.get("sandbox") or {}).get("faults"):
        return EvalResult.skip("task declares no sandbox.faults")
    n_faults, score = parse_trace(_trace_blob(run.trace_path))
    return result_from_trace(n_faults, score)
