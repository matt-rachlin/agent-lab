"""ADR-008 result-trust lifecycle: hash-chained transitions + validity gate.

`record_transition` appends an append-only, hash-chained row to
`trust_transitions` and advances `experiment_runs.trust_level`. The chain is
global and assumes serialized writes (the sweep GPU group); DB-level append-only
grants and Ed25519 signature enforcement on verified/finding promotions land in
Stage 0b action-control (#10). See docs/adr/ADR-008-result-trust-lifecycle.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Json

from lab.core.settings import get_settings

TRUST_LEVELS = (
    "raw",
    "validity_passed",
    "reliability_confirmed",
    "verification_attempted",
    "verified",
    "finding",
)


def _row_hash(
    prev_hash: str | None,
    run_id: str,
    from_level: str | None,
    to_level: str,
    actor: str,
    is_human: bool,
    evidence: dict[str, Any] | None,
    reason: str | None,
) -> str:
    """Tamper-evidence: bind this transition to the prior row's hash."""
    payload = json.dumps(
        {
            "prev": prev_hash,
            "run_id": run_id,
            "from": from_level,
            "to": to_level,
            "actor": actor,
            "is_human": is_human,
            "evidence": evidence,
            "reason": reason,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def record_transition(
    run_id: str,
    to_level: str,
    *,
    actor: str,
    is_human: bool = False,
    evidence: dict[str, Any] | None = None,
    reason: str | None = None,
    signature: str | None = None,
    conn: psycopg.Connection | None = None,
) -> str:
    """Append a hash-chained transition and advance the run's trust_level."""
    if to_level not in TRUST_LEVELS:
        raise ValueError(f"unknown trust level {to_level!r}")
    if to_level == "finding" and not is_human and signature is None:
        raise ValueError(
            "promotion to 'finding' requires human approval (is_human=True) or a signature (ADR-008)"
        )
    own = conn is None
    conn = conn or psycopg.connect(get_settings().pg_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT trust_level FROM experiment_runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"run {run_id!r} not found")
            from_level = row[0]
            cur.execute("SELECT row_hash FROM trust_transitions ORDER BY id DESC LIMIT 1")
            last = cur.fetchone()
            prev_hash = last[0] if last else None
            rh = _row_hash(
                prev_hash, run_id, from_level, to_level, actor, is_human, evidence, reason
            )
            cur.execute(
                "INSERT INTO trust_transitions (run_id, from_level, to_level, actor, "
                "is_human, evidence, reason, prev_hash, row_hash, signature) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    run_id,
                    from_level,
                    to_level,
                    actor,
                    is_human,
                    Json(evidence) if evidence is not None else None,
                    reason,
                    prev_hash,
                    rh,
                    signature,
                ),
            )
            cur.execute(
                "UPDATE experiment_runs SET trust_level = %s WHERE run_id = %s", (to_level, run_id)
            )
        if own:
            conn.commit()
        return rh
    finally:
        if own:
            conn.close()


@dataclass
class ValidityReport:
    """Did the eval measure the model (not the harness)? cf. F-017."""

    passed: bool
    violations: list[str]
    emitted: bool | None = None
    correct: bool | None = None


def decode_integrity(raw_response: dict[str, Any] | None) -> list[str]:
    """Advanced validity check: decode/template artefacts — truncation or an empty
    completion. A response was captured but mangled (cf. F-017 reasoning-token
    truncation). Code-doable; complements emission/correctness."""
    choices = (raw_response or {}).get("choices") or []
    if not choices:
        return ["decode: no choices in response"]
    ch = choices[0] or {}
    out: list[str] = []
    if ch.get("finish_reason") == "length":
        out.append("decode: response truncated (finish_reason=length)")
    msg = ch.get("message") or {}
    if not (msg.get("content") or msg.get("tool_calls")):
        out.append("decode: empty completion")
    return out


def baseline_sanity(value: float, lo: float | None, hi: float | None) -> list[str]:
    """Advanced validity check: flag an aggregate implausibly outside a known-good
    range, in EITHER direction (matching a published number is not proof). No range
    configured -> no-op. Ranges are populated from data later (a baselines registry)."""
    out: list[str] = []
    if lo is not None and value < lo:
        out.append(f"baseline: {value:.3f} below expected floor {lo:.3f}")
    if hi is not None and value > hi:
        out.append(f"baseline: {value:.3f} above expected ceiling {hi:.3f}")
    return out


def bfcl_validity(
    *,
    request_tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    bfcl_error_type: str | None,
    passed: Any,
) -> ValidityReport:
    """Validity gate for a BFCL cell. Generalises the F-017 lesson: a result is
    only admissible if the request was faithful (tools actually sent, tool_choice
    recorded) and emission is separated from correctness."""
    violations: list[str] = []
    if not request_tools:
        violations.append("precondition: BFCL task expects tools but none were sent to the model")
    if tool_choice is None:
        violations.append("request-fidelity: tool_choice not recorded")
    emitted = bfcl_error_type != "model_output:no_tool_call"
    correct = bool(passed)
    return ValidityReport(
        passed=not violations, violations=violations, emitted=emitted, correct=correct
    )


def single_turn_validity(
    *,
    request_sampling: dict[str, Any] | None,
    response_text: str | None,
    raw_response: dict[str, Any] | None,
) -> ValidityReport:
    """Validity for a single-turn (non-tool) cell: the request was recorded and
    the model actually produced output (telemetry integrity)."""
    violations: list[str] = []
    if not request_sampling:
        violations.append("request-fidelity: sampling params not recorded")
    has_output = bool(response_text) or bool((raw_response or {}).get("choices"))
    if not has_output:
        violations.append("telemetry: done run produced no model output")
    elif (raw_response or {}).get("choices"):
        violations.extend(decode_integrity(raw_response))
    return ValidityReport(
        passed=not violations, violations=violations, emitted=has_output, correct=None
    )
