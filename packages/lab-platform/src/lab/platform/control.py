"""Agent action-control: kill switch, budgets, append-only action audit (D5).

The control plane lives in `agent_control` (singleton) + `agent_action_log`
(append-only, hash-chained). The kill switch and budgets are checked by the
agent before any action; the audit records every action it takes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import psycopg
from psycopg.types.json import Json

from lab.core.settings import get_settings


def is_killed() -> tuple[bool, str | None]:
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT killed, killed_reason FROM agent_control WHERE id")
        row = cur.fetchone()
    return (bool(row[0]), row[1]) if row else (False, None)


def kill(reason: str) -> None:
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE agent_control SET killed=true, killed_reason=%s, killed_at=now(), "
            "updated_at=now() WHERE id",
            (reason,),
        )
        conn.commit()


def clear_kill() -> None:
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE agent_control SET killed=false, killed_reason=NULL, killed_at=NULL, "
            "updated_at=now() WHERE id"
        )
        conn.commit()


def budget_status() -> dict[str, Any]:
    """Today's spend vs configured daily caps (null cap => unlimited)."""
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT daily_usd_budget, daily_token_budget, daily_gpu_sec_budget "
            "FROM agent_control WHERE id"
        )
        cap = cur.fetchone() or (None, None, None)
        cur.execute(
            "SELECT COALESCE(SUM(cost_usd),0), "
            "COALESCE(SUM(tokens_in),0)+COALESCE(SUM(tokens_out),0), "
            "COALESCE(SUM(cost_gpu_sec),0) "
            "FROM experiment_runs WHERE started_at::date = CURRENT_DATE"
        )
        spent = cur.fetchone() or (0, 0, 0)
    return {
        "usd": {"cap": cap[0], "spent": float(spent[0])},
        "tokens": {"cap": cap[1], "spent": int(spent[1])},
        "gpu_sec": {"cap": cap[2], "spent": float(spent[2])},
    }


def budget_exceeded() -> bool:
    st = budget_status()
    for k in ("usd", "tokens", "gpu_sec"):
        cap = st[k]["cap"]
        if cap is not None and st[k]["spent"] >= float(cap):
            return True
    return False


def record_action(
    actor: str,
    action: str,
    *,
    args: dict[str, Any] | None = None,
    approved_by: str | None = None,
    outcome: str | None = None,
    signature: str | None = None,
) -> str:
    """Append a hash-chained row to the agent action audit; returns row_hash."""
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT row_hash FROM agent_action_log ORDER BY id DESC LIMIT 1")
        last = cur.fetchone()
        prev_hash = last[0] if last else None
        payload = json.dumps(
            {
                "prev": prev_hash,
                "actor": actor,
                "action": action,
                "args": args,
                "approved_by": approved_by,
                "outcome": outcome,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        row_hash = hashlib.sha256(payload.encode()).hexdigest()
        cur.execute(
            "INSERT INTO agent_action_log (actor, action, args, approved_by, outcome, "
            "prev_hash, row_hash) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (
                actor,
                action,
                Json(args) if args is not None else None,
                approved_by,
                outcome,
                prev_hash,
                row_hash,
            ),
        )
        conn.commit()
    return row_hash
