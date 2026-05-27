"""Ollama Cloud quota tracker.

Ollama doesn't expose a usage endpoint, so we estimate budget consumption from
our own call records (LiteLLM proxy spend ledger + experiment_runs). The
tracker reports rough "fraction of weekly budget consumed" per tier so we can
alert before hitting the 429 storm.

Heuristics from RESEARCH_OPS_PLAN §"Inference & sweep execution":

- Pro tier (3 concurrent, 50x free baseline): comfortable ~50-150 cloud
  invocations/day on 20B/120B models; ~10-30/day on 480B/671B.
- Estimate GPU-second cost as (input_tokens / 100 + output_tokens / 10).
  Lower bound; calibrate as data accumulates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import psycopg

from lab.core.settings import get_settings

# Per-model "cost weight" — relative GPU-seconds per 1K output tokens.
# Calibrate empirically as we accumulate data.
_MODEL_WEIGHTS: dict[str, float] = {
    "gpt-oss-20b-cloud": 1.0,
    "gpt-oss-120b-cloud": 6.0,
    "qwen3-coder-480b-cloud": 24.0,
    "deepseek-v31-671b-cloud": 33.0,
    "kimi-k2-thinking-cloud": 50.0,
    "qwen3-vl-235b-cloud": 12.0,
}

# Comfortable weekly budget per tier, expressed in the same weighted units
# as _MODEL_WEIGHTS. Calibrate by observation; these are starting guesses.
TIER_WEEKLY_BUDGET: dict[Literal["free", "pro", "max"], float] = {
    "free": 50.0,
    "pro": 2500.0,  # 50x free
    "max": 12500.0,  # 5x pro
}


@dataclass(frozen=True)
class TierUsage:
    tier: str
    window_hours: int
    runs: int
    tokens_in: int
    tokens_out: int
    weighted_units: float
    budget: float
    pct_consumed: float


def usage_window(
    *,
    tier: Literal["free", "pro", "max"] = "pro",
    window_hours: int = 168,  # 7d
) -> TierUsage:
    """Estimate cloud usage over a rolling window."""
    since = datetime.now(UTC) - timedelta(hours=window_hours)
    with psycopg.connect(get_settings().pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.litellm_id, COUNT(*) AS runs,
                   SUM(COALESCE(r.tokens_in, 0)) AS tin,
                   SUM(COALESCE(r.tokens_out, 0)) AS tout
            FROM experiment_runs r
            JOIN models m ON m.model_id = r.model_id
            WHERE m.backend = 'ollama-cloud'
              AND r.started_at >= %s
            GROUP BY m.litellm_id
            """,
            (since,),
        )
        rows = cur.fetchall()

    runs = tokens_in = tokens_out = 0
    weighted_units = 0.0
    for litellm_id, n, tin, tout in rows:
        runs += int(n)
        tokens_in += int(tin or 0)
        tokens_out += int(tout or 0)
        weight = _MODEL_WEIGHTS.get(litellm_id, 5.0)
        # Lower bound: scale by output tokens (~10/sec at frontier speeds)
        weighted_units += weight * (int(tout or 0) / 1000.0)

    budget = TIER_WEEKLY_BUDGET[tier] * (window_hours / 168.0)
    pct = (weighted_units / budget * 100.0) if budget > 0 else 0.0
    return TierUsage(
        tier=tier,
        window_hours=window_hours,
        runs=runs,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        weighted_units=round(weighted_units, 2),
        budget=round(budget, 2),
        pct_consumed=round(pct, 1),
    )


def alert_if_high(
    threshold_pct: float = 80.0,
    *,
    tier: Literal["free", "pro", "max"] = "pro",
    window_hours: int = 168,
) -> TierUsage:
    """Compute usage and send an ntfy alert if above threshold. Returns the TierUsage."""
    from lab.core.notify import notify

    u = usage_window(tier=tier, window_hours=window_hours)
    if u.pct_consumed >= threshold_pct:
        priority = "max" if u.pct_consumed >= 95.0 else "high"
        notify(
            f"Ollama Cloud {u.tier}: {u.pct_consumed:.0f}% of {window_hours}h budget "
            f"({u.weighted_units} / {u.budget} weighted units, {u.runs} runs)",
            title="lab cloud quota",
            priority=priority,  # type: ignore[arg-type]
            tags=["warning"] if u.pct_consumed < 95.0 else ["rotating_light"],
        )
    return u
