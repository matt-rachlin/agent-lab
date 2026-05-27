"""Model leaderboard.

Per-model aggregates from experiment_runs joined with the per-turn
score_breakdown in agent_logs.turns. Sortable; one row per litellm_id.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from lib import db  # noqa: E402

st.set_page_config(page_title="Leaderboard", layout="wide")
st.title("Model leaderboard")


@st.cache_data(ttl=60)
def _leaderboard() -> pd.DataFrame:
    """One row per model with aggregates over completed runs.

    end_state and tool_correctness live in agent_logs.turns per turn as
    a JSON array; we average the score_breakdown across each run's last
    turn (the typical "final verdict" turn).
    """
    return db.pg_query(
        """
        WITH last_turn AS (
            SELECT
                a.run_id,
                (a.turns -> -1) AS last
            FROM agent_logs a
            WHERE jsonb_typeof(a.turns) = 'array'
              AND jsonb_array_length(a.turns) > 0
        )
        SELECT
            m.litellm_id AS model,
            m.publisher,
            m.backend,
            COUNT(r.run_id)                                                       AS runs,
            AVG((lt.last -> 'score_breakdown' ->> 'end_state')::float)            AS mean_end_state,
            AVG((lt.last -> 'score_breakdown' ->> 'tool_correctness')::float)     AS mean_tool_corr,
            AVG(r.cost_usd)::float                                                AS mean_cost,
            AVG(r.latency_ms)::float                                              AS mean_latency_ms
        FROM models m
        LEFT JOIN experiment_runs r ON r.model_id = m.model_id AND r.status = 'done'
        LEFT JOIN last_turn       lt ON lt.run_id = r.run_id
        GROUP BY m.litellm_id, m.publisher, m.backend
        HAVING COUNT(r.run_id) > 0
        ORDER BY runs DESC
        """
    )


@st.cache_data(ttl=60)
def _fallback_leaderboard() -> pd.DataFrame:
    """Fallback when agent_logs.turns has no score_breakdown yet."""
    return db.pg_query(
        """
        SELECT
            m.litellm_id     AS model,
            m.publisher,
            m.backend,
            COUNT(r.run_id)  AS runs,
            AVG(r.cost_usd)::float    AS mean_cost,
            AVG(r.latency_ms)::float  AS mean_latency_ms
        FROM models m
        JOIN experiment_runs r ON r.model_id = m.model_id AND r.status = 'done'
        GROUP BY m.litellm_id, m.publisher, m.backend
        ORDER BY runs DESC
        """
    )


df = _leaderboard()
used_fallback = False
if df.empty or "_error" in df.columns:
    st.warning("No agent_logs scoring data yet - falling back to plain run stats.")
    df = _fallback_leaderboard()
    used_fallback = True

if df.empty or "_error" in df.columns:
    st.error(
        f"Could not load leaderboard: "
        f"{df['_error'].iloc[0] if '_error' in df.columns else 'no data'}"
    )
    st.stop()

# Display formatted
sort_col = st.selectbox(
    "Sort by",
    [c for c in df.columns if c not in ("model", "publisher", "backend")],
    index=0,
)
df = df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)

# Round float columns for display
display = df.copy()
for col in display.columns:
    if pd.api.types.is_float_dtype(display[col]):
        display[col] = display[col].round(4)

st.dataframe(display, hide_index=False, use_container_width=True)

if not used_fallback:
    st.caption(
        "mean_end_state / mean_tool_correctness aggregated from the **last turn** "
        "of each run's `agent_logs.turns[*].score_breakdown`."
    )
else:
    st.caption("Fallback view - score_breakdown not yet populated in agent_logs.turns.")
