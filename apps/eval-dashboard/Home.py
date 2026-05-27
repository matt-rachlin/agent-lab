"""Eval dashboard — Home page.

Service health, open experiments, recent findings, live sweep progress,
cumulative stats. Reads Postgres + MinIO directly; never imports from
`lab.*` so this app stays decoupled from sibling repo refactors.
"""

from __future__ import annotations

# Re-export for tests; no-op at runtime.
import sys
from pathlib import Path

import streamlit as st

# Make `lib` importable when streamlit launches us from any cwd.
_APP_DIR = Path(__file__).parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from lib import db, services  # noqa: E402

st.set_page_config(
    page_title="lab — eval dashboard",
    page_icon="lab",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _status_dot(ok: bool) -> str:
    return "🟢" if ok else "🔴"


@st.cache_data(ttl=10)
def _service_health() -> list[tuple[str, bool, str]]:
    return [(s.name, s.ok, s.detail) for s in services.all_services()]


@st.cache_data(ttl=30)
def _open_experiments():
    return db.pg_query(
        """
        SELECT slug, title, status, started_at, completed_at
        FROM experiments
        WHERE status IN ('planned', 'running')
        ORDER BY COALESCE(started_at, created_at) DESC
        LIMIT 25
        """
    )


@st.cache_data(ttl=30)
def _recent_findings():
    return db.pg_query(
        """
        SELECT slug, claim, confidence, status, created_at
        FROM findings
        ORDER BY finding_id DESC
        LIMIT 10
        """
    )


@st.cache_data(ttl=10)
def _sweep_progress():
    return db.pg_query(
        """
        SELECT
            e.slug AS experiment,
            COUNT(*) FILTER (WHERE r.status = 'done')    AS done,
            COUNT(*) FILTER (WHERE r.status = 'running') AS running,
            COUNT(*) FILTER (WHERE r.status = 'queued')  AS queued,
            COUNT(*) FILTER (WHERE r.status = 'error')   AS errored,
            COUNT(*) AS total
        FROM experiment_runs r
        JOIN experiments e USING (experiment_id)
        WHERE e.status = 'running'
        GROUP BY e.slug
        ORDER BY total DESC
        """
    )


@st.cache_data(ttl=30)
def _stats():
    return db.aggregate_stats()


st.title("lab — eval dashboard")
st.caption("Local-only. Postgres + MinIO direct reads. No deploy story.")

# Service health
st.subheader("Service health")
health = _service_health()
cols = st.columns(len(health))
for col, (name, ok, detail) in zip(cols, health, strict=False):
    col.markdown(f"{_status_dot(ok)} **{name}**")
    col.caption(detail)

# Cumulative stats
st.subheader("Cumulative")
stats = _stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Experiments", stats["experiments"])
c2.metric("Runs", stats["runs"])
c3.metric("Findings", stats["findings"])
c4.metric("7-day spend (USD)", f"${stats['spend_7d_usd']:.2f}")

# Two-column layout for tables
left, right = st.columns(2)

with left:
    st.subheader("Open experiments")
    exps = _open_experiments()
    if exps.empty or "_error" in exps.columns:
        st.info("No open experiments." if exps.empty else f"DB error: {exps['_error'].iloc[0]}")
    else:
        st.dataframe(exps, hide_index=True, use_container_width=True)

with right:
    st.subheader("Recent findings")
    fnd = _recent_findings()
    if fnd.empty or "_error" in fnd.columns:
        st.info("No findings yet." if fnd.empty else f"DB error: {fnd['_error'].iloc[0]}")
    else:
        st.dataframe(fnd, hide_index=True, use_container_width=True)

# Live sweep progress
st.subheader("Live sweep progress")
sweeps = _sweep_progress()
if sweeps.empty or "_error" in sweeps.columns:
    st.info("No running sweeps." if sweeps.empty else f"DB error: {sweeps['_error'].iloc[0]}")
else:
    for _, row in sweeps.iterrows():
        done = int(row["done"])
        total = int(row["total"]) or 1
        pct = done / total
        st.markdown(
            f"**{row['experiment']}** — "
            f"done={done} running={int(row['running'])} "
            f"queued={int(row['queued'])} errored={int(row['errored'])} / {total}"
        )
        st.progress(pct)

st.caption("Auto-cached 10-30s. Use the sidebar to drill into other pages.")
