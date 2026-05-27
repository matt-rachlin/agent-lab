"""Experiments explorer + cell drill-down.

Per-experiment summary (status, cell count, wall time, error rate) then
a model x task x seed grid. Clicking a cell loads the trajectory JSON
from MinIO. Hypothesis verdicts are surfaced from the docs/exp/ markdown
when available.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from lib import db  # noqa: E402
from lib import minio as minio_lib  # noqa: E402

st.set_page_config(page_title="Experiments", layout="wide")
st.title("Experiments")

LAB_HOME = Path("/data/lab/code")


@st.cache_data(ttl=60)
def _experiments_df():
    return db.pg_query(
        """
        SELECT
            e.experiment_id, e.slug, e.title, e.status,
            e.started_at, e.completed_at, e.hypothesis,
            COUNT(r.run_id) AS cells,
            COUNT(*) FILTER (WHERE r.status = 'error') AS errored,
            COALESCE(SUM(r.latency_ms), 0)::float / 1000 AS wall_sec
        FROM experiments e
        LEFT JOIN experiment_runs r USING (experiment_id)
        GROUP BY e.experiment_id
        ORDER BY e.experiment_id DESC
        """
    )


@st.cache_data(ttl=30)
def _cells_for(experiment_id: int):
    return db.pg_query(
        """
        SELECT
            r.run_id, r.status, r.seed, r.latency_ms, r.cost_usd, r.error,
            r.trace_path, r.started_at, r.completed_at,
            m.litellm_id AS model, t.slug AS task
        FROM experiment_runs r
        LEFT JOIN models m ON m.model_id = r.model_id
        LEFT JOIN tasks  t ON t.task_id  = r.task_id
        WHERE r.experiment_id = %s
        ORDER BY m.litellm_id, t.slug, r.seed
        """,
        (experiment_id,),
    )


def _verdicts_for(slug: str) -> str | None:
    """Look for matching docs/exp/EXP-NNN-...verdicts.md or hypothesis."""
    exp_dir = LAB_HOME / "docs" / "exp"
    if not exp_dir.is_dir():
        return None
    # slug looks like 'EXP-001-twelve-gb-agent'; match by prefix.
    prefix = re.match(r"^EXP-\d+", slug)
    if not prefix:
        return None
    key = prefix.group(0).lower()
    for p in sorted(exp_dir.glob("*.md")):
        if key in p.name.lower() and "verdict" in p.name.lower():
            return p.read_text(encoding="utf-8")
    return None


df = _experiments_df()
if df.empty or "_error" in df.columns:
    st.error(
        f"Could not load experiments: {df['_error'].iloc[0] if '_error' in df.columns else 'empty'}"
    )
    st.stop()

st.dataframe(
    df[["slug", "title", "status", "cells", "errored", "wall_sec", "completed_at"]],
    hide_index=True,
    use_container_width=True,
)

picked = st.selectbox("Drill into experiment", df["slug"].tolist())
exp_row = df[df["slug"] == picked].iloc[0]

st.subheader(f"{picked} - {exp_row['status']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cells", int(exp_row["cells"]))
c2.metric("Errored", int(exp_row["errored"]))
err_rate = (exp_row["errored"] / exp_row["cells"]) if exp_row["cells"] else 0.0
c3.metric("Error rate", f"{err_rate:.1%}")
c4.metric("Wall (s)", f"{exp_row['wall_sec']:.1f}")

if exp_row["hypothesis"]:
    with st.expander("Hypothesis", expanded=False):
        st.write(exp_row["hypothesis"])

cells = _cells_for(int(exp_row["experiment_id"]))
if cells.empty or "_error" in cells.columns:
    st.info("No runs yet." if cells.empty else f"DB error: {cells['_error'].iloc[0]}")
else:
    st.markdown("### Cell grid (model x task x seed)")
    st.dataframe(
        cells[["run_id", "model", "task", "seed", "status", "latency_ms", "cost_usd", "error"]],
        hide_index=True,
        use_container_width=True,
    )

    # Trajectory viewer
    run_ids = cells["run_id"].tolist()
    if run_ids:
        chosen_run = st.selectbox("Open trajectory for run", run_ids)
        crow = cells[cells["run_id"] == chosen_run].iloc[0]
        trace_path = crow.get("trace_path")
        if not trace_path:
            st.caption("No trace_path recorded for this run.")
        else:
            st.caption(f"Loading trace from {trace_path}")
            data = minio_lib.get_json(str(trace_path))
            if data is None:
                # Try as text (JSONL)
                text = minio_lib.get_text(str(trace_path))
                if text is None:
                    st.error("Could not load trajectory from MinIO.")
                else:
                    st.code(text[:50_000], language="json")
            else:
                st.json(data, expanded=False)

verdicts = _verdicts_for(picked)
if verdicts:
    with st.expander("Verdicts", expanded=False):
        st.markdown(verdicts)
