"""Sweep monitor (live).

10s auto-refresh. Progress bar per in-flight experiment. GPU lease state
(Valkey). Rerank queue depth. Per-cell error stream from
experiment_runs.error.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from lib import db  # noqa: E402

st.set_page_config(page_title="Sweep monitor", layout="wide")
st.title("Sweep monitor")

REFRESH_S = 10
# st.autorefresh: triggers a rerun every REFRESH_S seconds. If the
# component isn't available (e.g. very old streamlit), we just no-op.
with contextlib.suppress(AttributeError):
    st.autorefresh = st.experimental_autorefresh  # type: ignore[attr-defined]

if hasattr(st, "autorefresh"):
    with contextlib.suppress(Exception):
        st.autorefresh(interval=REFRESH_S * 1000, key="sweep-monitor-tick")


@st.cache_data(ttl=REFRESH_S)
def _running_sweeps():
    return db.pg_query(
        """
        SELECT
            e.experiment_id,
            e.slug,
            COUNT(*) FILTER (WHERE r.status = 'done')    AS done,
            COUNT(*) FILTER (WHERE r.status = 'running') AS running,
            COUNT(*) FILTER (WHERE r.status = 'queued')  AS queued,
            COUNT(*) FILTER (WHERE r.status = 'error')   AS errored,
            COUNT(*) AS total,
            MAX(r.started_at) AS last_started
        FROM experiments e
        JOIN experiment_runs r USING (experiment_id)
        WHERE e.status = 'running'
        GROUP BY e.experiment_id, e.slug
        ORDER BY total DESC
        """
    )


@st.cache_data(ttl=REFRESH_S)
def _recent_errors():
    return db.pg_query(
        """
        SELECT r.run_id, e.slug AS experiment, m.litellm_id AS model,
               t.slug AS task, r.error, r.completed_at
        FROM experiment_runs r
        JOIN experiments e USING (experiment_id)
        LEFT JOIN models m ON m.model_id = r.model_id
        LEFT JOIN tasks  t ON t.task_id  = r.task_id
        WHERE r.status = 'error'
          AND r.completed_at >= NOW() - INTERVAL '24 hours'
        ORDER BY r.completed_at DESC
        LIMIT 100
        """
    )


def _gpu_lease() -> dict[str, str | None]:
    """Read GPU lease state from Valkey. Optional - returns None values on fail."""
    out: dict[str, str | None] = {"holder": None, "ttl_s": None}
    try:
        import redis

        url = os.environ.get("LAB_REDIS_URL", "redis://localhost:6379/0")
        client = redis.Redis.from_url(url, socket_timeout=1.0)
        for key in ("lab:gpu:lease", "lab:gpu_lease", "gpu_lease"):
            holder = client.get(key)
            if holder:
                out["holder"] = holder.decode() if isinstance(holder, bytes) else str(holder)
                ttl = client.ttl(key)
                if ttl is not None and ttl >= 0:
                    out["ttl_s"] = str(ttl)
                break
    except Exception:  # noqa: S110 — Valkey is best-effort; degrade silently
        pass
    return out


def _rerank_queue_depth() -> int | None:
    """Best-effort: hit the rerank server /metrics or /healthz with queue info."""
    try:
        from urllib.request import urlopen

        with urlopen("http://127.0.0.1:8401/healthz", timeout=1.0) as r:
            text = r.read().decode()
        # rerank server's /healthz returns JSON with a 'queue' field when busy.
        import json

        data = json.loads(text)
        if isinstance(data, dict) and "queue" in data:
            return int(data["queue"])
    except Exception:
        return None
    return None


sweeps = _running_sweeps()
if sweeps.empty or "_error" in sweeps.columns:
    st.info("No running sweeps." if sweeps.empty else f"DB error: {sweeps['_error'].iloc[0]}")
else:
    for _, row in sweeps.iterrows():
        done = int(row["done"])
        total = int(row["total"]) or 1
        pct = done / total
        st.markdown(
            f"**{row['slug']}** - "
            f"done={done} / total={total} "
            f"(running={int(row['running'])} queued={int(row['queued'])} "
            f"errored={int(row['errored'])})"
        )
        st.progress(pct)

st.divider()
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("GPU lease (Valkey)")
    lease = _gpu_lease()
    if lease["holder"]:
        st.metric("Held by", lease["holder"])
        if lease["ttl_s"]:
            st.caption(f"TTL: {lease['ttl_s']}s")
    else:
        st.caption("No active GPU lease.")

with col_b:
    st.subheader("Rerank queue")
    depth = _rerank_queue_depth()
    if depth is None:
        st.caption("rerank-server unreachable or no /healthz queue field.")
    else:
        st.metric("Queue depth", depth)

st.subheader("Recent errors (24h)")
errs = _recent_errors()
if errs.empty or "_error" in errs.columns:
    st.caption("No errors in the last 24h.")
else:
    st.dataframe(errs, hide_index=True, use_container_width=True)

st.caption(f"Auto-refreshing every {REFRESH_S}s.")
