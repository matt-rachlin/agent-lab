"""Doc graph explorer.

Reads the Phase-14 SQLite catalog at ~/db/m/docs.db. Counts, per-zone
breakdown, search, and an in/out edge view for a selected doc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from lib import docs  # noqa: E402

st.set_page_config(page_title="Docs", layout="wide")
st.title("Doc graph")

if not docs.db_exists():
    st.error("Doc catalog not found at ~/db/m/docs.db. Run `m docs scan` to populate it.")
    st.stop()

# Top stats
stats = docs.stats()
c1, c2, c3 = st.columns(3)
c1.metric("Total docs", stats["total"])
c2.metric("Orphans", stats["orphans"])
c3.metric("Gaps (parse errors)", stats["gaps"])

st.subheader("By zone")
zones = docs.by_zone()
if zones.empty or "_error" in zones.columns:
    st.caption("No docs indexed yet.")
else:
    st.dataframe(zones, hide_index=True, use_container_width=True)

st.subheader("Search")
q = st.text_input("Substring match on title / doc_id / path", "")
if q:
    hits = docs.search(q, limit=200)
    if hits.empty or "_error" in hits.columns:
        st.caption("No hits.")
    else:
        st.dataframe(hits, hide_index=True, use_container_width=True)
        chosen = st.selectbox("Open doc", hits["doc_id"].tolist())
        if chosen:
            out_edges, in_edges = docs.edges_for(chosen)
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown("**Outgoing (depends_on)**")
                if out_edges.empty:
                    st.caption("(none)")
                else:
                    st.dataframe(out_edges, hide_index=True, use_container_width=True)
            with ec2:
                st.markdown("**Incoming (depended on by)**")
                if in_edges.empty:
                    st.caption("(none)")
                else:
                    st.dataframe(in_edges, hide_index=True, use_container_width=True)

            # Doc body if accessible
            doc_row = docs.query(
                "SELECT path, kind, status, last_updated FROM docs WHERE doc_id = ?",
                (chosen,),
            )
            if not doc_row.empty:
                row = doc_row.iloc[0]
                st.caption(
                    f"kind={row['kind']} status={row['status']} last_updated={row['last_updated']}"
                )
                p = Path(str(row["path"]))
                if p.is_file():
                    with st.expander("Body", expanded=False):
                        st.markdown(p.read_text(encoding="utf-8"))
else:
    st.caption("Enter a query to find docs.")

# Bottom: explore by zone selector
st.subheader("Browse by zone")
if not zones.empty and "_error" not in zones.columns:
    sel_zone = st.selectbox("Zone", zones["zone"].tolist())
    rows = docs.query(
        "SELECT doc_id, kind, status, title, last_updated FROM docs "
        "WHERE zone = ? ORDER BY last_updated DESC LIMIT 200",
        (sel_zone,),
    )
    if not rows.empty:
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _unused_pandas_anchor() -> pd.DataFrame:
    """Touch the pandas import (re-exported for tests). Streamlit only needs it
    transitively, but importing it here keeps the dependency explicit."""
    return pd.DataFrame()
