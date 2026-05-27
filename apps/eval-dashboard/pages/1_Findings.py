"""Findings explorer.

Filterable list of F-NNN findings from the Postgres `findings` table.
Selecting a row loads the markdown body from disk + renders the
depends_on graph from frontmatter using streamlit-agraph.
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

st.set_page_config(page_title="Findings", layout="wide")
st.title("Findings")

LAB_HOME = Path("/data/lab/code")
FINDINGS_DIR = LAB_HOME / "docs" / "findings"


@st.cache_data(ttl=60)
def _findings_df():
    return db.pg_query(
        """
        SELECT finding_id, slug, claim, confidence, status,
               source_exp, doc_path, created_at, superseded_by
        FROM findings
        ORDER BY finding_id DESC
        """
    )


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body). Tiny non-yaml parser for safety."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    fm_text, body = parts[1], parts[2].lstrip("\n")
    fm: dict[str, object] = {}
    current_key: str | None = None
    list_buf: list[dict] = []
    for raw in fm_text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("- "):
            item = line[2:].strip()
            if ":" in item:
                k, v = item.split(":", 1)
                list_buf.append({k.strip(): v.strip()})
            else:
                list_buf.append({"value": item})
            continue
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$", line)
        if m:
            if current_key and list_buf:
                fm[current_key] = list_buf
                list_buf = []
            key, val = m.group(1), m.group(2).strip()
            current_key = key
            fm[key] = val if val else None
    if current_key and list_buf:
        fm[current_key] = list_buf
    return fm, body


def _agraph(deps: list[dict], center: str):
    try:
        from streamlit_agraph import Config, Edge, Node, agraph
    except Exception:
        st.caption("streamlit-agraph unavailable; falling back to list.")
        for d in deps:
            st.markdown(f"- **{d.get('kind', '?')}** -> `{d.get('target', '?')}`")
        return
    nodes = [Node(id=center, label=center, size=22, color="#7aa2f7")]
    edges = []
    for d in deps:
        target = str(d.get("target", "?"))
        kind = str(d.get("kind", "?"))
        nodes.append(Node(id=target, label=target, size=14, color="#9ece6a"))
        edges.append(Edge(source=center, target=target, label=kind))
    config = Config(width=720, height=420, directed=True, physics=True, hierarchical=False)
    agraph(nodes=nodes, edges=edges, config=config)


df = _findings_df()
if df.empty or "_error" in df.columns:
    st.error(
        f"Could not load findings: {df['_error'].iloc[0] if '_error' in df.columns else 'empty'}"
    )
    st.stop()

# Filters
with st.sidebar:
    st.header("Filters")
    statuses = sorted(df["status"].dropna().unique().tolist())
    confs = sorted(df["confidence"].dropna().unique().tolist())
    sel_status = st.multiselect("Status", statuses, default=statuses)
    sel_conf = st.multiselect("Confidence", confs, default=confs)
    name_q = st.text_input("Search claim/slug", "")

filtered = df[
    df["status"].isin(sel_status)
    & df["confidence"].isin(sel_conf)
    & (
        df["slug"].str.contains(name_q, case=False, na=False)
        | df["claim"].str.contains(name_q, case=False, na=False)
    )
]

st.caption(f"{len(filtered)} of {len(df)} findings")
st.dataframe(
    filtered[["slug", "claim", "confidence", "status", "created_at"]],
    hide_index=True,
    use_container_width=True,
)

# Picker
slugs = filtered["slug"].tolist()
if not slugs:
    st.info("No findings match the filters.")
    st.stop()

picked = st.selectbox("Open finding", slugs)
row = filtered[filtered["slug"] == picked].iloc[0]
doc_path = row["doc_path"] or ""
fpath = LAB_HOME / doc_path if doc_path else None

st.subheader(f"{row['slug']} - {row['confidence']} confidence, status={row['status']}")
st.caption(row["claim"])

if fpath and fpath.is_file():
    text = fpath.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Body")
        st.markdown(body)
    with col2:
        st.markdown("### Frontmatter")
        st.json(fm, expanded=False)
        deps = fm.get("depends_on")
        if isinstance(deps, list) and deps:
            st.markdown("### Dependency graph")
            _agraph(deps, str(row["slug"]))
else:
    st.warning(f"Doc file not found on disk: {fpath}")
