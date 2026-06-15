"""Ask The Lab — grounded synthesis via the Research Synthesizer (NS-3 body).

Calls ``lab.synthesizer.synthesize`` with the user's question and renders the
answer + citation list. The synthesizer runs the Lab Agent Runtime with
web_search / arxiv_search / github_search / fetch_url; all citations are URLs
the agent actually fetched (grounded, not hallucinated).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `lib` importable when streamlit launches us from any cwd.
_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

st.set_page_config(page_title="Ask The Lab", layout="wide")
st.title("Ask The Lab")
st.caption(
    "Calls the Research Synthesizer (NS-3 body) for a grounded synthesis. "
    "Every citation is a URL the agent actually fetched — not model-hallucinated."
)

question = st.text_area("Question", height=100)

col_run, col_cfg = st.columns([3, 1])
with col_cfg:
    model = st.text_input("Model", value="qwen3-4b-ft-toolcall-q4-latest")
    max_tool_calls = st.number_input("Max tool calls", min_value=1, max_value=32, value=16)

with col_run:
    run_btn = st.button("Synthesize", type="primary")

if run_btn:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Synthesizing (may take up to 90 s)..."):
            try:
                from lab.synthesizer import synthesize

                result = synthesize(
                    question=question,
                    model=model,
                    max_tool_calls=int(max_tool_calls),
                )
            except Exception as e:
                st.error(f"Synthesizer error: {e}")
                st.exception(e)
            else:
                answer = result.get("answer") or ""
                citations: list[str] = result.get("citations") or []
                tool_calls: int = result.get("tool_calls", 0)
                stop: str = result.get("stop", "")

                st.markdown(answer)

                if citations:
                    st.subheader("Grounded citations")
                    for url in citations:
                        st.markdown(f"- [{url}]({url})")
                else:
                    st.caption("No URLs fetched during synthesis.")

                st.caption(f"tool_calls={tool_calls}  stop={stop}  model={model}")
