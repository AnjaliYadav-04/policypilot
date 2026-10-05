"""PolicyPilot UI. Runs on Databricks Apps (PP_ENV=databricks) or locally (PP_ENV=local)."""

from __future__ import annotations

import html
import os
from typing import Any

import streamlit as st

from agent.companies import COMPANIES, company_label
from agent.config import load_settings

st.set_page_config(page_title="PolicyPilot", page_icon="📑", layout="centered")

EXAMPLES = [
    "What supply chain risks does Apple disclose?",
    "How does NVIDIA describe the impact of export controls?",
    "What cybersecurity risks does JPMorgan Chase describe?",
]
AUTO = "Detect from my question"

st.markdown(
    """
    <style>
      .pp-source { font-size: 0.9rem; color: var(--text-color); opacity: 0.85; }
      .pp-refusal { border-left: 3px solid #B54708; padding: 0.6rem 0.9rem; background: #FFF6ED; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Starting the agent…")
def _graph() -> Any:
    from agent.graph import get_graph

    return get_graph()


def _md_safe(text: str) -> str:
    """Escape '$' so dollar figures aren't rendered as LaTeX."""
    return text.replace("$", r"\$")


def _render_answer(msg: dict[str, Any]) -> None:
    if msg.get("refused"):
        st.markdown(f'<div class="pp-refusal">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(_md_safe(msg["content"]))

    citations = msg.get("citations") or []
    if citations:
        st.caption("Sources")
        for c in citations:
            title = f"[{c['n']}] {c.get('company', c.get('ticker'))}, 10-K filed {c.get('filing_date')}, {c.get('section')}"
            with st.expander(title):
                excerpt = c.get("text", "")
                st.markdown(
                    f'<div class="pp-source">{html.escape(excerpt[:1200])}{"…" if len(excerpt) > 1200 else ""}</div>',
                    unsafe_allow_html=True,
                )
                if c.get("source_url"):
                    st.markdown(f"[Open the filing on SEC EDGAR]({c['source_url']})")
                st.caption(f"Accession number {c.get('accession_number')}")

    if msg.get("trace"):
        with st.expander("How this answer was produced"):
            st.write(f"Search query: `{msg.get('search_query', '')}`")
            st.write(f"Company filter: {company_label(msg['ticker']) if msg.get('ticker') else 'none (all companies)'}")
            st.dataframe(msg["trace"], hide_index=True, use_container_width=True)
            if msg.get("refusal_details"):
                st.write("Verification notes:")
                for d in msg["refusal_details"]:
                    st.write(f"- {d}")


def _ask(question: str, ticker: str | None) -> dict[str, Any]:
    from agent.graph import run_agent

    result = run_agent(question, ticker=ticker, graph=_graph())
    return {
        "role": "assistant",
        "content": result["answer"],
        "citations": result["citations"],
        "refused": result["refused"],
        "refusal_details": result["refusal_details"],
        "trace": result["trace"],
        "ticker": result["ticker"],
        "search_query": result["search_query"],
    }


# ---- sidebar ---------------------------------------------------------------------------------
settings = load_settings()
with st.sidebar:
    st.subheader("Company")
    options = [AUTO] + [company_label(t) for t in COMPANIES]
    choice = st.selectbox("Limit answers to one filing", options, label_visibility="collapsed")
    picked = None if choice == AUTO else choice.rsplit("(", 1)[1].rstrip(")")

    st.caption(
        "Answers come only from each company's latest Form 10-K (Business, Risk Factors and MD&A). "
        "Every sentence is cited, and questions the filing can't answer are refused."
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    user = st.context.headers.get("X-Forwarded-Email") if hasattr(st, "context") else None
    st.caption(f"Backend: {settings.env}" + (f", signed in as {user}" if user else ""))

# ---- main ------------------------------------------------------------------------------------
st.title("PolicyPilot")
st.write("Ask what a company discloses in its 10-K. You get an answer with a numbered citation for every claim.")

if "messages" not in st.session_state:
    st.session_state.messages = []

pending = st.session_state.pop("pending", None)

if not st.session_state.messages and not pending:
    st.caption("Try one of these")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(_md_safe(msg["content"]))
        else:
            _render_answer(msg)

question = st.chat_input("Ask about a 10-K, e.g. What supply chain risks does Apple disclose?") or pending
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(_md_safe(question))
    with st.chat_message("assistant"):
        with st.spinner("Searching the filing and checking citations…"):
            try:
                reply = _ask(question, picked)
            except Exception as exc:  # surface backend errors instead of a blank screen
                st.error(
                    f"The agent couldn't finish: {exc}. "
                    + ("Check the app's resources and permissions." if os.environ.get("DATABRICKS_HOST") else "")
                )
                st.stop()
        _render_answer(reply)
    st.session_state.messages.append(reply)
