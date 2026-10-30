"""
Streamlit frontend - Google.com style search interface for the Agentic AI RAG system.

Connects to the FastAPI /query endpoint. Simple, centred search box, results
with citations and guardrail status.

Run with:  streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import requests
import streamlit as st

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
QUERY_ENDPOINT = f"{API_BASE}/query/query"
INGEST_ENDPOINT = f"{API_BASE}/ingest/ingest"
HEALTH_ENDPOINT = f"{API_BASE}/health"

REQUEST_TIMEOUT = 60  # seconds


# ----------------------------------------------------------------------
# Page config (Google-like minimal style)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="News Search",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; max-width: 720px; }
    .stTextInput > div > div > input {
        font-size: 1.15rem;
        padding: 0.55rem 1rem;
    }
    .title { font-size: 2.6rem; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0.1rem; margin-top: 1.6rem; }
    .title emoji { vertical-align: middle; line-height: 1; }
    .subtitle { color: #666; font-size: 1rem; margin-bottom: 1.6rem; }
    .answer-card { border: 1px solid #e5e7ea; border-radius: 10px; padding: 1rem 1.2rem; margin-top: 1rem; background: #fafafa; }
    .citation { font-size: 0.85rem; color: #555; margin-top: 0.5rem; border-top: 1px solid #eee; padding-top: 0.5rem; }
    .meta { font-size: 0.8rem; color: #888; margin-top: 0.6rem; }
    .guardrail-bad { color: #c0392b; font-weight: 600; }
    .guardrail-good { color: #1e8449; font-weight: 600; }
    .stButton>button { width: 100%; font-size: 1rem; padding: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def check_health() -> bool:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run_query(query: str, top_k: int = 5, use_guardrails: bool = True) -> Dict[str, Any]:
    payload = {
        "query": query,
        "session_id": str(st.session_state.session_id) if st.session_state.session_id else None,
        "top_k": top_k,
        "use_guardrails": use_guardrails,
    }
    r = requests.post(QUERY_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def run_ingest(reset: bool = False) -> Dict[str, Any]:
    r = requests.post(INGEST_ENDPOINT, params={"reset": reset}, timeout=120)
    r.raise_for_status()
    return r.json()



# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.markdown('<div class="title"><span class="emoji">🔍</span> News Search</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Agentic AI RAG over your news PDFs</div>', unsafe_allow_html=True)

healthy = check_health()
if healthy:
    st.markdown('<span class="guardrail-good">●</span> Backend connected', unsafe_allow_html=True)
else:
    st.markdown('<span class="guardrail-bad">●</span> Backend not reachable — start FastAPI first', unsafe_allow_html=True)
    st.info("Start the backend with: `uv run uvicorn newpapersearch.api.main:app --reload`")

with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "Search news articles",
        placeholder="e.g. What were the top editorials this week?",
        label_visibility="collapsed",
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        submitted = st.form_submit_button("Search")
    with col2:
        ingest_btn = st.form_submit_button("Re-ingest PDFs")

if ingest_btn:
    with st.spinner("Indexing news PDFs into ChromaDB..."):
        try:
            result = run_ingest(reset=True)
            st.success(
                f"Ingestion complete: {result.get('documents_processed')} docs, "
                f"{result.get('chunks_indexed')} chunks indexed."
            )
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")

if submitted and query.strip():
    with st.spinner("Thinking..."):
        try:
            data = run_query(query.strip())
            st.session_state.session_id = str(data.get("session_id"))

            if not data.get("guardrail_passed", True):
                st.error(
                    f"<span class='guardrail-bad'>Guardrail blocked/sanitized:</span> "
                    f"{data.get('guardrail_message') or ''}",
                    icon="🚫",
                )
            else:
                st.markdown(
                    "<span class='guardrail-good'>✓ Guardrails passed</span>",
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="answer-card">', unsafe_allow_html=True)
            st.write(data.get("answer", "No answer generated."))
            st.markdown("</div>", unsafe_allow_html=True)

            agents = data.get("agents_used", []) or []
            latency = data.get("latency_ms", 0)
            val = data.get("validation", {}) or {}
            st.markdown(
                f'<div class="meta">'
                f"Agents: {', '.join(agents)} &nbsp;|&nbsp; "
                f"Latency: {latency:.0f} ms &nbsp;|&nbsp; "
                f"Grounded: {'Yes' if val.get('is_grounded') else 'No'} "
                f"({val.get('groundedness_score', 0):.2f})"
                f"</div>",
                unsafe_allow_html=True,
            )

            citations = data.get("citations", []) or []
            if citations:
                with st.expander("📚 Sources / Citations", expanded=False):
                    for i, c in enumerate(citations, 1):
                        page = f", page {c.get('page_number')}" if c.get("page_number") else ""
                        st.markdown(
                            f'<div class="citation">[{i}] {c.get("source_file")}{page} '
                            f"(relevance {c.get('relevance_score') or 0:.2f})<br>"
                            f"{c.get('snippet','')[:250]}...</div>",
                            unsafe_allow_html=True,
                        )

            st.session_state.history.insert(
                0,
                {
                    "query": query,
                    "answer": data.get("answer", ""),
                    "agents": agents,
                    "latency": latency,
                },
            )
        except Exception as exc:
            st.error(f"Query failed: {exc}")

if st.session_state.history:
    with st.expander("🕒 Recent queries", expanded=False):
        for item in st.session_state.history[:10]:
            st.markdown(
                f"**Q:** {item['query']}<br>"
                f"<small>A: {item['answer'][:200]}...</small> "
                f"({item['agents']}, {item['latency']:.0f} ms)",
                unsafe_allow_html=True,
            )
            st.divider()

st.markdown("---")
st.caption(
    "Agentic AI RAG System · LangGraph + ChromaDB + Groq · "
    "Backend: FastAPI + SQLite · Frontend: Streamlit"
)
