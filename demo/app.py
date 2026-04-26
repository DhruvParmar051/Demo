"""
AegisRAG - Streamlit demo.

Launch with:

    streamlit run demo/app.py -- --api http://localhost:8000

Talks to the FastAPI backend via HTTP / SSE. When the backend is down the
sidebar shows an error and the main panel falls back to a stub response.

Features:
- Chat history with Streamlit's native chat UI (chronological, persistent).
- User document upload: files are ingested into a session-specific collection
  and queried in isolation via /query/user_docs (no mixing with system KB).
- Search mode toggle: "System KB" vs "My Documents".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Any

import requests  # type: ignore
import streamlit as st  # type: ignore

# Allow importing src.* when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.components.cgal_trace import render_cgal_trace  # noqa: E402
from demo.components.citation_display import (  # noqa: E402
    render_citations,
    render_sources_panel,
)
from demo.components.decomp_indicator import render_decomp_indicator  # noqa: E402
from demo.components.latency_chart import render_latency_breakdown  # noqa: E402
from demo.components.streaming_chat import render_streaming_chat  # noqa: E402
from demo.components.verified_badge import render_verified_badge  # noqa: E402
from demo.upload_panel import render_upload_panel  # noqa: E402


def _get_api_url() -> str:
    """Resolve the API base URL from CLI arg, env var, or default."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--api", dest="api", default=None)
    known, _ = parser.parse_known_args()
    if known.api:
        return known.api
    return os.environ.get("AEGISRAG_API", "http://localhost:8000")


def _init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("api_url", _get_api_url())
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("user_docs_ingested", False)


def _call_query(
    api: str,
    query: str,
    tag: str,
    stream: bool,
    search_mode: str = "System KB",
    collection_id: str | None = None,
) -> dict[str, Any]:
    payload = {"query": query, "model_tag": tag}
    try:
        if search_mode == "My Documents" and collection_id:
            # Dense-only retrieval against the user's personal collection.
            r = requests.post(
                f"{api}/query/user_docs",
                params={"collection_id": collection_id},
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            return r.json()

        if stream:
            with requests.post(
                f"{api}/query/stream", json=payload, stream=True, timeout=120
            ) as r:
                r.raise_for_status()
                container = st.empty()
                return render_streaming_chat(
                    r.iter_content(chunk_size=None), container=container
                )
        else:
            r = requests.post(f"{api}/query", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()
    except requests.RequestException as exc:
        return {"error": str(exc), "answer": "", "citations": [], "tool_calls": []}


def main() -> None:
    st.set_page_config(page_title="AegisRAG", layout="wide")
    _init_state()

    api_url = st.session_state["api_url"]
    collection_id = f"user_{st.session_state['session_id']}"

    # ------------------------------------------------------------------ Sidebar
    with st.sidebar:
        st.title("AegisRAG")
        st.session_state["api_url"] = st.text_input(
            "API URL", value=st.session_state["api_url"]
        )
        api_url = st.session_state["api_url"]

        model_tag = st.selectbox(
            "Model",
            options=["m5", "m4", "m3", "m2", "m1", "b3", "b2", "b1"],
            index=0,
            help="M5 is the full system; b1-b3 are single-pass baselines.",
        )
        domain = st.selectbox(
            "Domain filter",
            options=["All", "DMV", "VA", "StudentAid", "SSA"],
            index=0,
        )
        stream = st.toggle("Stream tokens", value=True)
        st.caption("Temperature is locked to 0 for determinism.")
        st.slider("Temperature", 0.0, 1.0, 0.0, disabled=True)

        # Backend health check
        try:
            r = requests.get(f"{api_url}/health", timeout=2)
            if r.ok:
                st.success(f"Backend OK ({r.json().get('device', '?')})")
            else:
                st.error(f"Backend error: {r.status_code}")
        except requests.RequestException as exc:
            st.warning(f"Backend unreachable: {exc}")

        # ---- Document upload (session-isolated) ---------------------------
        st.divider()
        ingest_result = render_upload_panel(
            api_url,
            collection_id=collection_id,
            header="Upload Your Documents",
            help_text="Uploads are kept private to your session.",
        )
        if ingest_result and ingest_result.get("chunks_added", 0) > 0:
            st.session_state["user_docs_ingested"] = True

        search_mode = st.radio(
            "Search mode",
            options=["System KB", "My Documents"],
            index=0,
            disabled=not st.session_state["user_docs_ingested"],
            help=(
                "'My Documents' answers only from files you uploaded this session. "
                "Upload at least one document to enable."
            ),
        )

        if st.button("Clear chat history"):
            st.session_state["history"] = []
            st.rerun()

    # ------------------------------------------------------------------ Chat UI

    # Render existing turns chronologically.
    for turn in st.session_state["history"]:
        with st.chat_message("user"):
            mode_label = (
                " *(My Documents)*"
                if turn.get("search_mode") == "My Documents"
                else ""
            )
            st.markdown(f"{turn['query']}{mode_label}")

        with st.chat_message("assistant"):
            resp = turn["response"]
            if "error" in resp and resp["error"]:
                st.error(resp["error"])
            else:
                html = render_citations(
                    resp.get("answer", ""), resp.get("citations", [])
                )
                st.markdown(html, unsafe_allow_html=True)
                render_verified_badge(resp.get("verify_verdict"))
                render_decomp_indicator(
                    bool(resp.get("decomposed", False)),
                    list(resp.get("sub_queries", [])),
                )
                with st.expander("Sources & details", expanded=False):
                    render_sources_panel(resp.get("citations", []))
                    render_cgal_trace(resp)
                    render_latency_breakdown(resp)
                    conf = resp.get("confidence", 0.0)
                    st.metric("Confidence", f"{float(conf):.2f}")
                    if resp.get("ticket_id"):
                        st.warning(f"Escalated: ticket {resp['ticket_id']}")

    # Persistent chat input — always visible at the bottom.
    user_input = st.chat_input("Ask AegisRAG…")
    if user_input:
        response = _call_query(
            api_url,
            user_input,
            model_tag,
            stream,
            search_mode=search_mode,
            collection_id=collection_id,
        )
        st.session_state["history"].append(
            {
                "query": user_input,
                "response": response,
                "model_tag": model_tag,
                "domain": domain,
                "search_mode": search_mode,
            }
        )
        st.rerun()

    # Debug expander for the last response (development aid).
    if st.session_state["history"]:
        with st.expander("Raw JSON of last response", expanded=False):
            st.code(
                json.dumps(
                    st.session_state["history"][-1]["response"],
                    indent=2,
                    default=str,
                ),
                language="json",
            )


if __name__ == "__main__":
    main()
