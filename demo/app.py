"""
AegisRAG - Streamlit demo.

Launch with:

    streamlit run demo/app.py -- --api http://localhost:8000

Talks to the FastAPI backend via HTTP / SSE. When the backend is down the
sidebar shows an error and the main panel falls back to a stub response.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def _call_query(api: str, query: str, tag: str, stream: bool) -> dict[str, Any]:
    payload = {"query": query, "model_tag": tag}
    try:
        if stream:
            with requests.post(
                f"{api}/query/stream", json=payload, stream=True, timeout=120
            ) as r:
                r.raise_for_status()
                container = st.empty()
                return render_streaming_chat(r.iter_content(chunk_size=None),
                                              container=container)
        else:
            r = requests.post(f"{api}/query", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()
    except requests.RequestException as exc:
        return {"error": str(exc), "answer": "", "citations": [], "tool_calls": []}


def main() -> None:
    st.set_page_config(page_title="AegisRAG", layout="wide")
    _init_state()

    with st.sidebar:
        st.title("AegisRAG")
        st.session_state["api_url"] = st.text_input(
            "API URL", value=st.session_state["api_url"]
        )
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
        st.slider("Temperature", 0.0, 0.0, 0.0, disabled=True)
        try:
            r = requests.get(f"{st.session_state['api_url']}/health", timeout=2)
            if r.ok:
                st.success(f"Backend OK ({r.json().get('device', '?')})")
            else:
                st.error(f"Backend error: {r.status_code}")
        except requests.RequestException as exc:
            st.warning(f"Backend unreachable: {exc}")

    left, right = st.columns([2, 1])

    with left:
        st.header("Ask AegisRAG")
        query = st.text_area("Question", height=120, key="query_box")
        go = st.button("Ask", type="primary")
        if go and query.strip():
            response = _call_query(
                st.session_state["api_url"], query, model_tag, stream
            )
            st.session_state["history"].append(
                {"query": query, "response": response, "model_tag": model_tag,
                 "domain": domain}
            )

        for turn in reversed(st.session_state["history"]):
            resp = turn["response"]
            st.markdown(f"**You ({turn['model_tag']}):** {turn['query']}")
            if "error" in resp:
                st.error(resp["error"])
                continue
            html = render_citations(resp.get("answer", ""),
                                     resp.get("citations", []))
            st.markdown(html, unsafe_allow_html=True)
            render_verified_badge(resp.get("verify_verdict"))
            render_decomp_indicator(
                bool(resp.get("decomposed", False)),
                list(resp.get("sub_queries", [])),
            )
            render_sources_panel(resp.get("citations", []))
            st.divider()

    with right:
        if st.session_state["history"]:
            latest = st.session_state["history"][-1]["response"]
            if "error" not in latest:
                render_cgal_trace(latest)
                st.divider()
                render_latency_breakdown(latest)
                st.divider()
                conf = latest.get("confidence", 0.0)
                st.metric("Confidence", f"{float(conf):.2f}")
                if latest.get("ticket_id"):
                    st.warning(f"Escalated: ticket {latest['ticket_id']}")

    # Debug raw response expander
    if st.session_state["history"]:
        with st.expander("Raw JSON of last response"):
            st.code(
                json.dumps(st.session_state["history"][-1]["response"],
                           indent=2, default=str),
                language="json",
            )


if __name__ == "__main__":
    main()
