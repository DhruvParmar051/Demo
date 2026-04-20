"""Show decomposition state + sub-queries."""

from __future__ import annotations


def render_decomp_indicator(decomposed: bool, sub_queries: list[str]) -> None:
    import streamlit as st  # type: ignore

    if not decomposed:
        return
    st.info("Decomposed into sub-queries:")
    for i, q in enumerate(sub_queries, 1):
        st.write(f"{i}. {q}")
