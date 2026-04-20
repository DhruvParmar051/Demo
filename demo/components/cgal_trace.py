"""Render the CGAL per-iteration trace: confidence bars + tool calls."""

from __future__ import annotations

from typing import Any


def render_cgal_trace(response: dict[str, Any]) -> None:
    import streamlit as st  # type: ignore
    import pandas as pd  # type: ignore

    st.subheader("CGAL trace")
    tool_calls = response.get("tool_calls", []) or []
    iters: dict[int, dict[str, Any]] = {}
    for tc in tool_calls:
        i = int(tc.get("iteration", 0))
        iters.setdefault(i, {"iteration": i, "confidence": None, "tools": []})
        if tc.get("confidence_after") is not None:
            iters[i]["confidence"] = tc["confidence_after"]
        iters[i]["tools"].append(tc.get("tool_name", ""))

    if iters:
        df = pd.DataFrame(
            [
                {"iteration": r["iteration"],
                 "confidence": r["confidence"] or 0.0}
                for r in sorted(iters.values(), key=lambda r: r["iteration"])
            ]
        )
        st.bar_chart(df.set_index("iteration"))

    st.write("Tool call sequence:")
    for tc in tool_calls:
        st.write(
            f"- iter={tc.get('iteration', 0)} "
            f"{tc.get('tool_name')} "
            f"({tc.get('latency_ms', 0):.0f}ms)"
        )
