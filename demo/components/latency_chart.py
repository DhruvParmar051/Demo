"""Per-module latency breakdown bar chart."""

from __future__ import annotations

from typing import Any


def render_latency_breakdown(response: dict[str, Any]) -> None:
    import streamlit as st  # type: ignore
    import pandas as pd  # type: ignore

    tool_calls = response.get("tool_calls", []) or []
    by_module: dict[str, float] = {}
    for tc in tool_calls:
        name = tc.get("tool_name", "other")
        by_module[name] = by_module.get(name, 0.0) + float(tc.get("latency_ms", 0.0))

    total = float(response.get("latency_ms", 0.0))
    ttft = response.get("ttft_ms")

    if by_module:
        df = pd.DataFrame(
            [{"module": k, "latency_ms": v} for k, v in by_module.items()]
        ).sort_values("latency_ms", ascending=True)
        st.bar_chart(df.set_index("module"))

    col1, col2 = st.columns(2)
    col1.metric("Total latency", f"{total:.0f} ms")
    if ttft is not None:
        col2.metric("TTFT", f"{float(ttft):.0f} ms")
