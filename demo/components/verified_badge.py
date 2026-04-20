"""AnswerVerify verdict badge."""

from __future__ import annotations

import html


_VERDICT_STYLE = {
    "pass":    ("#4CAF50", "VERIFIED",   "NLI entailment confirmed the answer is supported by the citations."),
    "partial": ("#FFC107", "PARTIAL",    "Parts of the answer are supported; others could not be verified."),
    "fail":    ("#F44336", "UNVERIFIED", "NLI verification did not find support for the answer in the citations."),
    # `skipped` is shown as a subtle confidence shortcut, not a warning.
    "skipped": ("#4CAF50", "HIGH CONFIDENCE", "Confidence was high enough to skip extra NLI verification."),
}


def render_verified_badge(verdict: str | None) -> None:
    import streamlit as st  # type: ignore

    if verdict is None:
        return

    key = verdict.lower()
    # Unknown verdicts: don't pollute the UI with raw internal strings.
    if key not in _VERDICT_STYLE:
        return

    color, label, tooltip = _VERDICT_STYLE[key]
    st.markdown(
        f'<span title="{html.escape(tooltip)}" '
        f'style="background:{color};color:white;border-radius:4px;'
        f'padding:2px 8px;font-weight:600;font-size:80%;">{label}</span>',
        unsafe_allow_html=True,
    )
