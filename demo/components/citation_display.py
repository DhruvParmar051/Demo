"""Render answer text with colored citation spans and a sources panel."""

from __future__ import annotations

import html
import os
import re
from typing import Any

_CITATION_RE = re.compile(r"\[([a-f0-9]{6,32}):(\d+)-(\d+)\]")

_PALETTE = [
    "#FFE082", "#B2EBF2", "#C5E1A5", "#F8BBD0", "#D1C4E9",
    "#FFCCBC", "#B3E5FC", "#FFF59D", "#DCE775", "#E1BEE7",
]


def render_citations(answer: str, citations: list[dict[str, Any]]) -> str:
    """Convert an answer with `[doc_id:start-end]` markers to HTML.

    Each unique doc_id gets its own pastel background color; the marker
    text itself renders as a small subscript badge so the sentence stays
    readable.
    """
    cite_by_doc: dict[str, dict[str, Any]] = {}
    for c in citations:
        cite_by_doc.setdefault(c["doc_id"], c)

    colors: dict[str, str] = {}
    for i, doc_id in enumerate(sorted(cite_by_doc)):
        colors[doc_id] = _PALETTE[i % len(_PALETTE)]

    def _repl(m: re.Match) -> str:
        doc_id = m.group(1)
        span = f"{m.group(2)}-{m.group(3)}"
        color = colors.get(doc_id, "#EEEEEE")
        title = html.escape(cite_by_doc.get(doc_id, {}).get("cited_text", "")[:240])
        return (
            f'<span style="background:{color};border-radius:3px;'
            f'padding:0 3px;font-size:85%;" title="{title}">'
            f'[{doc_id[:6]}:{span}]</span>'
        )

    escaped = html.escape(answer)
    # Unescape only the citation markers so the regex matches.
    escaped = escaped.replace("&#91;", "[").replace("&#93;", "]")
    return _CITATION_RE.sub(_repl, escaped)


def _source_label(source: str) -> str:
    """Human-readable label for a source path or URL."""
    if not source:
        return "unknown source"
    if source.startswith(("http://", "https://")):
        return source
    return os.path.basename(source) or source


def render_sources_panel(citations: list[dict[str, Any]]) -> None:
    """Render an expandable 'Sources' list below the answer.

    Complements the inline markers in `render_citations`. Mobile-friendly
    (no hover required). Deduplicates by (doc_id, chunk_id).
    """
    import streamlit as st  # type: ignore

    if not citations:
        return

    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for c in citations:
        key = (c.get("doc_id", ""), c.get("chunk_id", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    # Assign the same palette colors used inline so badges match entries.
    color_map: dict[str, str] = {}
    for i, doc_id in enumerate(sorted({c.get("doc_id", "") for c in unique})):
        color_map[doc_id] = _PALETTE[i % len(_PALETTE)]

    with st.expander(f"Sources ({len(unique)})", expanded=False):
        for idx, c in enumerate(unique, start=1):
            doc_id = c.get("doc_id", "")
            color = color_map.get(doc_id, "#EEEEEE")
            label = _source_label(c.get("source", ""))
            page = c.get("page_number")
            url = c.get("source_url")
            verified = c.get("verified")

            verdict_badge = ""
            if verified is True:
                verdict_badge = (
                    '<span style="background:#4CAF50;color:white;'
                    'border-radius:3px;padding:0 5px;font-size:75%;'
                    'margin-left:6px;">verified</span>'
                )
            elif verified is False:
                verdict_badge = (
                    '<span style="background:#F44336;color:white;'
                    'border-radius:3px;padding:0 5px;font-size:75%;'
                    'margin-left:6px;">unverified</span>'
                )

            header_bits = [f"<b>[{idx}]</b>"]
            if url:
                link_text = html.escape(label)
                header_bits.append(
                    f'<a href="{html.escape(url)}" target="_blank" '
                    f'rel="noopener noreferrer">{link_text}</a>'
                )
            else:
                header_bits.append(html.escape(label))
            if page is not None:
                header_bits.append(f"(p.&nbsp;{int(page)})")
            header_bits.append(
                f'<span style="background:{color};border-radius:3px;'
                f'padding:0 4px;font-size:75%;margin-left:6px;">'
                f'{html.escape(doc_id[:8])}</span>'
            )
            if verdict_badge:
                header_bits.append(verdict_badge)

            st.markdown(" ".join(header_bits), unsafe_allow_html=True)

            cited = (c.get("cited_text") or "").strip()
            if cited:
                st.markdown(
                    f'<blockquote style="border-left:3px solid {color};'
                    f'margin:4px 0 10px 0;padding:2px 10px;color:#333;'
                    f'font-size:90%;">{html.escape(cited)}</blockquote>',
                    unsafe_allow_html=True,
                )
