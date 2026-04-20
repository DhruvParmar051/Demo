"""Streamlit file-upload panel for AegisRAG.

Wire it into the demo sidebar with one line::

    from demo.upload_panel import render_upload_panel
    render_upload_panel(api_url)

The panel accepts PDF / DOCX / TXT / MD files, POSTs them to the API's
``/ingest`` endpoint, and shows chunk counts + rejected files. After a
successful ingest, the existing chat pane automatically queries the
now-augmented KB -- no extra wiring needed because the M5 pipeline reads
the same ChromaDB + BM25 stores.
"""

from __future__ import annotations

from typing import Iterable
import streamlit as st
import requests

_ALLOWED = ("pdf", "docx", "txt", "md")


def render_upload_panel(
    api_url: str,
    *,
    key_prefix: str = "aegis_upload",
    header: str = "Upload Evidence",
    help_text: str | None = None,
    timeout_seconds: int = 120,
) -> dict | None:
    """Render the uploader and return the API response dict on success.

    Safe to call from both the sidebar and the main pane. Returns None when
    the user has not clicked the submit button yet.
    """


    st.markdown(f"### {header}")
    st.caption(help_text or f"Accepted: {', '.join(_ALLOWED).upper()}. Max 25 MB per file.")

    uploaded = st.file_uploader(
        "Drop documents here",
        type=list(_ALLOWED),
        accept_multiple_files=True,
        key=f"{key_prefix}_uploader",
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        submit = st.button(
            "Ingest",
            key=f"{key_prefix}_submit",
            disabled=not uploaded,
            use_container_width=True,
        )
    with col_b:
        clear = st.button(
            "Clear session",
            key=f"{key_prefix}_clear",
            use_container_width=True,
        )

    if clear:
        st.session_state.pop(f"{key_prefix}_last_response", None)
        st.toast("Cleared last upload response.", icon="🧹")

    last = st.session_state.get(f"{key_prefix}_last_response")

    if submit and uploaded:
        files_payload = _build_multipart(uploaded)
        endpoint = api_url.rstrip("/") + "/ingest"
        with st.spinner(f"Indexing {len(uploaded)} file(s) ..."):
            try:
                response = requests.post(endpoint, files=files_payload, timeout=timeout_seconds)
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                st.error(f"Ingest failed: {exc}")
                return None
            except ValueError:
                st.error("Ingest endpoint returned invalid JSON.")
                return None

        st.session_state[f"{key_prefix}_last_response"] = data
        last = data

        chunks = data.get("chunks_added", 0)
        accepted = data.get("files_accepted", [])
        rejected = data.get("files_rejected", [])
        if chunks:
            st.success(f"Indexed {chunks} chunk(s) from {len(accepted)} file(s).")
        elif accepted:
            st.info(f"Accepted {len(accepted)} file(s) but no new chunks were added (likely duplicates).")
        if rejected:
            st.warning(f"Rejected {len(rejected)} file(s). Expand below for details.")

    if last:
        with st.expander("Last ingest response", expanded=False):
            import json as _json
            st.code(_json.dumps(last, indent=2), language="json")

    return last


def _build_multipart(uploaded: Iterable) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Translate Streamlit UploadedFile objects into requests-multipart form.

    Uses the form-field name 'files' repeatedly so FastAPI's
    ``List[UploadFile]`` sees every file.
    """
    mime_map = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain",
        "md": "text/markdown",
    }
    payload: list[tuple[str, tuple[str, bytes, str]]] = []
    for f in uploaded:
        name = getattr(f, "name", "upload")
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        mime = mime_map.get(ext, "application/octet-stream")
        try:
            f.seek(0)
        except Exception:  # pragma: no cover - UploadedFile always supports seek
            pass
        payload.append(("files", (name, f.read(), mime)))
    return payload
