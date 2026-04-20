"""Consume SSE events from the AegisRAG API and update the chat container."""

from __future__ import annotations

import json
from typing import Any, Iterable


def parse_sse_stream(raw: Iterable[bytes]) -> Iterable[dict[str, Any]]:
    """Yield dicts of the form {event, data} from a raw byte stream."""
    buf = ""
    for chunk in raw:
        if isinstance(chunk, (bytes, bytearray, memoryview)):
            chunk = bytes(chunk).decode("utf-8", errors="replace")
        else:
            chunk = str(chunk)
        buf += chunk
        while "\n\n" in buf:
            frame, buf = buf.split("\n\n", 1)
            event = "message"
            data_lines: list[str] = []
            for line in frame.splitlines():
                if line.startswith("event:"):
                    event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].lstrip())
            data_str = "\n".join(data_lines)
            try:
                data = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                data = data_str
            yield {"event": event, "data": data}


def render_streaming_chat(
    response_stream: Iterable[bytes],
    container: Any = None,
) -> dict[str, Any]:
    """Consume the SSE stream, updating a Streamlit container with tokens.

    Returns the final response dict emitted on the ``done`` event (or an
    empty dict if the stream ended without one).
    """
    try:
        import streamlit as st  # type: ignore
    except ImportError:
        st = None

    placeholder = container.empty() if container is not None and st is not None else None
    answer_buf = ""
    final: dict[str, Any] = {}
    citations: list[dict[str, Any]] = []

    for evt in parse_sse_stream(response_stream):
        etype = evt["event"]
        data = evt["data"]
        if etype == "token" and isinstance(data, dict) and "text" in data:
            answer_buf += data["text"]
            if placeholder is not None:
                placeholder.markdown(answer_buf)
        elif etype == "citation" and isinstance(data, dict):
            citations.append(data)
        elif etype == "done" and isinstance(data, dict):
            final = data
            break
        elif etype == "error":
            final = {"error": data}
            break

    if not final:
        final = {"answer": answer_buf, "citations": citations}
    return final
