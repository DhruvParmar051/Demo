"""
Server-Sent Events (SSE) helpers for the FastAPI app.

Event taxonomy used across the ``/query/stream`` endpoint:

    token         -> a text chunk produced by the generator
    citation      -> a ``Citation`` dict
    tool_call     -> a ``ToolCall`` dict
    verify_start  -> signals the async NLI verifier has begun
    verify_result -> final NLI verdict
    done          -> final ``QueryResponse`` dict
    error         -> error payload ({message, type})
"""

from __future__ import annotations

import json
from typing import Any

EVENT_TOKEN = "token"
EVENT_CITATION = "citation"
EVENT_TOOL_CALL = "tool_call"
EVENT_VERIFY_START = "verify_start"
EVENT_VERIFY_RESULT = "verify_result"
EVENT_DONE = "done"
EVENT_ERROR = "error"

EVENT_NAMES = [
    EVENT_TOKEN,
    EVENT_CITATION,
    EVENT_TOOL_CALL,
    EVENT_VERIFY_START,
    EVENT_VERIFY_RESULT,
    EVENT_DONE,
    EVENT_ERROR,
]


def format_sse(event: str, data: Any) -> str:
    """Format an SSE frame as ``"event: {name}\\ndata: {json}\\n\\n"``."""
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False, default=str)
    # Split data on newlines -- SSE data lines cannot contain raw \n.
    lines = data.split("\n")
    data_block = "\n".join(f"data: {line}" for line in lines)
    return f"event: {event}\n{data_block}\n\n"
