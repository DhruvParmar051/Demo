"""Tests for SSE formatting and stream parsing."""

from __future__ import annotations

import json

from demo.components.streaming_chat import parse_sse_stream
from src.serving.sse import format_sse


def test_format_sse_basic():
    frame = format_sse("token", {"text": "hello"})
    assert frame.startswith("event: token\n")
    assert "data: " in frame
    assert frame.endswith("\n\n")


def test_format_sse_string_data():
    frame = format_sse("error", "boom")
    assert "data: boom" in frame


def test_parse_sse_roundtrip():
    frames = [
        format_sse("token", {"text": "hello "}).encode("utf-8"),
        format_sse("token", {"text": "world"}).encode("utf-8"),
        format_sse("done", {"answer": "hello world"}).encode("utf-8"),
    ]
    events = list(parse_sse_stream(frames))
    assert [e["event"] for e in events] == ["token", "token", "done"]
    assert events[-1]["data"]["answer"] == "hello world"
