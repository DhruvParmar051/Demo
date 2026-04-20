"""Tests for ToolExecutor and AnswerVerify gating."""

from __future__ import annotations

from typing import Any

import pytest

from src.tools.executor import ToolExecutor
from src.tools.schemas import validate_tool_args


class _StubRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, top_k=20, filters=None):
        return [(c, 0.5) for c in self._chunks[:top_k]]


def test_validate_known_and_unknown_schemas():
    ok, errs = validate_tool_args("SearchKB", {"query": "hi"})
    assert ok, errs
    ok, errs = validate_tool_args("SearchKB", {})
    assert not ok

    ok, errs = validate_tool_args("UnknownTool", {})
    assert not ok


def test_search_kb_returns_results(mini_corpus):
    exe = ToolExecutor(retriever=_StubRetriever(mini_corpus), reranker=None)
    out = exe.execute("SearchKB", {"query": "password", "top_k": 3})
    assert "results" in out
    assert len(out["results"]) <= 3


def test_get_policy_exact_and_fuzzy():
    policy = {
        "refund_international": {"text": "See policy §5.", "source": "policy.md"}
    }
    exe = ToolExecutor(retriever=None, policy_index=policy)
    r = exe.execute("GetPolicy", {"section_id": "refund_international"})
    assert r["found"] is True

    r = exe.execute("GetPolicy", {"section_id": "refund"})
    assert r["found"] is True
    assert r.get("fuzzy") is True


def test_create_ticket_persists(tmp_path):
    db = tmp_path / "audit.sqlite"
    exe = ToolExecutor(retriever=None, ticket_store_path=db)
    r = exe.execute(
        "CreateTicket",
        {"query": "urgent", "category": "billing", "severity": "high"},
    )
    assert r["ticket_id"].startswith("TKT-")
    assert r["estimated_response_time"] == "8h"
    tickets = exe.list_tickets()
    assert any(t["ticket_id"] == r["ticket_id"] for t in tickets)


def test_answer_verify_skips_high_confidence():
    from src.tools.answer_verify import AnswerVerify

    av = AnswerVerify.__new__(AnswerVerify)
    av.skip_confidence = 0.85
    av.entail_threshold = 0.7
    av.pass_threshold = 0.8
    av.partial_threshold = 0.5
    av._nli = None
    av._nlp = None

    out = av.verify("Some answer.", [], confidence=0.95)
    assert out["verdict"] == "skipped"
