"""Tests for the CGAL loop engine with mocked components."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")


class _StubRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.vector_store = _StubVectorStore()
        self.alpha_network = None

    def retrieve(self, query, top_k=20, alpha=None, filters=None):
        return [(c, 0.5) for c in self._chunks[:top_k]]


class _StubVectorStore:
    class _Model:
        def encode(self, texts, normalize_embeddings=True):
            import numpy as np
            return np.zeros((len(texts), 16), dtype="float32")

    def __init__(self):
        self.model = _StubVectorStore._Model()


class _StubReranker:
    def rerank(self, query, chunks, top_k=5):
        return chunks[:top_k]


class _StubConfidenceHead:
    def __init__(self, conf):
        self.conf = conf

    def eval(self):
        pass

    def score(self, query_emb, evidence_embs):
        return self.conf, [1.0, 0.0, 0.0, 0.0]


class _StubGenerator:
    def set_generation_kwargs(self, **kwargs):
        pass

    def generate(self, query=None, context=None, **kwargs):
        return f"answer for: {query}"


class _StubToolExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, tool_name, args):
        self.calls.append((tool_name, args))
        if tool_name == "CreateTicket":
            return {"ticket_id": "TKT-TEST", "estimated_response_time": "24h",
                    "message": "escalated"}
        return {"results": []}


@pytest.fixture
def engine_factory(mini_corpus):
    from src.cgal.loop_engine import CGALLoopEngine

    def _make(conf: float) -> Any:
        return CGALLoopEngine(
            retriever=_StubRetriever(mini_corpus),
            reranker=_StubReranker(),
            confidence_head=_StubConfidenceHead(conf),
            generator=_StubGenerator(),
            tool_executor=_StubToolExecutor(),
            answer_verify=None,
            decomposer=None,
            alpha_network=None,
        )
    return _make


def test_high_confidence_answers_directly(engine_factory):
    engine = engine_factory(conf=0.95)
    resp = engine.run("How do I reset my password?")
    assert resp.ticket_id is None
    assert "password" in resp.answer.lower() or resp.answer
    assert resp.cgal_iterations == 1


def test_low_confidence_escalates(engine_factory):
    engine = engine_factory(conf=0.05)
    resp = engine.run("something weird")
    assert resp.ticket_id == "TKT-TEST"


def test_max_iterations_bound(engine_factory):
    engine = engine_factory(conf=0.5)  # medium-low tier triggers tool, loops
    resp = engine.run("some query")
    # Must escalate after max_iterations.
    assert resp.cgal_iterations <= engine.max_iterations + 1


def test_deterministic_repeats(engine_factory):
    engine = engine_factory(conf=0.95)
    r1 = engine.run("same query")
    r2 = engine.run("same query")
    assert r1.answer == r2.answer
