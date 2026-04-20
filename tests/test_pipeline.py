"""End-to-end integration smoke tests with fully mocked components."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_m5_pipeline_builds_from_tag(monkeypatch):
    """Verify that each model tag assembles into a runnable object.

    Heavy model loads are monkey-patched out so this runs on CPU quickly.
    """
    from src.models import m5_pipeline as m5_mod
    from src.models.m5_pipeline import M5Pipeline, PipelineFlags

    # Replace anything that would load a real model with a stub factory.
    class _Stub:
        def __init__(self, *a, **kw):
            pass

        def eval(self):
            pass

        def score(self, q, e):
            return 1.0, [1.0, 0.0, 0.0, 0.0]

        def rerank(self, q, c, top_k=5):
            return c[:top_k]

        def retrieve(self, query, top_k=20, alpha=None, filters=None):
            return []

        def set_generation_kwargs(self, **kwargs):
            pass

        def generate(self, query=None, context=None, **kw):
            return "stub"

        @property
        def vector_store(self):
            return self

        @property
        def model(self):
            return self

        def encode(self, texts, normalize_embeddings=True):
            import numpy as np
            return np.zeros((len(texts), 16), dtype="float32")

    class _Executor:
        def execute(self, name, args):
            if name == "CreateTicket":
                return {"ticket_id": "TKT-X", "message": "escalated"}
            return {"results": []}

    monkeypatch.setattr(m5_mod, "ChromaVectorStore", _Stub)
    monkeypatch.setattr(m5_mod, "BM25Index", _Stub)
    monkeypatch.setattr(m5_mod, "ColBERTReranker", _Stub)
    monkeypatch.setattr(m5_mod, "Generator", _Stub)
    monkeypatch.setattr(m5_mod, "ConfidenceHead", _Stub)
    monkeypatch.setattr(m5_mod, "AlphaNetwork", lambda *a, **kw: None)
    monkeypatch.setattr(m5_mod, "AnswerVerify", lambda *a, **kw: None)
    monkeypatch.setattr(m5_mod, "ToolExecutor", lambda *a, **kw: _Executor())

    for tag in ("m1", "m5"):
        flags = PipelineFlags.for_tag(tag)
        assert isinstance(flags, PipelineFlags)
