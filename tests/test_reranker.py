"""Lightweight tests for the reranker interface (model load skipped)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_reranker_rerank_preserves_length_and_sorts(monkeypatch, mini_corpus):
    """Feed a fake cross-encoder to ColBERTReranker and check sort/length."""
    import numpy as np

    from src.reranker.reranker import ColBERTReranker

    # Avoid loading the real model; build the reranker but override _score_pairs.
    rr = ColBERTReranker.__new__(ColBERTReranker)
    rr.model_name = "mock"
    rr.batch_size = 4
    rr.max_length = 128
    rr.device = "cpu"
    rr.tokenizer = None
    rr.model = None

    def fake_score(pairs):
        return np.array([float(len(p[1])) for p in pairs])

    rr._score_pairs = fake_score  # type: ignore[attr-defined]

    chunks = [(c, 0.0) for c in mini_corpus]
    out = rr.rerank("anything", chunks, top_k=5)
    assert len(out) == 5
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)
