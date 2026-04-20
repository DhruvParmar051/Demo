"""Unit tests for BM25Index (pure-python, no heavy deps)."""

from __future__ import annotations

import pytest

from src.retrieval.bm25_index import BM25Index


def test_bm25_build_and_query(mini_corpus):
    idx = BM25Index()
    idx.build_index(mini_corpus)
    results = idx.query("reset password", top_k=3)
    assert results, "BM25 returned no results"
    top_chunk, top_score = results[0]
    assert "password" in top_chunk.text.lower()
    assert top_score > 0


def test_bm25_empty_query(mini_corpus):
    idx = BM25Index()
    idx.build_index(mini_corpus)
    # Empty query should still return something (or empty), not raise.
    results = idx.query("", top_k=3)
    assert isinstance(results, list)


def test_bm25_save_and_load(tmp_path, mini_corpus):
    idx = BM25Index()
    idx.build_index(mini_corpus)
    save_path = tmp_path / "bm25.pkl"
    idx.save(save_path)

    idx2 = BM25Index()
    idx2.load(save_path)
    r1 = idx.query("refund", top_k=2)
    r2 = idx2.query("refund", top_k=2)
    assert [c.chunk_id for c, _ in r1] == [c.chunk_id for c, _ in r2]


def test_bm25_topk_respected(mini_corpus):
    idx = BM25Index()
    idx.build_index(mini_corpus)
    results = idx.query("refund", top_k=2)
    assert len(results) <= 2
