"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from src.data.schema import Citation
from src.evaluation.calibration import compute_ece
from src.evaluation.metrics import (
    citation_f1,
    escalation_f1,
    grounding_score,
)


def test_grounding_score_all_supported():
    citation = Citation(
        doc_id="d1", chunk_id="c1", span_start=0, span_end=30,
        cited_text="the quick brown fox jumps over",
    )
    score = grounding_score("quick brown fox jumps", [citation])
    assert score >= 0.5


def test_grounding_score_unsupported():
    citation = Citation(
        doc_id="d1", chunk_id="c1", span_start=0, span_end=5,
        cited_text="alpha",
    )
    score = grounding_score("completely different sentence", [citation])
    assert score <= 0.3


def test_citation_f1_exact_match():
    pred = [Citation(doc_id="d1", chunk_id="c1", span_start=0, span_end=10,
                     cited_text="")]
    gold = [{"doc_id": "d1", "span_start": 0, "span_end": 10}]
    res = citation_f1(pred, gold)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0


def test_escalation_f1_perfect():
    res = escalation_f1([True, False, True], [True, False, True])
    assert res["f1"] == 1.0


def test_compute_ece_range():
    conf = np.linspace(0.05, 0.95, 50)
    acc = (conf > 0.5).astype(float)
    ece = compute_ece(conf, acc, n_bins=10)
    assert 0.0 <= ece <= 1.0
