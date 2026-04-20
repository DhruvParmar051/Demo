"""Tests for decomposer classifier / splitter / merger."""

from __future__ import annotations

import pytest

from src.data.schema import Citation, QueryResponse


class _StubGenerator:
    """Returns a hard-coded JSON array from its generate method."""

    def __init__(self, response: str):
        self.response = response

    def generate(self, query=None, context=None, prompt=None, **kw):
        return self.response


def test_splitter_heuristic_fallback():
    from src.decomposer.splitter import QuerySplitter

    gen = _StubGenerator("not-json")
    sp = QuerySplitter(generator=gen)
    subs = sp.split("Do X and also do Y?")
    # Falls back to heuristic and returns a list with 2 items.
    assert len(subs) >= 1


def test_splitter_parses_json_response():
    from src.decomposer.splitter import QuerySplitter

    gen = _StubGenerator('["Reset password?", "Update billing address?"]')
    sp = QuerySplitter(generator=gen)
    subs = sp.split("Reset password and update billing address?")
    assert subs == ["Reset password?", "Update billing address?"]


def test_heuristic_detects_multipart():
    from src.decomposer.splitter import QuerySplitter

    sp = QuerySplitter(generator=_StubGenerator(""))
    assert sp.heuristic_is_multi_part("Do X and do Y")
    assert not sp.heuristic_is_multi_part("Simple question?")


def test_merger_combines_sub_responses():
    from src.decomposer.merger import ResultMerger

    r1 = QueryResponse(
        answer="A1", citations=[Citation(doc_id="d1", chunk_id="c1",
                                          span_start=0, span_end=1,
                                          cited_text="x")],
        confidence=0.8, cgal_iterations=1, alpha=0.5,
    )
    r2 = QueryResponse(
        answer="A2", confidence=0.6, cgal_iterations=2, alpha=0.7,
    )
    merged = ResultMerger().merge([r1, r2], "original")
    assert "A1" in merged.answer
    assert "A2" in merged.answer
    assert merged.confidence == pytest.approx(0.6)  # min
    assert merged.cgal_iterations == 2  # max
    assert len(merged.citations) == 1
