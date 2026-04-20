"""Unit tests for Generator prompt + citation parsing (no model load)."""

from __future__ import annotations

from src.data.schema import ChunkRecord, RetrievalResult


def _make_generator_no_load():
    """Instantiate Generator without running __init__ model loads."""
    from src.models.generator import Generator

    g = Generator.__new__(Generator)
    g._generation_kwargs = {"temperature": 0.0, "do_sample": False, "max_new_tokens": 256}
    g._model = None
    g._tokenizer = None
    g._llama = None
    g.backend = "hf"
    return g


def test_parse_citations_resolves_cited_text():
    g = _make_generator_no_load()
    chunk = ChunkRecord(
        chunk_id="c1",
        doc_id="abcdef1234567890",
        text="Alpha beta gamma delta epsilon.",
        source="src.md",
        span_start=100,
        span_end=130,
    )
    rr = RetrievalResult(chunk=chunk, score=0.9)
    answer = "Gamma is the third letter [abcdef1234567890:110-115]."
    cites = g.parse_citations(answer, [rr])
    assert len(cites) == 1
    c = cites[0]
    assert c.doc_id == "abcdef1234567890"
    assert c.span_start == 110
    assert c.span_end == 115
    assert "beta" in c.cited_text or "gamma" in c.cited_text or c.cited_text


def test_build_prompt_includes_system_and_context():
    g = _make_generator_no_load()
    chunk = ChunkRecord(
        chunk_id="c1",
        doc_id="d1",
        text="hello world",
        source="s.md",
        span_start=0,
        span_end=11,
    )
    rr = RetrievalResult(chunk=chunk, score=0.9)
    prompt = g._build_prompt(None, "What?", [rr])
    assert "AegisRAG" in prompt
    assert "hello world" in prompt
    assert "[d1:0-11]" in prompt


def test_parse_citations_dedupes():
    g = _make_generator_no_load()
    chunk = ChunkRecord(
        chunk_id="c1", doc_id="doc1111", text="abcdef", source="s",
        span_start=0, span_end=6,
    )
    rr = RetrievalResult(chunk=chunk, score=0.9)
    answer = "Foo [doc1111:0-3] bar [doc1111:0-3] baz."
    cites = g.parse_citations(answer, [rr])
    assert len(cites) == 1
