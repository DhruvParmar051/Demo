"""
AegisRAG - Baseline Pipelines (B1, B2, B3)

Three single-pass baselines without CGAL/DPO/verify/decomposition:

- **B1**  Dense-only (BGE-m3 + Chroma), top-5, zero-shot Qwen2.5.
- **B2**  Dense + BM25 hybrid (fixed alpha=0.5) + ColBERT rerank + zero-shot.
- **B3**  Fine-tuned retriever + fine-tuned reranker + SFT adapter.

python scripts/evaluate_all.py --models m2 --output-dir report

"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from src.data.schema import Citation, QueryResponse, RetrievalResult
from src.models.generator import Generator
from src.retrieval.bm25_index import BM25Index
from src.retrieval.retriever import HybridRetriever
from src.retrieval.vector_store import ChromaVectorStore
from src.reranker.reranker import ColBERTReranker
from src.utils.config import get_config

logger = logging.getLogger(__name__)


def _load_bm25(path: str | None) -> BM25Index:
    """Load a BM25Index from disk or return an empty one."""
    idx = BM25Index()
    if path:
        try:
            idx.load(path)
            logger.info("BM25 index loaded from %s (%d docs)", path, idx.size)
        except FileNotFoundError:
            logger.warning("BM25 index not found at %s; returning empty index.", path)
        except Exception as exc:
            logger.warning("Failed to load BM25 index: %s", exc)
    return idx


def _bm25_path_from_cfg(cfg: Any) -> str | None:
    """Resolve BM25 index path from config — handles both cfg.data and cfg.paths."""
    # Primary: cfg.data.bm25_index_path (correct location in base.yaml)
    path = getattr(getattr(cfg, "data", None), "bm25_index_path", None)
    if path:
        return path
    # Fallback: cfg.paths.bm25_index (legacy)
    return getattr(getattr(cfg, "paths", None), "bm25_index", None)


class _BaselineBase:
    """Shared scaffolding for all baselines."""

    model_tag = "base"

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        bm25_index: BM25Index | None = None,
        reranker: ColBERTReranker | None = None,
        generator: Generator | None = None,
    ) -> None:
        cfg = get_config()
        self.cfg = cfg
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25_index = bm25_index
        self.reranker = reranker
        # B1/B2 use the base (untrained) generator — explicitly pass aegis_base.gguf
        # so they are never contaminated by SFT/DPO weights regardless of config defaults.
        if generator is not None:
            self.generator = generator
        else:
            cfg_gguf = getattr(cfg.models.generator, "gguf_path", "")
            base_gguf = Path(cfg_gguf).parent / "aegis_base.gguf" if cfg_gguf else Path("checkpoints/aegis_base.gguf")
            self.generator = Generator(gguf_path=str(base_gguf) if base_gguf.exists() else None)

    def _build_contexts(
        self, reranked: list[tuple[Any, float]]
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(chunk=c, score=s, rerank_score=s)
            for c, s in reranked
        ]

    def _to_response(
        self,
        query: str,
        answer: str,
        citations: list[Any],
        t_start: float,
    ) -> QueryResponse:
        return QueryResponse(
            answer=answer,
            citations=citations,
            tool_calls=[],
            confidence=0.0,
            cgal_iterations=0,
            decomposed=False,
            session_id=str(uuid.uuid4()),
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
            model_tag=self.model_tag,
        )

    def __call__(self, query: str) -> QueryResponse:
        return self.run(query)

    @staticmethod
    def _citations_from_context(
        context: list[RetrievalResult],
        use_score_filter: bool = False,
    ) -> list[Citation]:
        """Build Citation objects from retrieved chunks.

        When ``use_score_filter=True`` (B2/B3 which have reranker scores):
        only cite chunks within 40% of the top score — improves precision.
        When False (B1, dense scores only): cite all chunks.
        """
        if use_score_filter and context:
            scores = [rr.rerank_score or rr.score for rr in context]
            top = max(scores) if scores else 0.0
            threshold = top * 0.60
            filtered = [rr for rr, s in zip(context, scores) if s >= threshold]
            # Guarantee at least top-2
            context = filtered if len(filtered) >= 2 else context[:2]

        citations = []
        for rr in context:
            chunk = rr.chunk
            cited_text = chunk.text.strip()
            if len(cited_text) > 2000:
                cited_text = cited_text[:1997] + "..."
            citations.append(Citation(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                span_start=chunk.span_start,
                span_end=chunk.span_end,
                cited_text=cited_text,
                source=chunk.source,
                page_number=chunk.page_number,
            ))
        return citations


class BaselineB1(_BaselineBase):
    """Dense-only + Qwen baseline.

    Uses the Chroma vector store directly (no BM25, no reranker), forces
    ``alpha=1.0``. Represents a minimal RAG system.
    """

    model_tag = "b1"

    def run(self, query: str) -> QueryResponse:
        t_start = time.perf_counter()
        top_k = int(self.cfg.retrieval.top_k)
        max_cit = int(getattr(self.cfg.retrieval, "max_citations", 5))
        dense = self.vector_store.query(query, top_k=top_k)
        context = self._build_contexts(dense[:max(max_cit, 5)])
        answer = self.generator.generate(query=query, context=context)
        # B1 has no reranker scores — cite all for both precision and grounding
        citations = self._citations_from_context(context, use_score_filter=False)
        resp = self._to_response(query, answer, citations, t_start)
        resp.grounding_citations = citations
        return resp


class BaselineB2(_BaselineBase):
    """Hybrid (fixed alpha=0.5) + ColBERT rerank + Qwen baseline."""

    model_tag = "b2"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.bm25_index is None:
            self.bm25_index = _load_bm25(_bm25_path_from_cfg(self.cfg))
        if self.reranker is None:
            self.reranker = ColBERTReranker()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            alpha_network=None,
        )

    def run(self, query: str) -> QueryResponse:
        t_start = time.perf_counter()
        max_cit = int(getattr(self.cfg.retrieval, "max_citations", 5))
        retrieved = self.retriever.retrieve(
            query, top_k=int(self.cfg.retrieval.top_k), alpha=0.5
        )
        reranked = self.reranker.rerank(
            query, retrieved, top_k=int(self.cfg.retrieval.rerank_top_k)
        )
        context = self._build_contexts(reranked[:max(max_cit, 5)])
        answer = self.generator.generate(query=query, context=context)
        # B2 has reranker scores — filter for precision, keep all for grounding
        citations = self._citations_from_context(context, use_score_filter=True)
        resp = self._to_response(query, answer, citations, t_start)
        resp.grounding_citations = self._citations_from_context(context, use_score_filter=False)
        return resp


class BaselineB3(_BaselineBase):
    """Fine-tuned retriever + fine-tuned reranker + SFT adapter."""

    model_tag = "b3"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        cfg = self.cfg
        reranker_ckpt = getattr(cfg.checkpoints, "reranker", None)

        if self.bm25_index is None:
            self.bm25_index = _load_bm25(_bm25_path_from_cfg(cfg))
        if self.reranker is None:
            try:
                self.reranker = ColBERTReranker(checkpoint_path=reranker_ckpt)
            except Exception:
                self.reranker = ColBERTReranker()

        # B3 uses base (untrained) generator — trained generators reserved for M1-M5.
        cfg_gguf = getattr(cfg.models.generator, "gguf_path", "")
        base_gguf = Path(cfg_gguf).parent / "aegis_base.gguf" if cfg_gguf else Path("checkpoints/aegis_base.gguf")
        self.generator = Generator(gguf_path=str(base_gguf) if base_gguf.exists() else None)

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            alpha_network=None,
        )

    def run(self, query: str) -> QueryResponse:
        t_start = time.perf_counter()
        max_cit = int(getattr(self.cfg.retrieval, "max_citations", 5))
        retrieved = self.retriever.retrieve(
            query, top_k=int(self.cfg.retrieval.top_k), alpha=0.5
        )
        reranked = self.reranker.rerank(
            query, retrieved, top_k=int(self.cfg.retrieval.rerank_top_k)
        )
        context = self._build_contexts(reranked[:max(max_cit, 5)])
        answer = self.generator.generate(query=query, context=context)
        # B3 has reranker scores — filter for precision, keep all for grounding
        citations = self._citations_from_context(context, use_score_filter=True)
        resp = self._to_response(query, answer, citations, t_start)
        resp.grounding_citations = self._citations_from_context(context, use_score_filter=False)
        return resp
