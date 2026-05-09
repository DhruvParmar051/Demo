"""
AegisRAG - Baseline Pipelines (B1, B2, B3)

Three single-pass baselines without CGAL/DPO/verify/decomposition:

- **B1**  Dense-only (BGE-m3 + Chroma), top-5, zero-shot Qwen2.5.
- **B2**  Dense + BM25 hybrid (fixed alpha=0.5) + ColBERT rerank + zero-shot.
- **B3**  Fine-tuned retriever + fine-tuned reranker + SFT adapter.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from src.data.schema import QueryResponse, RetrievalResult
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
        except FileNotFoundError:
            logger.warning("BM25 index not found at %s; returning empty index.", path)
        except Exception as exc:
            logger.warning("Failed to load BM25 index: %s", exc)
    return idx


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
        self.generator = generator or Generator()

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


class BaselineB1(_BaselineBase):
    """Dense-only + Qwen baseline.

    Uses the Chroma vector store directly (no BM25, no reranker), forces
    ``alpha=1.0``. Represents a minimal RAG system.
    """

    model_tag = "b1"

    def run(self, query: str) -> QueryResponse:
        t_start = time.perf_counter()
        top_k = int(self.cfg.retrieval.rerank_top_k)
        dense = self.vector_store.query(query, top_k=top_k)
        reranked = dense  # no rerank
        context = self._build_contexts(reranked)
        answer, citations = self.generator.generate_with_citations(query, context)
        return self._to_response(query, answer, citations, t_start)


class BaselineB2(_BaselineBase):
    """Hybrid (fixed alpha=0.5) + ColBERT rerank + Qwen baseline."""

    model_tag = "b2"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.bm25_index is None:
            self.bm25_index = _load_bm25(getattr(self.cfg.paths, "bm25_index", None))
        if self.reranker is None:
            self.reranker = ColBERTReranker()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            alpha_network=None,
        )

    def run(self, query: str) -> QueryResponse:
        t_start = time.perf_counter()
        retrieved = self.retriever.retrieve(
            query, top_k=int(self.cfg.retrieval.top_k), alpha=0.5
        )
        reranked = self.reranker.rerank(
            query, retrieved, top_k=int(self.cfg.retrieval.rerank_top_k)
        )
        context = self._build_contexts(reranked)
        answer, citations = self.generator.generate_with_citations(query, context)
        return self._to_response(query, answer, citations, t_start)


class BaselineB3(_BaselineBase):
    """Fine-tuned retriever + fine-tuned reranker + SFT adapter."""

    model_tag = "b3"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        cfg = self.cfg
        # Try loading fine-tuned components from disk.
        retriever_ckpt = getattr(cfg.checkpoints, "retriever", None)
        reranker_ckpt = getattr(cfg.checkpoints, "reranker", None)
        sft_adapter = getattr(cfg.checkpoints, "generator_sft", None)

        # ChromaDB was indexed with the base embedding model; querying with the
        # fine-tuned retriever checkpoint shifts the embedding space and causes
        # ~50% of queries to return zero candidates.  Use base model for querying.
        if self.bm25_index is None:
            self.bm25_index = _load_bm25(getattr(cfg.paths, "bm25_index", None))
        if self.reranker is None:
            try:
                self.reranker = ColBERTReranker(checkpoint_path=reranker_ckpt)
            except Exception:
                self.reranker = ColBERTReranker()

        # Use the pre-merged SFT GGUF so adapter weights are actually applied.
        # GGUF (llama-cpp) cannot load HF LoRA adapters at runtime.
        legacy_gguf = Path(getattr(cfg.models.generator, "gguf_path", "checkpoints/aegis_final.gguf"))
        sft_gguf = legacy_gguf.parent / "aegis_sft.gguf"
        chosen_gguf = sft_gguf if sft_gguf.exists() else legacy_gguf
        if not sft_gguf.exists():
            logger.warning(
                "aegis_sft.gguf not found; falling back to %s. "
                "Run: python scripts/convert_to_gguf.py --variant sft",
                chosen_gguf,
            )
        self.generator = Generator(gguf_path=str(chosen_gguf))

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            alpha_network=None,
        )

    def run(self, query: str) -> QueryResponse:
        t_start = time.perf_counter()
        retrieved = self.retriever.retrieve(
            query, top_k=int(self.cfg.retrieval.top_k), alpha=0.5
        )
        reranked = self.reranker.rerank(
            query, retrieved, top_k=int(self.cfg.retrieval.rerank_top_k)
        )
        context = self._build_contexts(reranked)
        answer, citations = self.generator.generate_with_citations(query, context)
        return self._to_response(query, answer, citations, t_start)
