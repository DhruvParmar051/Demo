"""
AegisRAG - M5 Pipeline

Configurable wrapper around :class:`CGALLoopEngine` that supports feature
toggles so the same class can emulate M1, M2, M3, M4, or the full M5
system described in the plan.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.cgal.alpha_network import AlphaNetwork
from src.cgal.confidence_head import ConfidenceHead
from src.cgal.loop_engine import CGALLoopEngine
from src.data.schema import QueryResponse
from src.decomposer.classifier import DecompositionClassifier
from src.decomposer.merger import ResultMerger
from src.decomposer.splitter import QuerySplitter
from src.models.generator import Generator
from src.retrieval.bm25_index import BM25Index
from src.retrieval.retriever import HybridRetriever
from src.retrieval.vector_store import ChromaVectorStore
from src.reranker.reranker import ColBERTReranker
from src.tools.answer_verify import AnswerVerify
from src.tools.executor import ToolExecutor
from src.utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class PipelineFlags:
    """Feature toggles that select a model variant."""

    cgal: bool = True
    dpo: bool = True
    verify: bool = True
    adaptive_alpha: bool = True
    decomposition: bool = True
    rule_based_tools: bool = False

    @classmethod
    def for_tag(cls, tag: str) -> "PipelineFlags":
        tag = tag.lower().strip()
        mapping = {
            "m1": cls(cgal=False, dpo=False, verify=False,
                      adaptive_alpha=False, decomposition=False,
                      rule_based_tools=True),
            "m2": cls(cgal=True, dpo=False, verify=False,
                      adaptive_alpha=False, decomposition=False),
            "m3": cls(cgal=True, dpo=True, verify=False,
                      adaptive_alpha=False, decomposition=False),
            "m4": cls(cgal=True, dpo=True, verify=True,
                      adaptive_alpha=False, decomposition=False),
            "m5": cls(cgal=True, dpo=True, verify=True,
                      adaptive_alpha=True, decomposition=True),
        }
        if tag not in mapping:
            raise ValueError(f"Unknown model tag: {tag}")
        return mapping[tag]


class _Decomposer:
    """Minimal glue exposing ``is_multi_part``, ``split``, ``merger``."""

    def __init__(
        self,
        classifier: DecompositionClassifier | None,
        splitter: QuerySplitter,
        merger: ResultMerger,
        encoder: Any,
    ) -> None:
        self.classifier = classifier
        self.splitter = splitter
        self.merger = merger
        self.encoder = encoder

    def is_multi_part(self, query: str) -> bool:
        if self.classifier is None:
            return self.splitter.heuristic_is_multi_part(query)
        try:
            emb = self.encoder.encode([query], normalize_embeddings=True)[0]
            flag, _ = self.classifier.is_multi_part(emb)
            return bool(flag)
        except Exception as exc:
            logger.warning("DecompositionClassifier failed: %s", exc)
            return self.splitter.heuristic_is_multi_part(query)

    def split(self, query: str) -> list[str]:
        return self.splitter.split(query)


class M5Pipeline:
    """Configurable full-system pipeline."""

    def __init__(
        self,
        flags: PipelineFlags,
        vector_store: ChromaVectorStore | None = None,
        bm25_index: BM25Index | None = None,
        reranker: ColBERTReranker | None = None,
        generator: Generator | None = None,
        confidence_head: ConfidenceHead | None = None,
        alpha_network: AlphaNetwork | None = None,
        answer_verify: AnswerVerify | None = None,
        tool_executor: ToolExecutor | None = None,
        decomposer: _Decomposer | None = None,
        config: Any = None,
        model_tag: str = "m5",
    ) -> None:
        cfg = config if config is not None else get_config()
        self.cfg = cfg
        self.flags = flags
        self.model_tag = model_tag

        self.vector_store = vector_store or ChromaVectorStore()
        if bm25_index is None:
            bm25_index = BM25Index()
            bm25_path = getattr(cfg.paths, "bm25_index", None)
            if bm25_path:
                try:
                    bm25_index.load(bm25_path)
                except Exception as exc:
                    logger.warning("BM25 load failed (%s); using empty index.", exc)
        self.bm25_index = bm25_index
        self.reranker = reranker or ColBERTReranker()
        self.generator = generator or Generator()

        # Confidence head -- required for CGAL; synthesize no-op if disabled.
        if flags.cgal:
            self.confidence_head = confidence_head or ConfidenceHead()
            self.confidence_head.eval()
        else:
            self.confidence_head = _StubConfidenceHead(high_conf=1.0)

        # Alpha network only in M5.
        self.alpha_network = alpha_network if flags.adaptive_alpha else None

        # Verifier only enabled when flag is set.
        self.answer_verify = answer_verify if flags.verify else None
        if self.answer_verify is not None and hasattr(self.answer_verify, "warmup"):
            logger.info("Starting AnswerVerify warmup in background thread")
            threading.Thread(
                target=self.answer_verify.warmup,
                daemon=True,
                name="answer_verify_warmup",
            ).start()

        # Retriever wraps alpha network (or fixed alpha).
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            alpha_network=self.alpha_network,
        )

        # Tool executor.
        ticket_db = Path(cfg.paths.audit_db) if hasattr(cfg, "paths") else None
        self.tool_executor = tool_executor or ToolExecutor(
            retriever=self.retriever,
            reranker=self.reranker,
            policy_index=None,
            ticket_store_path=ticket_db,
        )

        # Decomposer only in M5.
        if flags.decomposition:
            splitter = QuerySplitter(generator=self.generator)
            merger = ResultMerger()
            try:
                classifier = DecompositionClassifier()
            except Exception:
                classifier = None
            self.decomposer: _Decomposer | None = _Decomposer(
                classifier=classifier,
                splitter=splitter,
                merger=merger,
                encoder=self.vector_store.model,
            )
        else:
            self.decomposer = None

        self.engine = CGALLoopEngine(
            retriever=self.retriever,
            reranker=self.reranker,
            confidence_head=self.confidence_head,
            generator=self.generator,
            tool_executor=self.tool_executor,
            answer_verify=self.answer_verify,
            decomposer=self.decomposer,
            alpha_network=self.alpha_network,
            config=cfg,
        )

    # ------------------------------------------------------------------

    @classmethod
    def from_tag(cls, tag: str, config: Any = None) -> "M5Pipeline":
        """Factory that builds a pipeline with the flags implied by *tag*."""
        flags = PipelineFlags.for_tag(tag)
        return cls(flags=flags, config=config, model_tag=tag)

    def run(self, query: str, stream: bool = False) -> Any:
        """Run the pipeline. Returns QueryResponse or an async iterator."""
        response = self.engine.run(query, stream=stream)
        if isinstance(response, QueryResponse):
            response.model_tag = self.model_tag
        return response


class _StubConfidenceHead:
    """No-op confidence head returning a constant high score.

    Used when ``cgal`` flag is False (baseline M1). Forces the loop engine
    to take the direct-answer path on iteration 0.
    """

    def __init__(self, high_conf: float = 1.0) -> None:
        self.high_conf = high_conf

    def eval(self) -> None:
        return

    def score(self, query_emb: Any, evidence_embs: Any) -> tuple[float, list[float]]:
        return self.high_conf, [1.0, 0.0, 0.0, 0.0]
