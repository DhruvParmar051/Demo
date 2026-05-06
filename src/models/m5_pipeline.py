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

        # Load fine-tuned retriever when available (mirrors BaselineB3 pattern).
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            retriever_ckpt = getattr(cfg.checkpoints, "retriever", None)
            if retriever_ckpt and Path(retriever_ckpt).exists():
                try:
                    self.vector_store = ChromaVectorStore(
                        embedding_model_name=retriever_ckpt
                    )
                    logger.info("Loaded fine-tuned retriever from %s", retriever_ckpt)
                except Exception as exc:
                    logger.warning(
                        "Fine-tuned retriever load failed (%s); using base model.", exc
                    )
                    self.vector_store = ChromaVectorStore()
            else:
                self.vector_store = ChromaVectorStore()

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

        # Load SFT or DPO adapter for the generator.
        # M3/M4/M5 have flags.dpo=True → prefer DPO adapter; M1/M2 get SFT only.
        if generator is not None:
            self.generator = generator
        else:
            sft_path = getattr(cfg.checkpoints, "generator_sft", None)
            dpo_path = getattr(cfg.checkpoints, "generator_dpo", None)
            adapter: str | None = None
            if flags.dpo and dpo_path and Path(dpo_path).exists():
                adapter = dpo_path
                logger.info("Generator will load DPO adapter from %s", adapter)
            elif sft_path and Path(sft_path).exists():
                adapter = sft_path
                logger.info("Generator will load SFT adapter from %s", adapter)
            self.generator = Generator(adapter_path=adapter)

        # Confidence head -- required for CGAL; synthesize no-op if disabled.
        if flags.cgal:
            if confidence_head is not None:
                self.confidence_head = confidence_head
            else:
                head_ckpt_dir = getattr(cfg.checkpoints, "confidence_head", None)
                head_ckpt = (
                    Path(head_ckpt_dir) / "confidence_head.pt"
                    if head_ckpt_dir
                    else None
                )
                if head_ckpt and head_ckpt.exists():
                    try:
                        self.confidence_head = ConfidenceHead.load(head_ckpt)
                        logger.info("Loaded ConfidenceHead from %s", head_ckpt)
                    except Exception as exc:
                        logger.warning(
                            "ConfidenceHead load failed (%s); using untrained head.", exc
                        )
                        self.confidence_head = ConfidenceHead()
                else:
                    self.confidence_head = ConfidenceHead()
            self.confidence_head.eval()
        else:
            self.confidence_head = _StubConfidenceHead(high_conf=1.0)

        # Alpha network only in M5 -- load from checkpoint when available.
        if not flags.adaptive_alpha:
            self.alpha_network = None
        elif alpha_network is not None:
            self.alpha_network = alpha_network
        else:
            alpha_ckpt_dir = getattr(cfg.checkpoints, "alpha_network", None)
            alpha_ckpt = (
                Path(alpha_ckpt_dir) / "model.pt" if alpha_ckpt_dir else None
            )
            if alpha_ckpt and alpha_ckpt.exists():
                try:
                    self.alpha_network = AlphaNetwork.load(alpha_ckpt)
                    logger.info("Loaded AlphaNetwork from %s", alpha_ckpt)
                except Exception as exc:
                    logger.warning(
                        "AlphaNetwork load failed (%s); using fixed alpha.", exc
                    )
                    self.alpha_network = None
            else:
                self.alpha_network = None

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

    def run(
        self,
        query: str,
        stream: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> Any:
        """Run the pipeline. Returns QueryResponse or an async iterator."""
        response = self.engine.run(query, stream=stream, history=history)
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
