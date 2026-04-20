"""
AegisRAG - Confidence Soft-Label Generator

For each input query: retrieve the top-k evidence chunks, prompt the
generator to answer using only those chunks, then score that answer
against the gold answer via BERTScore F1. The F1 value (in [0, 1])
serves as the soft label for the confidence head.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.schema import ConfidenceLabel, QAPair
from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


class ConfidenceLabelGenerator:
    """Produce soft-label confidence examples using BERTScore F1.

    Parameters
    ----------
    retriever : object
        Must expose ``retrieve(query: str, top_k: int) -> list[(ChunkRecord, float)]``.
    generator : object
        Must expose ``generate(prompt: str) -> str``.
    bertscore_model : str
        HuggingFace model id for BERTScore (default: roberta-large).
    top_k : int
        Number of evidence chunks per query (default 5).
    seed : int
        Deterministic seed (default 42).
    limit : int or None
        Target number of labels (default: 3000 or
        ``cfg.synthetic_data.confidence_labels``).
    """

    def __init__(
        self,
        retriever: Any,
        generator: Any | None = None,
        bertscore_model: str = "roberta-large",
        top_k: int = 5,
        seed: int = 42,
        limit: int | None = None,
    ) -> None:
        set_seed(seed)
        self.seed = seed
        self.rng = random.Random(seed)

        cfg = get_config()
        self.cfg = cfg

        self.retriever = retriever
        self._generator = generator
        self.bertscore_model = bertscore_model
        self.top_k = int(top_k)

        if limit is not None:
            self.target_count = int(limit)
        else:
            # Config key may be absent; fall back to 3000.
            self.target_count = int(getattr(cfg.synthetic_data, "confidence_labels", 3000))
            if self.target_count < 3000:
                self.target_count = 3000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        qa_pairs: Sequence[QAPair],
        output_path: Path | str | None = None,
    ) -> list[ConfidenceLabel]:
        """Score a batch of QA pairs and emit :class:`ConfidenceLabel` records."""
        if not qa_pairs:
            logger.warning("ConfidenceLabelGenerator.generate called with no QA pairs")
            return []

        output_path = self._resolve_output(output_path, "confidence_labels.jsonl")

        gen = self._get_generator()

        pool = list(qa_pairs)
        self.rng.shuffle(pool)
        pool = pool[: self.target_count]

        # Pass 1: produce (query, evidence_answer, gold_answer) rows.
        queries: list[str] = []
        evidence_answers: list[str] = []
        gold_answers: list[str] = []
        top5_ids: list[list[str]] = []

        for qa in pool:
            try:
                retrieved = self.retriever.retrieve(qa.query, top_k=self.top_k)
            except Exception as exc:
                logger.warning("Retriever failed on query '%s': %s", qa.query[:60], exc)
                continue

            chunks = [c for c, _ in retrieved]
            if not chunks:
                continue
            ids = [c.chunk_id for c in chunks]

            prompt = self._build_prompt(qa.query, chunks)
            try:
                answer = gen.generate(prompt) or ""
            except Exception as exc:
                logger.warning("Generator failed: %s", exc)
                continue
            answer = answer.strip()
            if not answer:
                continue

            queries.append(qa.query)
            evidence_answers.append(answer)
            gold_answers.append(qa.answer_with_citations)
            top5_ids.append(ids)

        if not evidence_answers:
            logger.warning("No evidence answers produced; skipping BERTScore")
            return []

        f1_scores = self._bertscore_f1(evidence_answers, gold_answers)

        labels: list[ConfidenceLabel] = []
        for q, ev, gold, ids, f1 in zip(
            queries, evidence_answers, gold_answers, top5_ids, f1_scores
        ):
            soft = float(max(0.0, min(1.0, f1)))
            labels.append(
                ConfidenceLabel(
                    query=q,
                    top5_chunk_ids=ids,
                    soft_label=soft,
                    gold_answer=gold,
                    evidence_answer=ev,
                )
            )

        self._write_jsonl(labels, output_path)
        logger.info(
            "ConfidenceLabelGenerator wrote %d labels to %s",
            len(labels),
            output_path,
        )
        return labels

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(query: str, chunks: Sequence[Any]) -> str:
        ctx_lines = []
        for c in chunks:
            marker = f"[{c.chunk_id}:{c.span_start}-{c.span_end}]"
            ctx_lines.append(f"{marker}\n{c.text}")
        context = "\n\n".join(ctx_lines)
        return (
            "Answer the user's question using ONLY the provided context. "
            "Cite every factual claim using the [chunk_id:start-end] markers "
            "shown above each passage. If the context does not answer the "
            "question, say so briefly.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
        )

    # ------------------------------------------------------------------
    # BERTScore
    # ------------------------------------------------------------------

    def _bertscore_f1(
        self, candidates: Sequence[str], references: Sequence[str]
    ) -> list[float]:
        """Compute BERTScore F1 in [0, 1]."""
        try:
            from bert_score import score as bertscore
        except ImportError as exc:
            raise ImportError(
                "ConfidenceLabelGenerator requires 'bert_score'. "
                "Install with: pip install bert-score"
            ) from exc

        logger.info(
            "Computing BERTScore F1 for %d pairs using %s",
            len(candidates),
            self.bertscore_model,
        )
        _, _, f1 = bertscore(
            cands=list(candidates),
            refs=list(references),
            model_type=self.bertscore_model,
            lang="en",
            rescale_with_baseline=False,
            verbose=False,
        )
        return [float(x) for x in f1.tolist()]

    # ------------------------------------------------------------------
    # Generator resolution
    # ------------------------------------------------------------------

    def _get_generator(self) -> Any:
        if self._generator is not None:
            return self._generator
        try:
            from src.models.generator import Generator  # lazy import
        except ImportError as exc:
            raise ImportError(
                "ConfidenceLabelGenerator needs src.models.generator.Generator; "
                "inject a generator instance via the constructor or install "
                "the generator module."
            ) from exc
        self._generator = Generator()
        return self._generator

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _resolve_output(
        self, output_path: Path | str | None, default_name: str
    ) -> Path:
        if output_path is None:
            base = self.cfg.resolve_path(self.cfg.data.synthetic_dir)
            output_path = base / default_name
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    @staticmethod
    def _write_jsonl(items: Iterable[Any], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for item in items:
                payload = item.to_dict() if hasattr(item, "to_dict") else item
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
