"""
AegisRAG - Confidence Soft-Label Generator

For each input query: retrieve the top-k evidence chunks, prompt the
generator to answer using only those chunks, then score that answer
against the gold answer via a fast NLI-based similarity score.

"""

from __future__ import annotations

import json
import logging
from multiprocessing import pool
import random
import torch
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.schema import ConfidenceLabel, QAPair
from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)

# FIX 2: Faster model; DeBERTa MNLI is ~3x faster than roberta-large BERTScore
_DEFAULT_FAST_MODEL = "cross-encoder/nli-deberta-v3-small"
# FIX 2: Batch size for scoring to avoid OOM
_SCORE_BATCH_SIZE = 32


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class ConfidenceLabelGenerator:
    """Produce soft-label confidence examples using fast NLI similarity.

    FIX 2: Uses a DeBERTa MNLI cross-encoder instead of BERTScore
    (roberta-large) for speed. Scoring is batched to avoid OOM.

    Parameters
    ----------
    retriever : object
        Must expose ``retrieve(query: str, top_k: int) -> list[(ChunkRecord, float)]``.
    generator : object
        Must expose ``generate(prompt: str) -> str``.
    bertscore_model : str
        Kept for API compatibility; the fast NLI model is used instead.
    top_k : int
        Number of evidence chunks per query (default 5).
    seed : int
        Deterministic seed (default 42).
    limit : int or None
        Target number of labels.
    score_batch_size : int
        Batch size for NLI scoring (FIX 2).
    """



    def __init__(
        self,
        retriever: Any,
        generator: Any | None = None,
        bertscore_model: str = "roberta-large",  # kept for API compat
        top_k: int = 5,
        seed: int = 42,
        limit: int | None = None,
        score_batch_size: int = _SCORE_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        set_seed(seed)
        self.seed = seed
        self.rng = random.Random(seed)
        self.device = device or get_best_device()

        cfg = get_config()
        self.cfg = cfg

        self.retriever = retriever
        self._generator = generator
        self.top_k = int(top_k)
        self.score_batch_size = int(score_batch_size)

        if limit is not None:
            self.target_count = int(limit)
        else:
            configured = int(getattr(cfg.synthetic_data, "confidence_labels", 3000))
            self.target_count = max(configured, 3000)

        # FIX 2: Use fast NLI model; lazy-loaded
        self._nli_model: Any = None

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

        queries: list[str] = []
        evidence_answers: list[str] = []
        gold_answers: list[str] = []
        top5_ids: list[list[str]] = []

        total = len(pool)

        for idx, qa in enumerate(pool, 1):
            logger.info(f"[{idx}/{total}] Processing query: {qa.query[:80]}...")
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
            logger.warning("No evidence answers produced; skipping scoring")
            return []

        # FIX 2: Use fast batched NLI scoring instead of BERTScore
        f1_scores = self._fast_similarity_scores(evidence_answers, gold_answers)

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
    # FIX 2: Fast batched NLI scoring
    # ------------------------------------------------------------------

    def _fast_similarity_scores(
        self, candidates: Sequence[str], references: Sequence[str]
    ) -> list[float]:
        """Compute soft similarity scores using a fast NLI cross-encoder.

        FIX 2: ~3x faster than BERTScore with roberta-large. Processes
        pairs in batches of ``self.score_batch_size`` to avoid OOM.

        Falls back to BERTScore if the NLI model fails to load.
        """
        try:
            nli = self._get_nli_model()
            pairs = list(zip(candidates, references))
            scores: list[float] = []

            logger.info(
                "Scoring %d pairs with NLI model (batch_size=%d)...",
                len(pairs),
                self.score_batch_size,
            )

            for start in range(0, len(pairs), self.score_batch_size):
                batch = pairs[start : start + self.score_batch_size]
                try:
                    # NLI returns [contradiction, entailment, neutral] or
                    # a single entailment score depending on model head.
                    raw = nli.predict(batch)
                    import numpy as np

                    for row in raw:
                        row_arr = np.asarray(row, dtype=float)
                        if row_arr.ndim == 0 or row_arr.size == 1:
                            scores.append(float(row_arr.flat[0]))
                        elif row_arr.size == 3:
                            # [contradiction, entailment, neutral] -> softmax -> entailment prob
                            ex = np.exp(row_arr - row_arr.max())
                            probs = ex / ex.sum()
                            scores.append(float(probs[1]))  # entailment index
                        else:
                            scores.append(float(row_arr.max()))
                except Exception as exc:
                    logger.warning("NLI batch scoring failed: %s", exc)
                    scores.extend([0.5] * len(batch))

            return scores

        except Exception as exc:
            logger.warning(
                "NLI model unavailable (%s); falling back to BERTScore", exc
            )
            return self._bertscore_f1(candidates, references)

    def _get_nli_model(self) -> Any:
        """Lazy-load the NLI cross-encoder (FIX 2)."""
        if self._nli_model is not None:
            return self._nli_model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc
        logger.info("Loading fast NLI model: %s", _DEFAULT_FAST_MODEL)
        self._nli_model = CrossEncoder(_DEFAULT_FAST_MODEL, max_length=512, device=self.device)
        logger.info(f"Initialized Generator on DEVICE: {self.device}")
        return self._nli_model

    # ------------------------------------------------------------------
    # BERTScore fallback (kept for compatibility)
    # ------------------------------------------------------------------

    def _bertscore_f1(
        self, candidates: Sequence[str], references: Sequence[str]
    ) -> list[float]:
        """Compute BERTScore F1 in [0, 1] — used as fallback only."""
        try:
            from bert_score import score as bertscore
        except ImportError as exc:
            raise ImportError(
                "ConfidenceLabelGenerator requires 'bert_score'. "
                "Install with: pip install bert-score"
            ) from exc

        logger.info(
            "Computing BERTScore F1 for %d pairs (fallback mode)", len(candidates)
        )
        # FIX 2: Batch BERTScore as well to avoid OOM
        all_f1: list[float] = []
        for start in range(0, len(candidates), self.score_batch_size):
            cands_batch = list(candidates[start : start + self.score_batch_size])
            refs_batch = list(references[start : start + self.score_batch_size])
            _, _, f1 = bertscore(
                cands=cands_batch,
                refs=refs_batch,
                lang="en",
                rescale_with_baseline=False,
                verbose=False,
            )
            all_f1.extend([float(x) for x in f1.tolist()])
        return all_f1

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
    # Generator resolution
    # ------------------------------------------------------------------

    def _get_generator(self) -> Any:
        if self._generator is not None:
            return self._generator
        try:
            from src.models.generator import Generator  # lazy import
        except ImportError as exc:
            raise ImportError(
                "ConfidenceLabelGenerator needs src.models.generator.Generator."
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