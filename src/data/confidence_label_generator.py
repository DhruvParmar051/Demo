"""
AegisRAG - Confidence Soft-Label Generator

For each input query: retrieve the top-k evidence chunks, prompt the
generator to answer using only those chunks, then score that answer
against the gold answer via a fast NLI-based similarity score.

Supports resumable generation via checkpoint files. If the process is
interrupted, re-running with the same output_path will skip
already-scored queries and append only new labels.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import torch
import numpy as np
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.schema import ConfidenceLabel, QAPair
from src.utils.config import get_config
from src.utils.determinism import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# FIX 2: Faster model; DeBERTa MNLI is ~3x faster than roberta-large BERTScore
_DEFAULT_FAST_MODEL = "cross-encoder/nli-deberta-v3-small"
# FIX 2: Batch size for scoring to avoid OOM
_SCORE_BATCH_SIZE = 32
# How often (in queries) to flush labels + checkpoint to disk
_DEFAULT_CHECKPOINT_INTERVAL = 20


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def _query_hash(query: str) -> str:
    """Deterministic short hash of a query string for checkpoint tracking."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


class ConfidenceLabelGenerator:
    """Produce soft-label confidence examples using fast NLI similarity.

    FIX 2: Uses a DeBERTa MNLI cross-encoder instead of BERTScore
    (roberta-large) for speed. Scoring is batched to avoid OOM.

    Supports resumable generation: intermediate results are flushed to
    disk every ``checkpoint_interval`` queries, and a sidecar
    ``.checkpoint.json`` file tracks which queries have been scored. If
    the process is interrupted, re-running with the same ``output_path``
    will skip already-completed queries automatically.

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
    checkpoint_interval : int
        Flush labels and checkpoint to disk every N queries (default 50).
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
        checkpoint_interval: int = _DEFAULT_CHECKPOINT_INTERVAL,
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
        self.checkpoint_interval = int(checkpoint_interval)

        if limit is not None:
            self.target_count = int(limit)
        else:
            configured = int(getattr(cfg.synthetic_data, "confidence_labels", 3000))
            self.target_count = max(configured, 3000)

        # FIX 2: Use fast NLI model; lazy-loaded
        self._nli_model: Any = None

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _checkpoint_path(output_path: Path) -> Path:
        """Sidecar checkpoint file lives next to the output JSONL.

        Example: confidence_labels.jsonl -> confidence_labels.checkpoint.json
        """
        return output_path.with_suffix(".checkpoint.json")

    @staticmethod
    def _load_checkpoint(ckpt_path: Path) -> set[str]:
        """Load the set of already-processed query hashes from disk.

        Returns an empty set if the checkpoint file doesn't exist or is
        corrupted, so the run starts fresh.
        """
        if not ckpt_path.exists():
            return set()
        try:
            data = json.loads(ckpt_path.read_text(encoding="utf-8"))
            hashes = set(data.get("done_hashes", []))
            logger.info("Checkpoint loaded: %d queries already done", len(hashes))
            return hashes
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Corrupt checkpoint (%s); starting fresh", exc)
            return set()

    @staticmethod
    def _save_checkpoint(ckpt_path: Path, done_hashes: set[str]) -> None:
        """Persist the set of completed query hashes atomically.

        Writes to a temp file first, then renames, so a crash mid-write
        won't corrupt the checkpoint.
        """
        tmp = ckpt_path.with_suffix(".tmp")
        payload = json.dumps({"done_hashes": sorted(done_hashes)}, ensure_ascii=False)
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(ckpt_path)  # atomic on POSIX

    @staticmethod
    def _append_jsonl(labels: list[ConfidenceLabel], path: Path) -> None:
        """Append a batch of labels to the output JSONL file."""
        with open(path, "a", encoding="utf-8") as fh:
            for item in labels:
                payload = item.to_dict() if hasattr(item, "to_dict") else item
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        qa_pairs: Sequence[QAPair],
        output_path: Path | str | None = None,
    ) -> list[ConfidenceLabel]:
        """Score a batch of QA pairs and emit :class:`ConfidenceLabel` records.

        If a previous run was interrupted, already-scored queries are
        skipped automatically based on the checkpoint file.
        """
        if not qa_pairs:
            logger.warning("ConfidenceLabelGenerator.generate called with no QA pairs")
            return []

        output_path = self._resolve_output(output_path, "confidence_labels.jsonl")
        ckpt_path = self._checkpoint_path(output_path)

        # --- Resume: load checkpoint and filter out done queries ---
        done_hashes = self._load_checkpoint(ckpt_path)

        gen = self._get_generator()

        pool = list(qa_pairs)
        self.rng.shuffle(pool)
        pool = pool[: self.target_count]

        # Skip queries that were already processed in a previous run
        if done_hashes:
            before = len(pool)
            pool = [qa for qa in pool if _query_hash(qa.query) not in done_hashes]
            skipped = before - len(pool)
            logger.info("Resuming: skipped %d already-scored queries", skipped)

        if not pool:
            logger.info("All queries already processed; nothing to do")
            # Return labels from the existing output file
            return self._read_existing_labels(output_path)

        # --- Collect retrieval + generation results in a buffer ---
        # We accumulate a batch, score it, flush, repeat.
        buffer_queries: list[str] = []
        buffer_evidence: list[str] = []
        buffer_gold: list[str] = []
        buffer_ids: list[list[str]] = []

        all_labels: list[ConfidenceLabel] = []
        total = len(pool)
        t_start = time.perf_counter()

        for idx, qa in enumerate(pool, 1):
            logger.info("[%d/%d] Processing query: %s...", idx, total, qa.query[:80])
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

            buffer_queries.append(qa.query)
            buffer_evidence.append(answer)
            buffer_gold.append(qa.answer_with_citations)
            buffer_ids.append(ids)

            # --- Intermediate flush every checkpoint_interval queries ---
            if len(buffer_queries) >= self.checkpoint_interval:
                batch_labels = self._score_and_build(
                    buffer_queries, buffer_evidence, buffer_gold, buffer_ids
                )
                # Persist to disk immediately
                self._append_jsonl(batch_labels, output_path)
                done_hashes.update(_query_hash(q) for q in buffer_queries)
                self._save_checkpoint(ckpt_path, done_hashes)
                all_labels.extend(batch_labels)

                logger.info(
                    "Flushed %d labels (total %d) | elapsed %.1fs",
                    len(batch_labels),
                    len(all_labels),
                    time.perf_counter() - t_start,
                )

                # Reset buffer
                buffer_queries.clear()
                buffer_evidence.clear()
                buffer_gold.clear()
                buffer_ids.clear()

        # --- Flush remaining buffer ---
        if buffer_queries:
            batch_labels = self._score_and_build(
                buffer_queries, buffer_evidence, buffer_gold, buffer_ids
            )
            self._append_jsonl(batch_labels, output_path)
            done_hashes.update(_query_hash(q) for q in buffer_queries)
            self._save_checkpoint(ckpt_path, done_hashes)
            all_labels.extend(batch_labels)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "ConfidenceLabelGenerator wrote %d new labels to %s (%.1fs total)",
            len(all_labels),
            output_path,
            elapsed,
        )
        return all_labels

    # ------------------------------------------------------------------
    # Scoring + label construction (extracted from generate for reuse)
    # ------------------------------------------------------------------

    def _score_and_build(
        self,
        queries: list[str],
        evidence_answers: list[str],
        gold_answers: list[str],
        top5_ids: list[list[str]],
    ) -> list[ConfidenceLabel]:
        """Score a batch of evidence vs gold answers and build ConfidenceLabel objects.

        Factored out of ``generate`` so it can be called per-checkpoint-interval
        without duplicating the scoring + clamping logic.
        """
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
        return labels

    # ------------------------------------------------------------------
    # FIX 2: Fast batched NLI scoring
    # ------------------------------------------------------------------

    def _fast_similarity_scores(
        self, candidates: Sequence[str], references: Sequence[str]
    ) -> list[float]:
        """Compute soft similarity scores using a fast NLI cross-encoder.

        Processes pairs in batches of ``score_batch_size`` to stay within
        GPU memory limits. Falls back to BERTScore if the NLI model
        fails to load.
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

                    for row in raw:
                        row_arr = np.asarray(row, dtype=float)
                        if row_arr.ndim == 0 or row_arr.size == 1:
                            # Single score (e.g. regression head)
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
        self._nli_model = CrossEncoder(
            _DEFAULT_FAST_MODEL, max_length=512, device=self.device
        )
        logger.info("Initialized NLI model on DEVICE: %s", self.device)
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
    def _read_existing_labels(path: Path) -> list[ConfidenceLabel]:
        """Read back labels from an existing JSONL output file.

        Used when resuming and all queries are already done, so the
        caller still gets a populated return value.
        """
        labels: list[ConfidenceLabel] = []
        if not path.exists():
            return labels
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    labels.append(ConfidenceLabel(**data))
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("Skipping malformed JSONL line: %s", exc)
        logger.info("Loaded %d existing labels from %s", len(labels), path)
        return labels

    @staticmethod
    def _write_jsonl(items: Iterable[Any], path: Path) -> None:
        """Overwrite the output file with all items (used by legacy callers)."""
        with open(path, "w", encoding="utf-8") as fh:
            for item in items:
                payload = item.to_dict() if hasattr(item, "to_dict") else item
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")