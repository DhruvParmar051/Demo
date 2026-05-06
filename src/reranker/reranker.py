"""
AegisRAG - ColBERT Reranker

Re-scores retrieved chunks using Jina-ColBERT-v2 loaded as a
cross-encoder for pairwise relevance scoring.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.schema import ChunkRecord
from src.utils.config import get_config
from src.utils.device import get_device

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _mask_mps():
    """Temporarily report MPS as unavailable.

    jina-colbert-v2 (trust_remote_code) calls torch.backends.mps.is_available()
    during __init__ and moves itself to MPS, triggering a hard LLVM abort on
    GQA matmul shapes that MPS cannot compile.  Masking MPS keeps the model
    on CPU for the duration of initialisation and inference.
    """
    original = torch.backends.mps.is_available
    torch.backends.mps.is_available = lambda: False
    try:
        yield
    finally:
        torch.backends.mps.is_available = original


class ColBERTReranker:
    """Cross-encoder reranker backed by Jina-ColBERT-v2.

    The model is loaded via HuggingFace ``transformers`` as a sequence-
    classification model.  Each (query, passage) pair is scored and the
    top-k results are returned in descending relevance order.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the reranker.
    checkpoint_path : str or None
        Path to a fine-tuned checkpoint directory.  When provided this
        overrides *model_name* for weight loading while keeping the
        same tokenizer.
    max_length : int or None
        Maximum token length for the model input.  Falls back to the
        config value ``models.reranker.max_seq_length`` if not given.
    batch_size : int
        Number of (query, passage) pairs to score in a single forward pass.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        checkpoint_path: str | None = None,
        max_length: int | None = None,
        batch_size: int = 32,
        inference_max_length: int = 512,
    ) -> None:
        cfg = get_config()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = (
            max_length
            if max_length is not None
            else int(cfg.models.reranker.max_seq_length)
        )
        self._inference_max_length = min(self.max_length, inference_max_length)

        self.device = get_device(cfg.device.preferred_device)

        load_path = checkpoint_path if checkpoint_path else model_name

        logger.info(
            "Loading reranker model from %s on %s (inference_max_length=%d)",
            load_path,
            self.device,
            self._inference_max_length,
        )

        # Load tokenizer from the same location as the weights. When a
        # fine-tuned checkpoint ships with a modified tokenizer (added
        # special tokens, resized vocab), using the base HF tokenizer leads
        # to silent ID mismatches at inference.
        pt_weights = Path(load_path) / "model.pt" if Path(load_path).is_dir() else None
        has_state_dict = pt_weights is not None and pt_weights.exists()
        has_hf_config = (Path(load_path) / "config.json").exists() if Path(load_path).is_dir() else False

        with _mask_mps():
            if has_state_dict and not has_hf_config:
                logger.info(
                    "Checkpoint %s has model.pt but no config.json; "
                    "loading base model '%s' then applying fine-tuned weights.",
                    load_path, model_name,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                state = torch.load(str(pt_weights), map_location="cpu")
                if isinstance(state, dict) and "model" in state and not any(
                    k.startswith("encoder.") for k in state
                ):
                    state = state["model"]
                base_keys = set(self.model.state_dict().keys())
                if any(k.startswith("encoder.") for k in state) and not any(
                    k.startswith("encoder.") for k in base_keys
                ):
                    state = {
                        k.replace("encoder.", "bert.", 1) if k.startswith("encoder.") else k: v
                        for k, v in state.items()
                    }
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                if missing:
                    logger.warning("Missing keys when loading fine-tuned weights: %s", missing[:5])
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    load_path, trust_remote_code=True
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    load_path, trust_remote_code=True
                )

            self.model.to(self.device)
            self.model.eval()
        logger.info("Reranker ready (%s)", load_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: list[tuple[ChunkRecord, float]],
        top_k: int = 5,
    ) -> list[tuple[ChunkRecord, float]]:
        """Re-score and re-order chunks by cross-encoder relevance.

        Parameters
        ----------
        query : str
            The user query.
        chunks : list of (ChunkRecord, float)
            Candidate chunks (typically from the hybrid retriever).
            The original retrieval score is discarded; only the
            reranker score is used for ranking.
        top_k : int
            Number of results to return.

        Returns
        -------
        list of (ChunkRecord, float)
            Top-k chunks sorted by descending reranker score.
        """
        if not chunks:
            return []

        pairs: list[tuple[str, str]] = [
            (query, chunk.text) for chunk, _ in chunks
        ]

        scores = self._score_pairs(pairs)

        # Pair scores back with chunks and sort
        scored: list[tuple[ChunkRecord, float]] = [
            (chunk, float(score))
            for (chunk, _), score in zip(chunks, scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_pairs(
        self, pairs: list[tuple[str, str]]
    ) -> np.ndarray:
        """Score a list of (query, passage) pairs in batches.

        Returns an array of shape ``(len(pairs),)`` with relevance
        logits (higher = more relevant).
        """
        all_scores: list[float] = []

        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            queries = [p[0] for p in batch]
            passages = [p[1] for p in batch]

            encoded = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self._inference_max_length,
                return_tensors="pt",
            ).to(self.device)

            with _mask_mps(), torch.no_grad():
                outputs = self.model(**encoded)

            # outputs.logits shape: (batch, num_labels)
            # For binary / single-label rerankers, take the last logit or
            # the single score depending on model head output size.
            logits = outputs.logits
            if logits.shape[-1] == 1:
                batch_scores = logits.squeeze(-1)
            else:
                # If the model outputs >1 logits (e.g. [not_relevant, relevant]),
                # take the "relevant" logit.
                batch_scores = logits[:, -1]

            all_scores.extend(batch_scores.cpu().tolist())

        return np.array(all_scores, dtype=np.float64)
