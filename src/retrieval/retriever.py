"""
AegisRAG - Hybrid Retriever

Fuses dense (BGE-m3 / ChromaDB) and sparse (BM25) retrieval with an
adaptive alpha weight.  When an ``alpha_network`` is provided, the
per-query optimal dense/sparse weight is predicted by a learned 2-layer
MLP; otherwise a fixed alpha (default 0.5) is used.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch

from src.data.schema import ChunkRecord
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.config import get_config

logger = logging.getLogger(__name__)


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize an array of scores to [0, 1] via min-max scaling."""
    s_min = scores.min()
    s_max = scores.max()
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores) if s_max == 0 else np.ones_like(scores)
    return (scores - s_min) / (s_max - s_min)


class HybridRetriever:
    """Hybrid dense + sparse retriever with adaptive alpha fusion.

    Parameters
    ----------
    vector_store : ChromaVectorStore
        Dense retrieval backend.
    bm25_index : BM25Index
        Sparse retrieval backend.
    alpha_network : torch.nn.Module or None
        Optional learned MLP that predicts the dense weight from query
        features.  Expected to accept a tensor of shape ``(1, 12)`` and
        return a scalar in ``[0, 1]`` (after sigmoid).
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_index: BM25Index,
        alpha_network: torch.nn.Module | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.alpha_network = alpha_network
        self._retrieval_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="hybrid_retrieval",
        )

        cfg = get_config()
        self._default_alpha: float = float(cfg.retrieval.initial_alpha)
        self._alpha_min: float = float(cfg.retrieval.alpha_min)
        self._alpha_max: float = float(cfg.retrieval.alpha_max)

    # ------------------------------------------------------------------
    # Main retrieval entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        alpha: float | None = None,
        filters: dict[str, Any] | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> list[tuple[ChunkRecord, float]]:
        """Run hybrid retrieval and return fused, deduplicated results.

        Parameters
        ----------
        query : str
            User query string.
        top_k : int
            Number of results to return after fusion.
        alpha : float or None
            Dense weight (0 = pure BM25, 1 = pure dense).
            * If provided, used as-is (clamped to [alpha_min, alpha_max]).
            * If *None* and *alpha_network* exists, predicted from query.
            * If *None* and no network, ``cfg.retrieval.initial_alpha`` is used.
        filters : dict or None
            Optional ChromaDB metadata filters forwarded to the vector store.

        Returns
        -------
        list of (ChunkRecord, float)
            Top-k chunks sorted by descending fused score.
        """
        # --- Determine alpha -------------------------------------------------
        if alpha is not None:
            alpha = float(np.clip(alpha, self._alpha_min, self._alpha_max))
        elif self.alpha_network is not None:
            alpha = self._predict_alpha(query)
        else:
            alpha = self._default_alpha

        logger.debug("Hybrid retrieval alpha=%.3f for query: %s", alpha, query[:80])

        # --- Fetch candidates from both backends (request extra to help dedup)
        fetch_k = top_k * 2

        def _dense_retrieve():
            if query_embedding is not None and hasattr(self.vector_store, "query_by_embedding"):
                return self.vector_store.query_by_embedding(
                    query_embedding, top_k=fetch_k, filters=filters
                )
            return self.vector_store.query(
                query_text=query, top_k=fetch_k, filters=filters
            )

        def _sparse_retrieve():
            return self.bm25_index.query(query_text=query, top_k=fetch_k)

        dense_future = self._retrieval_executor.submit(_dense_retrieve)
        sparse_future = self._retrieval_executor.submit(_sparse_retrieve)

        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

        # --- Build score maps ------------------------------------------------
        dense_map: dict[str, float] = {
            c.chunk_id: s for c, s in dense_results
        }
        sparse_map: dict[str, float] = {
            c.chunk_id: s for c, s in sparse_results
        }

        # Collect all unique chunk_ids and their ChunkRecord objects
        chunk_lookup: dict[str, ChunkRecord] = {}
        for c, _ in dense_results:
            chunk_lookup[c.chunk_id] = c
        for c, _ in sparse_results:
            chunk_lookup.setdefault(c.chunk_id, c)

        all_ids = list(chunk_lookup.keys())
        if not all_ids:
            return []

        # --- Normalize scores to [0, 1] --------------------------------------
        dense_scores_raw = np.array(
            [dense_map.get(cid, 0.0) for cid in all_ids]
        )
        sparse_scores_raw = np.array(
            [sparse_map.get(cid, 0.0) for cid in all_ids]
        )

        dense_norm = _min_max_normalize(dense_scores_raw)
        sparse_norm = _min_max_normalize(sparse_scores_raw)

        # --- Fuse ------------------------------------------------------------
        combined = alpha * dense_norm + (1.0 - alpha) * sparse_norm

        # --- Sort descending, take top_k ------------------------------------
        ranked_indices = combined.argsort()[::-1][:top_k]

        results: list[tuple[ChunkRecord, float]] = []
        for idx in ranked_indices:
            cid = all_ids[idx]
            results.append((chunk_lookup[cid], float(combined[idx])))

        return results

    # ------------------------------------------------------------------
    # Alpha prediction
    # ------------------------------------------------------------------

    def _predict_alpha(self, query: str) -> float:
        """Use the alpha network to predict the optimal dense weight."""
        alpha_net = self.alpha_network
        if alpha_net is None:
            return self._default_alpha

        # Get query embedding for norm feature
        try:
            query_emb = self.vector_store.model.encode(
                [query], normalize_embeddings=True
            )[0]
        except Exception:
            import numpy as _np
            query_emb = _np.zeros(1, dtype=_np.float32)

        # Delegate feature extraction + inference to the network itself
        # so feature dim always matches whatever checkpoint is loaded.
        alpha = alpha_net.predict_alpha(query, query_emb, domain="")
        alpha = float(__import__("numpy").clip(alpha, self._alpha_min, self._alpha_max))
        logger.debug("Alpha network predicted alpha=%.3f", alpha)
        return alpha

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_query_features(self, query: str) -> dict[str, float]:
        """Compute lightweight features for the alpha network.

        Returns
        -------
        dict with keys:
            - ``length``: number of whitespace-separated tokens (log-scaled).
            - ``keyword_density``: ratio of unique words to total words.
            - ``embedding_norm``: L2 norm of the query embedding.
            - ``has_exact_phrase``: 1.0 if the query contains a quoted phrase,
              else 0.0.
        """
        tokens = query.lower().split()
        length = float(np.log1p(len(tokens)))

        unique = set(tokens)
        keyword_density = len(unique) / max(len(tokens), 1)

        # Embedding norm via the dense model. Use normalize_embeddings=True
        # to match how embeddings are stored in the vector DB — otherwise the
        # alpha network sees a different distribution at train vs. inference.
        embedding = self.vector_store.model.encode(
            [query], normalize_embeddings=True
        )
        embedding_norm = float(np.linalg.norm(embedding[0]))

        # Check for quoted exact-phrase markers
        has_exact_phrase = 1.0 if re.search(r'"[^"]+"', query) else 0.0

        return {
            "length": length,
            "keyword_density": keyword_density,
            "embedding_norm": embedding_norm,
            "has_exact_phrase": has_exact_phrase,
        }

    def __del__(self) -> None:
        try:
            self._retrieval_executor.shutdown(wait=False)
        except Exception:
            pass
