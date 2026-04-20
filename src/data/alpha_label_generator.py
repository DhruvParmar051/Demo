"""
AegisRAG - Alpha-Label Generator

For each query in a dev set, grid-searches alpha in [0.0, 0.1, ..., 1.0]
and records the value that maximises recall@20 against the provided
``gold_chunk_ids``.

FIX 4: Dense and sparse retrieval are now called ONCE per query.
Hybrid fusion across all alpha values is performed locally using the
cached score vectors, reducing retriever calls from N_grid per query
to 2 per query.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.data.schema import AlphaLabel, QAPair
from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    s_min = scores.min()
    s_max = scores.max()
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores) if s_max == 0 else np.ones_like(scores)
    return (scores - s_min) / (s_max - s_min)


class AlphaLabelGenerator:
    """Grid-search optimal alpha per query for the alpha network.

    FIX 4: Retrieves dense and sparse scores once per query, then
    performs fusion locally for every alpha value — eliminating
    N_grid redundant retriever round-trips per query.

    Parameters
    ----------
    retriever : object
        Must expose ``vector_store`` (ChromaVectorStore) and
        ``bm25_index`` (BM25Index) attributes for direct access, or
        fall back to ``retrieve(query, top_k, alpha)`` calls (slower).
    grid : iterable of float, optional
        Alpha values to evaluate (default ``[0.0, 0.1, ..., 1.0]``).
    top_k : int
        Recall cutoff (default 20).
    seed : int
        Deterministic RNG seed (default 42).
    limit : int or None
        Target label count (default 1000 or config).
    """

    def __init__(
        self,
        retriever: Any,
        grid: Sequence[float] | None = None,
        top_k: int = 20,
        seed: int = 42,
        limit: int | None = None,
    ) -> None:
        set_seed(seed)
        self.seed = seed
        self.rng = random.Random(seed)

        cfg = get_config()
        self.cfg = cfg

        self.retriever = retriever
        self.grid = tuple(grid) if grid is not None else tuple(round(i * 0.1, 1) for i in range(11))
        self.top_k = int(top_k)

        if limit is not None:
            self.target_count = int(limit)
        else:
            configured = int(getattr(cfg.synthetic_data, "alpha_labels", 1000))
            self.target_count = max(configured, 1000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        qa_pairs: Sequence[QAPair],
        output_path: Path | str | None = None,
    ) -> list[AlphaLabel]:
        if not qa_pairs:
            logger.warning("AlphaLabelGenerator.generate called with no QA pairs")
            return []

        output_path = self._resolve_output(output_path, "alpha_labels.jsonl")

        pool = [qa for qa in qa_pairs if qa.gold_chunk_ids]
        self.rng.shuffle(pool)
        pool = pool[: self.target_count]

        labels: list[AlphaLabel] = []
        for qa in pool:
            gold = set(qa.gold_chunk_ids)
            # FIX 4: Retrieve dense and sparse scores once, then fuse locally
            label = self._compute_label_fast(qa.query, gold)
            if label is not None:
                labels.append(label)

            if len(labels) % 100 == 0:
                logger.info(
                    "AlphaLabelGenerator: %d / %d labels",
                    len(labels),
                    self.target_count,
                )

        self._write_jsonl(labels, output_path)
        logger.info(
            "AlphaLabelGenerator wrote %d labels to %s", len(labels), output_path
        )
        return labels

    # ------------------------------------------------------------------
    # FIX 4: Single-pass retrieval with local fusion
    # ------------------------------------------------------------------

    def _compute_label_fast(
        self, query: str, gold: set[str]
    ) -> AlphaLabel | None:
        """Compute the best alpha by retrieving once and fusing locally.

        Dense and sparse retrieval are each called once. For every alpha
        in the grid, the fused scores are computed in-memory.
        """
        # Try direct access to sub-components (fast path)
        vector_store = getattr(self.retriever, "vector_store", None)
        bm25_index = getattr(self.retriever, "bm25_index", None)

        if vector_store is not None and bm25_index is not None:
            return self._fuse_locally(query, gold, vector_store, bm25_index)

        # Fallback: call retriever directly for each alpha (slower, original behaviour)
        logger.warning(
            "Retriever does not expose vector_store/bm25_index; "
            "falling back to per-alpha retrieval calls."
        )
        return self._fuse_via_retriever(query, gold)

    def _fuse_locally(
        self,
        query: str,
        gold: set[str],
        vector_store: Any,
        bm25_index: Any,
    ) -> AlphaLabel | None:
        """Retrieve dense + sparse once, then compute recall for all alpha values."""
        try:
            dense_results = vector_store.query(query_text=query, top_k=self.top_k * 2)
            sparse_results = bm25_index.query(query_text=query, top_k=self.top_k * 2)
        except Exception as exc:
            logger.warning("Retrieval failed for query '%s': %s", query[:60], exc)
            return None

        # Build unified chunk_id -> score maps
        dense_map: dict[str, float] = {c.chunk_id: s for c, s in dense_results}
        sparse_map: dict[str, float] = {c.chunk_id: s for c, s in sparse_results}
        all_ids = list(set(dense_map) | set(sparse_map))

        if not all_ids:
            return None

        dense_arr = np.array([dense_map.get(cid, 0.0) for cid in all_ids])
        sparse_arr = np.array([sparse_map.get(cid, 0.0) for cid in all_ids])
        dense_norm = _min_max_normalize(dense_arr)
        sparse_norm = _min_max_normalize(sparse_arr)

        # Score each alpha value locally — no retriever calls
        grid_scores: dict[str, float] = {}
        best_alpha = self.grid[0]
        best_recall = -1.0

        for alpha in self.grid:
            combined = float(alpha) * dense_norm + (1.0 - float(alpha)) * sparse_norm
            top_idx = combined.argsort()[::-1][: self.top_k]
            retrieved_ids = {all_ids[i] for i in top_idx}

            recall = len(retrieved_ids & gold) / float(len(gold)) if gold else 0.0
            grid_scores[f"{alpha:.2f}"] = recall

            if recall > best_recall:
                best_recall = recall
                best_alpha = float(alpha)

        return AlphaLabel(
            query=query,
            optimal_alpha=float(best_alpha),
            grid_scores=grid_scores,
        )

    def _fuse_via_retriever(self, query: str, gold: set[str]) -> AlphaLabel | None:
        """Fallback: call retriever for each alpha (original behaviour)."""
        grid_scores: dict[str, float] = {}
        best_alpha = self.grid[0]
        best_recall = -1.0

        for alpha in self.grid:
            try:
                results = self.retriever.retrieve(
                    query, top_k=self.top_k, alpha=float(alpha)
                )
            except TypeError:
                results = self.retriever.retrieve(query, top_k=self.top_k)
            except Exception as exc:
                logger.warning("Retriever failed at alpha=%.2f: %s", alpha, exc)
                continue

            retrieved_ids = {c.chunk_id for c, _ in results}
            recall = len(retrieved_ids & gold) / float(len(gold)) if gold else 0.0
            grid_scores[f"{alpha:.2f}"] = recall

            if recall > best_recall:
                best_recall = recall
                best_alpha = float(alpha)

        if not grid_scores:
            return None

        return AlphaLabel(
            query=query,
            optimal_alpha=float(best_alpha),
            grid_scores=grid_scores,
        )

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