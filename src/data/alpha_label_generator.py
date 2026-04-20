"""
AegisRAG - Alpha-Label Generator

For each query in a dev set, grid-searches alpha in [0.0, 0.1, ..., 1.0]
and records the value that maximises recall@20 against the provided
``gold_chunk_ids``. The labels are used to train the adaptive alpha
fusion network.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.schema import AlphaLabel, QAPair
from src.utils.config import get_config
from src.utils.determinism import set_seed

logger = logging.getLogger(__name__)


class AlphaLabelGenerator:
    """Grid-search optimal alpha per query for the alpha network.

    Parameters
    ----------
    retriever : object
        Must expose ``retrieve(query, top_k, alpha) -> list[(ChunkRecord, float)]``.
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
            self.target_count = 1000 if configured >= 1000 else configured

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        qa_pairs: Sequence[QAPair],
        output_path: Path | str | None = None,
    ) -> list[AlphaLabel]:
        """Produce :class:`AlphaLabel` records for up to ``target_count`` queries."""
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
            grid_scores: dict[str, float] = {}
            best_alpha = self.grid[0]
            best_recall = -1.0

            for alpha in self.grid:
                try:
                    results = self.retriever.retrieve(
                        qa.query, top_k=self.top_k, alpha=float(alpha)
                    )
                except TypeError:
                    # Retriever may not accept ``alpha`` kwarg.
                    results = self.retriever.retrieve(qa.query, top_k=self.top_k)
                except Exception as exc:
                    logger.warning(
                        "Retriever failed at alpha=%.2f: %s", alpha, exc
                    )
                    continue

                retrieved_ids = {c.chunk_id for c, _ in results}
                if not gold:
                    recall = 0.0
                else:
                    recall = len(retrieved_ids & gold) / float(len(gold))

                grid_scores[f"{alpha:.2f}"] = recall
                if recall > best_recall:
                    best_recall = recall
                    best_alpha = float(alpha)

            labels.append(
                AlphaLabel(
                    query=qa.query,
                    optimal_alpha=float(best_alpha),
                    grid_scores=grid_scores,
                )
            )

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
