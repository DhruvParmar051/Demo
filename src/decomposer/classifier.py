"""
AegisRAG - Decomposition Classifier

Binary classifier that decides whether a query should be decomposed into
multiple sub-queries.  Operates on the CLS token of a BGE-m3 embedding
(dimension 1024 by default) and emits a single sigmoid probability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DecompositionClassifier(nn.Module):
    """Linear binary classifier on a frozen BGE-m3 CLS embedding.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the input embedding (1024 for BGE-m3).
    threshold : float
        Decision boundary for :meth:`is_multi_part`.  Defaults to 0.5.
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError(
                f"embedding_dim must be positive, got {embedding_dim}."
            )
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"threshold must be in (0, 1), got {threshold}."
            )

        self.embedding_dim = int(embedding_dim)
        self.threshold = float(threshold)

        self.linear = nn.Linear(self.embedding_dim, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, query_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass returning *sigmoid* probabilities.

        Parameters
        ----------
        query_emb : torch.Tensor
            Shape ``(B, embedding_dim)`` or ``(embedding_dim,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)`` in ``[0, 1]``.
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if query_emb.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected last dim {self.embedding_dim}, "
                f"got {query_emb.shape[-1]}."
            )
        logits = self.linear(query_emb).squeeze(-1)
        return torch.sigmoid(logits)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def is_multi_part(
        self,
        query_emb: np.ndarray | torch.Tensor,
    ) -> tuple[bool, float]:
        """Predict multi-part vs. atomic for a single query embedding.

        Parameters
        ----------
        query_emb : array-like
            Shape ``(embedding_dim,)`` or ``(1, embedding_dim)``.

        Returns
        -------
        (bool, float)
            ``(is_multi_part, probability)`` where the bool compares the
            probability against ``self.threshold``.
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(query_emb, np.ndarray):
            q = torch.from_numpy(np.ascontiguousarray(query_emb))
        elif isinstance(query_emb, torch.Tensor):
            q = query_emb
        else:
            q = torch.tensor(query_emb)
        q = q.to(device=device, dtype=torch.float32)
        if q.dim() == 1:
            q = q.unsqueeze(0)

        with torch.no_grad():
            prob = self.forward(q)
        prob_val = float(prob.squeeze().cpu().item())
        return prob_val >= self.threshold, prob_val

    # ------------------------------------------------------------------
    # Training step (single mini-batch)
    # ------------------------------------------------------------------

    def train_step(
        self,
        query_embs: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        pos_weight: float | None = None,
    ) -> float:
        """Single training step using BCE-with-logits loss.

        Parameters
        ----------
        query_embs : torch.Tensor
            Shape ``(B, embedding_dim)``.
        labels : torch.Tensor
            Shape ``(B,)``, float in ``{0.0, 1.0}``.
        optimizer : torch.optim.Optimizer
            Already wrapping this module's parameters.
        pos_weight : float or None
            Positive-class weight for imbalanced data.

        Returns
        -------
        float
            Scalar loss value for logging.
        """
        self.train()
        optimizer.zero_grad()

        if query_embs.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, "
                f"got {query_embs.shape[-1]}."
            )
        logits = self.linear(query_embs).squeeze(-1)
        target = labels.float().to(logits.device)

        pw = None
        if pos_weight is not None:
            pw = torch.tensor(
                [float(pos_weight)], device=logits.device, dtype=logits.dtype
            )
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _config_dict(self) -> dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "threshold": self.threshold,
        }

    def save(self, path: str | Path) -> None:
        """Persist weights + threshold to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "config": self._config_dict()},
            path,
        )
        logger.info("Saved DecompositionClassifier to %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "DecompositionClassifier":
        """Load a saved classifier."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"DecompositionClassifier checkpoint not found: {path}"
            )
        payload = torch.load(path, map_location=map_location or "cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise RuntimeError(
                f"Invalid DecompositionClassifier checkpoint at {path}: "
                "missing state_dict."
            )
        cfg = payload.get("config", {})
        model = cls(
            embedding_dim=int(cfg.get("embedding_dim", 1024)),
            threshold=float(cfg.get("threshold", 0.5)),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        logger.info("Loaded DecompositionClassifier from %s", path)
        return model
