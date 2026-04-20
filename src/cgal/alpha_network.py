"""
AegisRAG - Alpha Network

Lightweight 2-layer MLP that predicts the adaptive dense/sparse fusion
weight ``alpha in [0, 1]`` from cheap per-query features:

    [log_query_length, keyword_density, domain_hash,
     query_embedding_norm, has_exact_phrase]

Trained against a grid-searched oracle alpha (recall@k) via MSE.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Dimensionality of the hand-crafted feature vector.
ALPHA_FEATURE_DIM: int = 5


class AlphaNetwork(nn.Module):
    """Two-layer MLP ``Linear(5, 32) -> ReLU -> Linear(32, 1) -> Sigmoid``.

    Parameters
    ----------
    input_dim : int
        Feature vector size.  Defaults to :data:`ALPHA_FEATURE_DIM` = 5.
    hidden_dim : int
        Size of the hidden layer.
    alpha_min : float
        Lower clamp applied in :meth:`predict_alpha` when ``safety=True``.
    alpha_max : float
        Upper clamp applied in :meth:`predict_alpha` when ``safety=True``.
    safety_clamp : bool
        When True, predictions from :meth:`predict_alpha` are clamped to
        ``[alpha_min, alpha_max]`` so pure-dense or pure-sparse regimes are
        never reached at inference time.
    """

    def __init__(
        self,
        input_dim: int = ALPHA_FEATURE_DIM,
        hidden_dim: int = 32,
        alpha_min: float = 0.3,
        alpha_max: float = 0.7,
        safety_clamp: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if not 0.0 <= alpha_min < alpha_max <= 1.0:
            raise ValueError(
                f"Expected 0 <= alpha_min < alpha_max <= 1, "
                f"got ({alpha_min}, {alpha_max})."
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.safety_clamp = bool(safety_clamp)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(B, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)`` -- sigmoid-squashed alpha predictions.
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {features.shape[-1]}."
            )
        return self.net(features)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _domain_hash(domain: str, buckets: int = 32) -> float:
        """Deterministic hash of a domain string into ``[0, 1)``.

        Hashing the domain (rather than one-hot encoding) keeps the feature
        vector a fixed size 5, so the network schema is stable across a
        growing set of domains.  The ``buckets`` parameter controls
        granularity.
        """
        if not domain:
            return 0.0
        digest = hashlib.sha256(domain.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % max(buckets, 1)
        return bucket / max(buckets, 1)

    def extract_features(
        self,
        query: str,
        query_emb: np.ndarray,
        domain: str = "",
    ) -> torch.Tensor:
        """Compute the 5-dim feature vector.

        Feature order:

        1. ``log_query_length``  -- ``log(1 + num_tokens)``.
        2. ``keyword_density``   -- ``|unique_tokens| / |tokens|``.
        3. ``domain_hash``       -- deterministic hash of ``domain`` in ``[0, 1)``.
        4. ``query_emb_norm``    -- L2 norm of ``query_emb``.
        5. ``has_exact_phrase``  -- 1.0 if a quoted phrase is present else 0.0.

        Returns
        -------
        torch.Tensor
            Shape ``(1, 5)`` on the same device as this module's parameters.
        """
        if query_emb is None:
            raise ValueError("query_emb must be a non-None numpy array.")

        tokens = [t for t in re.split(r"\s+", query.strip()) if t]
        n_tokens = len(tokens)
        log_len = float(np.log1p(n_tokens))

        unique = len({t.lower() for t in tokens})
        keyword_density = float(unique / n_tokens) if n_tokens > 0 else 0.0

        dom_hash = self._domain_hash(domain)

        emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        emb_norm = float(np.linalg.norm(emb)) if emb.size > 0 else 0.0

        has_exact = 1.0 if re.search(r'"[^"]+"', query) else 0.0

        features = np.array(
            [log_len, keyword_density, dom_hash, emb_norm, has_exact],
            dtype=np.float32,
        )
        device = next(self.parameters()).device
        return torch.from_numpy(features).to(device=device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_alpha(
        self,
        query: str,
        query_emb: np.ndarray,
        domain: str = "",
    ) -> float:
        """Predict the optimal dense/sparse fusion weight for a query.

        Parameters
        ----------
        query : str
            Raw user query (used for length / phrase detection only).
        query_emb : np.ndarray
            Dense embedding of the query (used for norm feature).
        domain : str
            Optional domain/topic label to condition on.

        Returns
        -------
        float
            Predicted ``alpha`` in ``[0, 1]``.  Clamped to
            ``[alpha_min, alpha_max]`` when ``self.safety_clamp`` is True.
        """
        self.eval()
        features = self.extract_features(query, query_emb, domain)
        with torch.no_grad():
            raw = self.forward(features)
        alpha = float(raw.squeeze().cpu().item())
        if self.safety_clamp:
            alpha = float(np.clip(alpha, self.alpha_min, self.alpha_max))
        else:
            alpha = float(np.clip(alpha, 0.0, 1.0))
        logger.debug(
            "AlphaNetwork predicted alpha=%.4f (safety=%s) for query: %s",
            alpha,
            self.safety_clamp,
            query[:80],
        )
        return alpha

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _config_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "safety_clamp": self.safety_clamp,
        }

    def save(self, path: str | Path) -> None:
        """Persist weights + hyperparameters to a single torch file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self._config_dict(),
            },
            path,
        )
        logger.info("Saved AlphaNetwork to %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "AlphaNetwork":
        """Load a previously saved AlphaNetwork."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"AlphaNetwork checkpoint not found: {path}")
        payload = torch.load(path, map_location=map_location or "cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise RuntimeError(
                f"Invalid AlphaNetwork checkpoint at {path}: missing state_dict."
            )
        cfg = payload.get("config", {})
        model = cls(
            input_dim=int(cfg.get("input_dim", ALPHA_FEATURE_DIM)),
            hidden_dim=int(cfg.get("hidden_dim", 32)),
            alpha_min=float(cfg.get("alpha_min", 0.3)),
            alpha_max=float(cfg.get("alpha_max", 0.7)),
            safety_clamp=bool(cfg.get("safety_clamp", True)),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        logger.info("Loaded AlphaNetwork from %s", path)
        return model
