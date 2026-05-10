"""
AegisRAG - Alpha Network

MLP that predicts the adaptive dense/sparse fusion weight ``alpha in [0, 1]``
from 12 cheap per-query features:

    [log_query_length, keyword_density, domain_hash, query_emb_norm,
     has_exact_phrase, is_wh_question, numeric_density, avg_word_len,
     stopword_ratio, capitalized_ratio, has_definition_cue, verb_density]

Trained against a grid-searched oracle alpha (recall@k) via Huber loss.
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
ALPHA_FEATURE_DIM: int = 12

# Common English stopwords (lightweight, no NLTK dependency)
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "about",
    "and", "or", "but", "if", "not", "that", "this", "it", "its", "i",
    "you", "he", "she", "we", "they", "what", "which", "who", "when",
    "where", "why", "how",
})

# Definition / explanation cue words → sparse BM25 tends to win
_DEFINITION_CUES = frozenset({
    "what", "define", "definition", "meaning", "means", "explain",
    "describe", "described", "difference", "between", "vs", "versus",
    "type", "types", "kind", "kinds", "category", "categories",
})

# Common English verbs (rough signal for procedural / how-to queries)
_COMMON_VERBS = frozenset({
    "apply", "calculate", "compute", "determine", "find", "get", "make",
    "use", "file", "submit", "report", "claim", "request", "obtain",
    "qualify", "meet", "exceed", "increase", "decrease", "provide",
    "complete", "include", "exclude", "pay", "receive", "require",
})


class AlphaNetwork(nn.Module):
    """3-layer MLP with BatchNorm + Dropout for robust alpha prediction.

    Architecture:
        Linear(12, 64) -> BN -> ReLU -> Dropout(0.2)
        -> Linear(64, 32) -> ReLU
        -> Linear(32, 1) -> Sigmoid

    Parameters
    ----------
    input_dim : int
        Feature vector size.  Defaults to :data:`ALPHA_FEATURE_DIM` = 12.
    hidden_dim : int
        Size of the first hidden layer (second is hidden_dim // 2).
    alpha_min : float
        Lower clamp applied in :meth:`predict_alpha` when ``safety=True``.
    alpha_max : float
        Upper clamp applied in :meth:`predict_alpha` when ``safety=True``.
    safety_clamp : bool
        When True, predictions from :meth:`predict_alpha` are clamped to
        ``[alpha_min, alpha_max]`` so extreme regimes are avoided.
    """

    def __init__(
        self,
        input_dim: int = ALPHA_FEATURE_DIM,
        hidden_dim: int = 64,
        alpha_min: float = 0.4,
        alpha_max: float = 0.8,
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

        h2 = max(hidden_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
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
        """Deterministic hash of a domain string into ``[0, 1)``."""
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
        """Compute the 12-dim feature vector.

        Feature order:

        1.  ``log_query_length``    -- ``log(1 + num_tokens)``
        2.  ``keyword_density``     -- ``|unique_tokens| / |tokens|``
        3.  ``domain_hash``         -- deterministic hash of ``domain`` in [0,1)
        4.  ``query_emb_norm``      -- L2 norm of ``query_emb``
        5.  ``has_exact_phrase``    -- 1.0 if a quoted phrase is present
        6.  ``is_wh_question``      -- 1.0 if starts with what/who/when/where/why/how
        7.  ``numeric_density``     -- fraction of tokens that are numeric
        8.  ``avg_word_len``        -- average character length of tokens (normalised /10)
        9.  ``stopword_ratio``      -- fraction of tokens that are stopwords
        10. ``capitalized_ratio``   -- fraction of tokens starting with uppercase
        11. ``has_definition_cue``  -- 1.0 if a definition/explanation keyword present
        12. ``verb_density``        -- fraction of tokens matching common action verbs

        Returns
        -------
        torch.Tensor
            Shape ``(1, 12)`` on the same device as this module's parameters.
        """
        if query_emb is None:
            raise ValueError("query_emb must be a non-None numpy array.")

        raw_tokens = re.split(r"\s+", query.strip())
        tokens = [t for t in raw_tokens if t]
        n = len(tokens) or 1
        lower_tokens = [t.lower().rstrip("?.,!;:") for t in tokens]

        # 1. log query length
        log_len = float(np.log1p(n))

        # 2. keyword density (unique ratio)
        keyword_density = len(set(lower_tokens)) / n

        # 3. domain hash
        dom_hash = self._domain_hash(domain)

        # 4. embedding norm
        emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        emb_norm = float(np.linalg.norm(emb)) if emb.size > 0 else 0.0

        # 5. exact phrase (quoted)
        has_exact = 1.0 if re.search(r'"[^"]+"', query) else 0.0

        # 6. wh-question
        wh_words = {"what", "who", "when", "where", "why", "how", "which", "whose"}
        is_wh = 1.0 if lower_tokens and lower_tokens[0] in wh_words else 0.0

        # 7. numeric density
        numeric_count = sum(1 for t in lower_tokens if re.fullmatch(r"\d[\d,.$%]*", t))
        numeric_density = numeric_count / n

        # 8. avg word length (normalised)
        avg_word_len = float(np.mean([len(t) for t in tokens])) / 10.0

        # 9. stopword ratio
        stopword_count = sum(1 for t in lower_tokens if t in _STOPWORDS)
        stopword_ratio = stopword_count / n

        # 10. capitalised ratio (signals proper nouns / acronyms → BM25-friendly)
        cap_count = sum(1 for t in tokens if t and t[0].isupper())
        capitalized_ratio = cap_count / n

        # 11. definition cue
        has_def_cue = 1.0 if any(t in _DEFINITION_CUES for t in lower_tokens) else 0.0

        # 12. verb density (action / procedural queries → dense helpful)
        verb_count = sum(1 for t in lower_tokens if t in _COMMON_VERBS)
        verb_density = verb_count / n

        features = np.array([
            log_len, keyword_density, dom_hash, emb_norm, has_exact,
            is_wh, numeric_density, avg_word_len, stopword_ratio,
            capitalized_ratio, has_def_cue, verb_density,
        ], dtype=np.float32)

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
