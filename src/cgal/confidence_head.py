"""
AegisRAG - Confidence Head

Small MLP that consumes a query embedding, the mean-pooled top-k evidence
embedding, and optional scalar features to produce:

* a scalar ``confidence`` in ``[0, 1]`` used by the CGAL loop router, and
* a 4-way ``tool_logits`` distribution over
  ``[AnswerDirect, SearchKB, GetPolicy, CreateTicket]``.

The head is trained with soft labels (KL divergence) and calibrated post-hoc
via temperature scaling.
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


TOOL_NAMES: tuple[str, str, str, str] = (
    "AnswerDirect",
    "SearchKB",
    "GetPolicy",
    "CreateTicket",
)


class ConfidenceHead(nn.Module):
    """Joint confidence + tool-policy head.

    Architecture
    ------------
    ``concat([q_emb (D), mean_evidence_emb (D), extra_features (F)])``
        -> ``Linear(2D+F, 512)`` -> ReLU -> Dropout(0.1)
        -> ``Linear(512, 128)`` -> ReLU
        -> two heads:
            * ``Linear(128, 1)`` -> Sigmoid   (confidence)
            * ``Linear(128, 4)`` -> Softmax   (tool policy)

    Parameters
    ----------
    embedding_dim : int
        Dimensionality ``D`` of each input embedding (1024 for BGE-m3).
    extra_feature_dim : int
        Dimensionality ``F`` of optional scalar features.  Pass ``0`` to
        disable extra features entirely.
    dropout : float
        Dropout probability between the two hidden layers.
    num_tools : int
        Number of tool classes.  Must remain 4 unless
        :data:`TOOL_NAMES` is updated consistently.
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        extra_feature_dim: int = 0,
        dropout: float = 0.1,
        num_tools: int = 4,
    ) -> None:
        super().__init__()
        if num_tools != len(TOOL_NAMES):
            raise ValueError(
                f"num_tools must equal len(TOOL_NAMES)={len(TOOL_NAMES)}, "
                f"got {num_tools}."
            )

        self.embedding_dim = int(embedding_dim)
        self.extra_feature_dim = int(extra_feature_dim)
        self.num_tools = int(num_tools)

        in_dim = 2 * self.embedding_dim + self.extra_feature_dim

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.confidence_head = nn.Linear(128, 1)
        self.tool_policy_head = nn.Linear(128, self.num_tools)

        # Temperature scaling for confidence calibration.  Held as a buffer
        # so it moves with the model but is not optimized alongside the
        # trunk -- it is fit post-hoc on a held-out set.
        self.register_buffer(
            "temperature", torch.tensor(1.0, dtype=torch.float32)
        )
        self._calibrated: bool = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query_emb: torch.Tensor,
        evidence_emb: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        Parameters
        ----------
        query_emb : torch.Tensor
            Shape ``(B, D)``.  Pre-normalized BGE-m3 query embedding.
        evidence_emb : torch.Tensor
            Shape ``(B, D)``.  Mean-pooled top-k evidence embedding.
        features : torch.Tensor or None
            Shape ``(B, F)``.  Optional scalar features.  Must be provided
            when ``extra_feature_dim > 0``.

        Returns
        -------
        dict
            ``{"confidence": tensor (B,), "tool_logits": tensor (B, num_tools)}``.
            The confidence is returned *post-sigmoid* and *post-temperature*.
            Tool logits are raw (pre-softmax) -- apply softmax at the call site
            for a probability distribution.
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if evidence_emb.dim() == 1:
            evidence_emb = evidence_emb.unsqueeze(0)

        if query_emb.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"query_emb last dim {query_emb.shape[-1]} does not match "
                f"embedding_dim={self.embedding_dim}."
            )
        if evidence_emb.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"evidence_emb last dim {evidence_emb.shape[-1]} does not "
                f"match embedding_dim={self.embedding_dim}."
            )

        parts = [query_emb, evidence_emb]
        if self.extra_feature_dim > 0:
            if features is None:
                raise ValueError(
                    "features tensor required when extra_feature_dim > 0."
                )
            feat: torch.Tensor = features
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            if feat.shape[-1] != self.extra_feature_dim:
                raise ValueError(
                    f"features last dim {feat.shape[-1]} does not match "
                    f"extra_feature_dim={self.extra_feature_dim}."
                )
            parts.append(feat)
        elif features is not None and features.numel() > 0:
            # Silently ignore features if the head was configured without them,
            # but log so users notice the mismatch.
            logger.debug(
                "ConfidenceHead received features but extra_feature_dim=0; "
                "ignoring."
            )

        x = torch.cat(parts, dim=-1)
        hidden = self.trunk(x)

        # Confidence: logit -> temperature-scaled sigmoid
        conf_logit = self.confidence_head(hidden).squeeze(-1)
        temp = torch.clamp(self.temperature, min=1e-3)
        confidence = torch.sigmoid(conf_logit / temp)

        tool_logits = self.tool_policy_head(hidden)

        return {"confidence": confidence, "tool_logits": tool_logits}

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def score(
        self,
        query_emb: np.ndarray | torch.Tensor,
        evidence_embs: np.ndarray | torch.Tensor,
        features: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[float, list[float]]:
        """Compute ``(confidence, tool_probs)`` for a single query.

        Parameters
        ----------
        query_emb : array-like
            Shape ``(D,)`` or ``(1, D)``.
        evidence_embs : array-like
            Shape ``(K, D)``.  Up to the top-5 evidence embeddings.  They
            are mean-pooled across the K axis before being fed to the trunk.
            If empty, a zero vector is used.
        features : array-like or None
            Shape ``(F,)`` or ``(1, F)``.

        Returns
        -------
        (float, list[float])
            The calibrated confidence scalar in ``[0, 1]`` and the 4-vector
            of tool probabilities (softmax of tool_logits) whose order
            matches :data:`TOOL_NAMES`.
        """
        self.eval()
        device = next(self.parameters()).device

        q = _to_tensor(query_emb, dtype=torch.float32, device=device)
        if q.dim() == 1:
            q = q.unsqueeze(0)

        e = _to_tensor(evidence_embs, dtype=torch.float32, device=device)
        if e.numel() == 0:
            evidence_mean = torch.zeros(
                (1, self.embedding_dim), dtype=torch.float32, device=device
            )
        else:
            if e.dim() == 1:
                e = e.unsqueeze(0)
            evidence_mean = e.mean(dim=0, keepdim=True)

        feat_tensor: torch.Tensor | None = None
        if features is not None and self.extra_feature_dim > 0:
            ft = _to_tensor(features, dtype=torch.float32, device=device)
            if ft.dim() == 1:
                ft = ft.unsqueeze(0)
            feat_tensor = ft

        with torch.no_grad():
            out = self.forward(q, evidence_mean, feat_tensor)
            conf = float(out["confidence"].squeeze().cpu().item())
            probs = F.softmax(out["tool_logits"], dim=-1)
            probs_list = [float(p) for p in probs.squeeze(0).cpu().tolist()]

        return conf, probs_list

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def apply_calibration(self, temperature: float) -> None:
        """Set the temperature-scaling parameter.

        Parameters
        ----------
        temperature : float
            Positive scalar; typically fit by minimizing NLL on a held-out
            validation set.  ``temperature > 1`` softens overly confident
            predictions; ``< 1`` sharpens them.
        """
        if temperature <= 0:
            raise ValueError(
                f"Calibration temperature must be > 0, got {temperature}."
            )
        with torch.no_grad():
            self.temperature.fill_(float(temperature))
        self._calibrated = True
        logger.info("ConfidenceHead calibrated with T=%.4f", temperature)

    @property
    def is_calibrated(self) -> bool:
        """True if :meth:`apply_calibration` has been called."""
        return self._calibrated

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _config_dict(self) -> dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "extra_feature_dim": self.extra_feature_dim,
            "num_tools": self.num_tools,
            "calibrated": self._calibrated,
        }

    def save(self, path: str | Path) -> None:
        """Persist the state_dict and config as a single ``.pt`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "config": self._config_dict(),
        }
        torch.save(payload, path)
        logger.info("Saved ConfidenceHead to %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "ConfidenceHead":
        """Load a previously saved ConfidenceHead."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ConfidenceHead checkpoint not found: {path}")
        payload = torch.load(path, map_location=map_location or "cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise RuntimeError(
                f"Invalid ConfidenceHead checkpoint at {path}: missing state_dict."
            )
        cfg = payload.get("config", {})
        model = cls(
            embedding_dim=int(cfg.get("embedding_dim", 1024)),
            extra_feature_dim=int(cfg.get("extra_feature_dim", 0)),
            num_tools=int(cfg.get("num_tools", 4)),
        )
        model.load_state_dict(payload["state_dict"])
        model._calibrated = bool(cfg.get("calibrated", False))
        model.eval()
        logger.info("Loaded ConfidenceHead from %s", path)
        return model


def _to_tensor(
    x: np.ndarray | torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert an array-like to a contiguous torch tensor on the target device."""
    tensor: torch.Tensor
    if isinstance(x, torch.Tensor):
        tensor = x
    elif isinstance(x, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(x))
    else:
        return torch.tensor(x, dtype=dtype, device=device)
    return tensor.to(device=device, dtype=dtype)
