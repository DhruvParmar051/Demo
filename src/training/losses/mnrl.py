"""
Multiple Negatives Ranking Loss (MNRL).

In-batch negatives with optional hard negatives. For a batch of ``N``
query/positive pairs, the similarity matrix has the positive at the
diagonal; cross-entropy over rows yields the contrastive loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultipleNegativesRankingLoss(nn.Module):
    """Scaled cosine similarity with in-batch (and optional hard) negatives.

    Parameters
    ----------
    scale : float
        Temperature multiplier applied to similarities before softmax.
    similarity : {"cos", "dot"}
        Similarity metric.
    """

    def __init__(self, scale: float = 20.0, similarity: str = "cos") -> None:
        super().__init__()
        self.scale = float(scale)
        self.similarity = similarity

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        hard_negatives_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the MNR loss.

        Shapes
        ------
        query_emb : (B, D)
        positive_emb : (B, D)
        hard_negatives_emb : (B, K, D) or None
        """
        if self.similarity == "cos":
            query = F.normalize(query_emb, dim=-1)
            pos = F.normalize(positive_emb, dim=-1)
        else:
            query, pos = query_emb, positive_emb

        # (B, B) -- each row against the batch's positives (in-batch negatives).
        sims = query @ pos.T

        if hard_negatives_emb is not None:
            hn = hard_negatives_emb
            if self.similarity == "cos":
                hn = F.normalize(hn, dim=-1)
            # (B, K) hard-negative scores per query.
            hn_sims = torch.einsum("bd,bkd->bk", query, hn)
            sims = torch.cat([sims, hn_sims], dim=1)  # (B, B + K)

        sims = sims * self.scale
        targets = torch.arange(sims.size(0), device=sims.device)
        return F.cross_entropy(sims, targets)
