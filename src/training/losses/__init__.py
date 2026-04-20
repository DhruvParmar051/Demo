"""
AegisRAG - Custom Loss Functions.

Re-exports:

* :class:`CitationWeightedCELoss` -- citation-aware cross-entropy used during
  generator SFT to up-weight tokens inside ``[doc_id:start-end]`` markers.
* :class:`MultipleNegativesRankingLoss` -- contrastive loss for bi-encoder
  retriever fine-tuning with in-batch and hard negatives.
"""

from __future__ import annotations

from src.training.losses.citation_weighted_ce import CitationWeightedCELoss
from src.training.losses.mnrl import MultipleNegativesRankingLoss

__all__ = [
    "CitationWeightedCELoss",
    "MultipleNegativesRankingLoss",
]
