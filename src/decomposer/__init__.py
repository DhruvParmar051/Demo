"""
AegisRAG - Query Decomposer package.

Exports the three components used by the serving layer to detect, split,
and merge multi-part queries.
"""

from __future__ import annotations

from src.decomposer.classifier import DecompositionClassifier
from src.decomposer.merger import ResultMerger
from src.decomposer.splitter import QuerySplitter

__all__ = [
    "DecompositionClassifier",
    "QuerySplitter",
    "ResultMerger",
]
