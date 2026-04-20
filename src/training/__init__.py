"""
AegisRAG - Training Package.

Re-exports the ``train`` entrypoint of every trainer module so callers
(notably ``run.py``) can dispatch with a simple mapping:

    from src.training import TRAINERS
    TRAINERS["generator"](cfg)

Each trainer exposes ``train(cfg) -> dict[str, Any]`` and a
``main()`` CLI entrypoint.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from src.training.train_alpha import train as train_alpha
from src.training.train_confidence import train as train_confidence
from src.training.train_decomposer import train as train_decomposer
from src.training.train_dpo import train as train_dpo
from src.training.train_generator import train as train_generator
from src.training.train_reranker import train as train_reranker
from src.training.train_retriever import train as train_retriever

TRAINERS: Dict[str, Callable[[Any], Dict[str, Any]]] = {
    "retriever": train_retriever,
    "reranker": train_reranker,
    "generator": train_generator,
    "dpo": train_dpo,
    "confidence": train_confidence,
    "alpha": train_alpha,
    "decomposer": train_decomposer,
}

__all__ = [
    "TRAINERS",
    "train_retriever",
    "train_reranker",
    "train_generator",
    "train_dpo",
    "train_confidence",
    "train_alpha",
    "train_decomposer",
]
