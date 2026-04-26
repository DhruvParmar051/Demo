"""
AegisRAG - Training Package.

Provides a ``TRAINERS`` mapping from component name to its ``train``
callable.  Imports are lazy so that loading this package does not pull
in heavy ML dependencies (torch, transformers, sentence-transformers)
until a specific trainer is actually requested.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


def _make_lazy(module_path: str) -> Callable[[Any], Dict[str, Any]]:
    """Return a wrapper that imports ``module_path.train`` on first call."""
    def _trainer(cfg: Any = None) -> Dict[str, Any]:
        import importlib
        mod = importlib.import_module(module_path)
        return mod.train(cfg)
    _trainer.__name__ = module_path.split(".")[-1]
    return _trainer


TRAINERS: Dict[str, Callable[[Any], Dict[str, Any]]] = {
    "retriever":  _make_lazy("src.training.train_retriever"),
    "reranker":   _make_lazy("src.training.train_reranker"),
    "generator":  _make_lazy("src.training.train_generator"),
    "dpo":        _make_lazy("src.training.train_dpo"),
    "confidence": _make_lazy("src.training.train_confidence"),
    "alpha":      _make_lazy("src.training.train_alpha"),
    "decomposer": _make_lazy("src.training.train_decomposer"),
}

__all__ = ["TRAINERS"]
