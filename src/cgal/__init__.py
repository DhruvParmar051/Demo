"""
AegisRAG - Confidence-Gated Action Loop (CGAL) package.

Exposes the three top-level primitives used by the serving layer:

* :class:`ConfidenceHead` -- soft-label confidence + tool-policy model.
* :class:`AlphaNetwork`   -- adaptive dense/sparse fusion weight predictor.
* :class:`CGALLoopEngine` -- the bounded retry/escalation orchestrator.
"""

from __future__ import annotations

from src.cgal.alpha_network import AlphaNetwork
from src.cgal.confidence_head import ConfidenceHead
from src.cgal.loop_engine import CGALLoopEngine

__all__ = [
    "AlphaNetwork",
    "ConfidenceHead",
    "CGALLoopEngine",
]
