"""Training orchestrator (SFT-free, decomposer-free).

Canonical order for the lean pipeline:

    retriever -> reranker -> dpo -> confidence -> alpha

SFT (``generator``) is removed: DPO runs directly on the base generator.
Decomposer training is removed: we use the rule-based splitter at runtime
(``src.decomposer.splitter._heuristic_split``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


_COMPONENT_ORDER: tuple[str, ...] = (
    "retriever",
    "reranker",
    "dpo",
    "confidence",
    "alpha",
)


def _load_trainer(component: str) -> Callable[..., dict[str, Any]]:
    if component == "retriever":
        from src.training.train_retriever import train
    elif component == "reranker":
        from src.training.train_reranker import train
    elif component == "dpo":
        from src.training.train_dpo import train
    elif component == "confidence":
        from src.training.train_confidence import train
    elif component == "alpha":
        from src.training.train_alpha import train
    elif component in {"generator", "sft"}:
        raise ValueError(
            "SFT ('generator') training has been removed; DPO runs on the base model."
        )
    elif component == "decomposer":
        raise ValueError(
            "Decomposer training has been removed; rule-based splitter is used at runtime."
        )
    else:
        raise ValueError(f"Unknown training component: {component}")
    return train


def run_training(
    component: str = "all",
    config: Any | None = None,
) -> dict[str, dict[str, Any]]:
    targets = _COMPONENT_ORDER if component == "all" else (component,)
    results: dict[str, dict[str, Any]] = {}
    for name in targets:
        logger.info("=== Training %s ===", name)
        try:
            trainer = _load_trainer(name)
            results[name] = trainer(config) or {"status": "ok"}
        except Exception as exc:
            logger.exception("Training %s failed", name)
            results[name] = {"status": "error", "error": str(exc)}
    return results
