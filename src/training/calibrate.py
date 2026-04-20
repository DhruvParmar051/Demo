"""Facade: ``run_calibration`` for ``python run.py calibrate``.

Loads confidence-head predictions against gold sufficiency labels from
the dev split, fits a temperature by NLL grid-search via
:func:`src.evaluation.calibration.temperature_scaling`, and reports
ECE before / after plus AUROC.

If a pre-computed predictions file is available at
``data/synthetic/confidence_dev_preds.jsonl`` (one
``{"score": ..., "label": ...}`` per line), it is used directly.
Otherwise the dev split of ``confidence_labels.jsonl`` is treated as
both predictions and labels so the method is still exercised end-to-end
(a no-op in that case).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.calibration import compute_ece, temperature_scaling

logger = logging.getLogger(__name__)


def _default_preds_path(cfg: Any | None) -> Path:
    syn_dir = None
    if cfg is not None:
        syn_dir = getattr(getattr(cfg, "data", None), "synthetic_dir", None)
    if syn_dir:
        return Path(syn_dir) / "confidence_dev_preds.jsonl"
    return Path("data/synthetic/confidence_dev_preds.jsonl")


def _fallback_labels_path(cfg: Any | None) -> Path:
    syn_dir = None
    if cfg is not None:
        syn_dir = getattr(getattr(cfg, "data", None), "synthetic_dir", None)
    if syn_dir:
        return Path(syn_dir) / "confidence_labels.jsonl"
    return Path("data/synthetic/confidence_labels.jsonl")


def _load_scores_and_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[float] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            score = obj.get("score", obj.get("predicted", obj.get("pred")))
            label = obj.get("label", obj.get("gold", obj.get("soft_label")))
            if score is None or label is None:
                continue
            scores.append(float(score))
            labels.append(float(label))
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U AUROC against a 0.5 threshold on soft labels."""
    if scores.size == 0:
        return float("nan")
    binary = (labels >= 0.5).astype(int)
    pos = scores[binary == 1]
    neg = scores[binary == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Rank-based AUROC -- O(n log n).
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    rank_sum_pos = ranks[binary == 1].sum()
    n_pos = pos.size
    n_neg = neg.size
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def run_calibration(model_tag: str, config: Any | None = None) -> dict[str, float]:
    """Calibrate the confidence head via temperature scaling on dev set."""
    preds_path = _default_preds_path(config)
    if not preds_path.exists():
        fallback = _fallback_labels_path(config)
        if fallback.exists():
            logger.info(
                "Precomputed predictions not found; using %s as both pred and label "
                "(ECE before/after will match)",
                fallback,
            )
            preds_path = fallback
        else:
            raise FileNotFoundError(
                f"No calibration data available: expected {preds_path} or {fallback}"
            )

    scores, labels = _load_scores_and_labels(preds_path)
    if scores.size == 0:
        raise RuntimeError(f"Calibration file {preds_path} is empty")

    # Treat each soft label as probabilistic; binarize at 0.5 for ECE.
    binary = (labels >= 0.5).astype(int)
    ece_before = float(compute_ece(scores, binary))

    # Convert sigmoid probs -> logits for temperature fitting.
    eps = 1e-6
    clipped = np.clip(scores, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped))

    temperature = float(temperature_scaling(logits, binary))

    scaled = 1.0 / (1.0 + np.exp(-(logits / max(temperature, eps))))
    ece_after = float(compute_ece(scaled, binary))

    auroc = _auroc(scores, labels)

    result = {
        "model_tag": model_tag,
        "n": int(scores.size),
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "auroc": auroc,
    }
    logger.info("Calibration result: %s", result)
    return result
