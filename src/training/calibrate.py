"""
Facade: ``run_calibration`` for ``python run.py calibrate``.

This module handles the post-hoc calibration of the AegisRAG confidence head.
It specifically implements Temperature Scaling to align model scores with 
actual empirical correctness, minimizing Expected Calibration Error (ECE).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

# Internal imports for metric calculation and scaling logic
from src.evaluation.calibration import compute_ece, temperature_scaling

logger = logging.getLogger(__name__)


def _default_preds_path(cfg: Any | None) -> Path:
    """
    Determines the path to the pre-computed model predictions.
    
    Args:
        cfg: Configuration object containing project directory paths.
        
    Returns:
        Path: Path object pointing to confidence_dev_preds.jsonl.
    """
    syn_dir = None
    if cfg is not None:
        syn_dir = getattr(getattr(cfg, "data", None), "synthetic_dir", None)
    if syn_dir:
        return Path(syn_dir) / "confidence_dev_preds.jsonl"
    return Path("data/synthetic/confidence_dev_preds.jsonl")


def _fallback_labels_path(cfg: Any | None) -> Path:
    """
    Determines the path to the gold labels to be used if predictions are missing.
    
    Args:
        cfg: Configuration object.
        
    Returns:
        Path: Path object pointing to confidence_labels.jsonl.
    """
    syn_dir = None
    if cfg is not None:
        syn_dir = getattr(getattr(cfg, "data", None), "synthetic_dir", None)
    if syn_dir:
        return Path(syn_dir) / "confidence_labels.jsonl"
    return Path("data/synthetic/confidence_labels.jsonl")


def _load_scores_and_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses a .jsonl file to extract prediction scores and ground truth labels.
    
    Args:
        path: The filesystem path to the .jsonl file.
        
    Returns:
        tuple: (scores_array, labels_array) as float64 numpy arrays.
    """
    scores: list[float] = []
    labels: list[float] = []
    
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            
            # Support multiple possible keys for flexibility across different pipeline stages
            score = obj.get("score", obj.get("predicted", obj.get("pred", obj.get("soft_label"))))
            label = obj.get("label", obj.get("gold", obj.get("soft_label")))
            
            if score is None or label is None:
                continue
            
            scores.append(float(score))
            labels.append(float(label))
            
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the Area Under the Receiver Operating Characteristic curve.
    Uses the Mann-Whitney U statistic approach for O(n log n) efficiency.
    
    Args:
        scores: Predicted probabilities.
        labels: Ground truth (soft or binary labels).
        
    Returns:
        float: The AUROC score, or NaN if calculation is impossible.
    """
    if scores.size == 0:
        return float("nan")
    
    # Binarize labels: 1 if sufficiency >= 0.5, else 0
    binary = (labels >= 0.5).astype(int)
    pos = scores[binary == 1]
    neg = scores[binary == 0]
    
    if pos.size == 0 or neg.size == 0:
        return float("nan")
        
    # Rank-based calculation of AUROC
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    
    rank_sum_pos = ranks[binary == 1].sum()
    n_pos = pos.size
    n_neg = neg.size
    
    # Calculate U statistic
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def run_calibration(model_tag: str, config: Any | None = None) -> dict[str, float]:
    """
    Executes the calibration pipeline: loading data, fitting temperature, 
    and reporting performance metrics (ECE and AUROC).
    
    Args:
        model_tag: Identifier for the model being calibrated (e.g., 'm5').
        config: Configuration dictionary or object.
        
    Returns:
        dict: Summary metrics including temperature, ECE before/after, and AUROC.
        
    Raises:
        FileNotFoundError: If no data files are found.
        RuntimeError: If the provided data file is empty.
    """
    # 1. Resolve Data Path
    preds_path = _default_preds_path(config)
    if not preds_path.exists():
        fallback = _fallback_labels_path(config)
        if fallback.exists():
            logger.info(
                "Precomputed predictions not found; using %s as both pred and label. "
                "Note: ECE before/after will match as this is a no-op.",
                fallback,
            )
            preds_path = fallback
        else:
            raise FileNotFoundError(
                f"No calibration data available: expected {preds_path} or {fallback}"
            )

    # 2. Load and Validate Data
    scores, labels = _load_scores_and_labels(preds_path)
    if scores.size == 0:
        raise RuntimeError(f"Calibration file {preds_path} is empty")

    # 3. Initial Metric Calculation
    # Binarize labels for ECE calculation
    binary = (labels >= 0.5).astype(int)
    ece_before = float(compute_ece(scores, binary))

    # 4. Temperature Scaling Logic
    # Temperature scaling operates on logits. We must inverse the sigmoid: 
    # logit = log(p / (1-p))
    eps = 1e-6
    clipped = np.clip(scores, eps, 1.0 - eps) # Avoid log(0) or division by zero
    logits = np.log(clipped / (1.0 - clipped))

    # Perform grid search or optimization to find the T that minimizes NLL
    temperature = float(temperature_scaling(logits, binary))

    # 5. Apply Scaling and Re-evaluate
    # Scaled probability = sigmoid(logits / T)
    scaled = 1.0 / (1.0 + np.exp(-(logits / max(temperature, eps))))
    ece_after = float(compute_ece(scaled, binary))

    auroc = _auroc(scores, labels)

    # 6. Logging and Results
    result = {
        "model_tag": model_tag,
        "n": int(scores.size),
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "auroc": auroc,
    }
    logger.info("Calibration result for %s: %s", model_tag, result)
    
    return result