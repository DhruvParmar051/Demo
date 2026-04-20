"""
AegisRAG - Calibration utilities.

Implements Expected Calibration Error (ECE), reliability-diagram
plotting, and temperature scaling (grid-search variant) for the
confidence head.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (equal-width binning).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        confidences: 1-D array of predicted confidences in [0, 1].
        accuracies: 1-D array of binary correctness (0/1).
        n_bins: Number of equal-width bins spanning [0, 1].

    Returns:
        ECE value in [0, 1]. Returns 0.0 for an empty input.
    """
    confidences = np.asarray(confidences, dtype=float).ravel()
    accuracies = np.asarray(accuracies, dtype=float).ravel()
    if confidences.shape != accuracies.shape:
        raise ValueError(
            f"Shape mismatch: confidences={confidences.shape} "
            f"accuracies={accuracies.shape}"
        )
    n = confidences.size
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_acc = float(accuracies[mask].mean())
        bin_conf = float(confidences[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    save_path: Path,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
) -> None:
    """Save a reliability diagram (accuracy vs confidence per bin) as PNG.

    Args:
        confidences: 1-D array of predicted confidences in [0, 1].
        accuracies: 1-D array of binary correctness (0/1).
        save_path: Destination PNG path (parent dirs are created).
        n_bins: Number of equal-width bins.
        title: Plot title.

    Returns:
        None. Writes a PNG to ``save_path``.
    """
    import matplotlib  # lazy import

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confidences = np.asarray(confidences, dtype=float).ravel()
    accuracies = np.asarray(accuracies, dtype=float).ravel()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.any():
            bin_acc[b] = accuracies[mask].mean()
            bin_conf[b] = confidences[mask].mean()
            bin_count[b] = int(mask.sum())

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(
        bin_centers,
        bin_acc,
        width=bin_widths * 0.9,
        edgecolor="black",
        color="steelblue",
        alpha=0.8,
        label="Accuracy",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left")
    ece = compute_ece(confidences, accuracies, n_bins=n_bins)
    ax.text(
        0.02,
        0.95,
        f"ECE = {ece:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    t_min: float = 0.5,
    t_max: float = 3.0,
    n_steps: int = 51,
) -> float:
    """Grid-search temperature T that minimizes negative log-likelihood.

    Supports binary logits (1-D) or multi-class logits (2-D).

    Args:
        logits: For binary case, a 1-D array of raw logits. For multi-
            class, a 2-D array of shape (N, C).
        labels: For binary, 0/1 array of shape (N,). For multi-class,
            integer class indices of shape (N,).
        t_min: Minimum temperature (inclusive) for the grid.
        t_max: Maximum temperature (inclusive) for the grid.
        n_steps: Number of grid points sampled in [t_min, t_max].

    Returns:
        The temperature T (float) that minimizes NLL on ``(logits, labels)``.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels)
    if logits.size == 0:
        return 1.0

    temperatures = np.linspace(t_min, t_max, n_steps)
    best_t = 1.0
    best_nll = float("inf")

    if logits.ndim == 1:
        # Binary.
        labels_bin = labels.astype(float).ravel()
        for t in temperatures:
            scaled = logits / t
            # numerically-stable sigmoid log
            log_sigmoid = -np.logaddexp(0.0, -scaled)
            log_one_minus = -np.logaddexp(0.0, scaled)
            nll = -float(
                np.mean(labels_bin * log_sigmoid + (1.0 - labels_bin) * log_one_minus)
            )
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)
    elif logits.ndim == 2:
        labels_idx = labels.astype(int).ravel()
        n = logits.shape[0]
        rows = np.arange(n)
        for t in temperatures:
            scaled = logits / t
            # log_softmax
            m = scaled.max(axis=1, keepdims=True)
            log_probs = scaled - m - np.log(np.exp(scaled - m).sum(axis=1, keepdims=True))
            nll = -float(log_probs[rows, labels_idx].mean())
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)
    else:
        raise ValueError(f"Unsupported logits shape: {logits.shape}")

    return best_t
