"""
Determinism utilities for reproducible experiments.

Provides seed-setting across all random number generators and
optional fully-deterministic mode for PyTorch.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For hash-based operations. NOTE: Python's hash randomization is
    # decided at interpreter startup, so setting this here only affects
    # *subsequently spawned* child processes (e.g. DataLoader workers).
    # For full determinism of the main process, export PYTHONHASHSEED in
    # the shell before invoking python.
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("Random seeds set to %d", seed)


def enable_deterministic_mode() -> None:
    """
    Enable fully deterministic behavior in PyTorch.

    This may reduce performance due to disabled optimizations. Some operations
    on CUDA do not have deterministic implementations and will raise errors;
    set CUBLAS_WORKSPACE_CONFIG to handle cuBLAS non-determinism.

    Should be called AFTER set_seed().
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Required for deterministic cuBLAS on CUDA >= 10.2
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    logger.info(
        "Deterministic mode enabled. "
        "torch.use_deterministic_algorithms=True, "
        "cudnn.deterministic=True, cudnn.benchmark=False"
    )


def setup_reproducibility(seed: int = 42, deterministic: bool = False) -> None:
    """
    Convenience wrapper: set seeds and optionally enable deterministic mode.

    Args:
        seed: The random seed.
        deterministic: If True, also enable fully deterministic algorithms.
    """
    set_seed(seed)
    if deterministic:
        enable_deterministic_mode()
