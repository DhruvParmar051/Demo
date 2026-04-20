"""Tests for AlphaNetwork feature extraction and inference."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def test_extract_features_produces_fixed_dim():
    import torch

    from src.cgal.alpha_network import AlphaNetwork

    net = AlphaNetwork()
    emb = np.ones(1024, dtype=np.float32) * 0.1
    feats = net.extract_features("what is a refund?", emb, "billing")
    assert isinstance(feats, torch.Tensor)
    assert feats.dim() == 1
    assert feats.shape[0] >= 4


def test_predict_alpha_in_unit_interval():
    from src.cgal.alpha_network import AlphaNetwork

    net = AlphaNetwork()
    emb = np.ones(1024, dtype=np.float32) * 0.1
    alpha = net.predict_alpha("a simple query", emb, "billing")
    assert 0.0 <= alpha <= 1.0
