"""Tests for the ALLERT PyTorch Bi-LSTM model builder."""
import numpy as np
import torch

from src.config_loader import load_config
from src import bilstm_model as bm
from src.pytorch_model import DEVICE


def _cfg_without_spike_head():
    """Return a config copy with the spike head disabled for shape tests."""
    cfg = load_config()
    cfg["model"]["use_spike_head"] = False
    return cfg


def test_model_output_shape():
    cfg = _cfg_without_spike_head()
    bm.set_random_seeds(cfg)
    model = bm.build_bilstm((2016, 56), cfg)
    model.to(DEVICE)
    model.eval()
    X_dummy = torch.zeros((2, 2016, 56), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        y = model(X_dummy)
    assert y.shape == (2, 6)


def test_model_total_params_order():
    cfg = _cfg_without_spike_head()
    bm.set_random_seeds(cfg)
    model = bm.build_bilstm((2016, 56), cfg)
    model.to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 2-layer 96/128 Bi-LSTM with 56-feature input is ~355K; with spike head ~467K.
    assert 300_000 < params < 500_000


def test_predict_shape():
    cfg = _cfg_without_spike_head()
    bm.set_random_seeds(cfg)
    model = bm.build_bilstm((2016, 56), cfg)
    model.to(DEVICE)
    model.eval()
    X_dummy = torch.zeros((2, 2016, 56), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        y = model(X_dummy)
    assert y.shape == (2, 6)


def test_spike_head_returns_tuple():
    """When the spike head is enabled, the model returns (regression, spike_logits)."""
    cfg = load_config()
    cfg["model"]["use_spike_head"] = True
    bm.set_random_seeds(cfg)
    model = bm.build_bilstm((2016, 56), cfg)
    model.to(DEVICE)
    model.eval()
    X_dummy = torch.zeros((2, 2016, 56), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        out = model(X_dummy)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (2, 6)
    assert out[1].shape == (2, 3)


def test_composite_loss():
    from src.pytorch_model import composite_loss
    y_pred = torch.zeros((4, 6))
    y_true = torch.zeros((4, 6))
    loss = composite_loss(y_pred, y_true)
    assert loss.item() == 0.0
