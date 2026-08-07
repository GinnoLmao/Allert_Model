"""Tests for the ALLERT configuration loader."""
import copy

import pytest
import yaml

from src.config_loader import load_config, _validate_config


def test_load_config_ok():
    cfg = load_config()
    assert cfg["model"]["sequence_length"] == 2016
    assert abs(sum(cfg["model"]["loss_weights"]) - 1.0) < 1e-6


def _make_bad_config(cfg, path, value):
    bad = copy.deepcopy(cfg)
    keys = path.split(".")
    node = bad
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    return bad


def test_bad_splits_rejected():
    cfg = load_config()
    bad = _make_bad_config(cfg, "training.val_split", 0.25)
    with pytest.raises(ValueError, match="splits must sum"):
        _validate_config(bad)


def test_bad_loss_weights_rejected():
    cfg = load_config()
    bad = _make_bad_config(cfg, "model.loss_weights", [0.5, 0.3, 0.3])
    with pytest.raises(ValueError, match="Loss weights must sum"):
        _validate_config(bad)


def test_bad_lag_weights_rejected():
    cfg = load_config()
    bad = _make_bad_config(cfg, "risk.distributed_lag_weights", [0.5, 0.25, 0.1, 0.05])
    with pytest.raises(ValueError, match="Distributed lag weights must sum"):
        _validate_config(bad)


def test_horizon_loss_weight_mismatch_rejected():
    cfg = load_config()
    bad = _make_bad_config(cfg, "model.loss_weights", [0.5, 0.5])
    with pytest.raises(ValueError, match="must match forecast horizons"):
        _validate_config(bad)
