"""Tests for enhanced HPO configuration."""
import pytest

from src.config_loader import load_config, _validate_config


def test_hpo_sampler_pruner_validation():
    cfg = load_config()
    # Default should pass
    _validate_config(cfg)


def test_hpo_sampler_rejected():
    cfg = load_config()
    cfg["hpo"]["sampler"] = "invalid"
    with pytest.raises(ValueError, match="hpo.sampler must be one of"):
        _validate_config(cfg)


def test_hpo_pruner_rejected():
    cfg = load_config()
    cfg["hpo"]["pruner"] = "invalid"
    with pytest.raises(ValueError, match="hpo.pruner must be one of"):
        _validate_config(cfg)


def test_hpo_storage_sqlite_allowed():
    cfg = load_config()
    assert cfg["hpo"].get("storage") is not None
    assert "sqlite" in cfg["hpo"].get("storage", "")
    _validate_config(cfg)


def test_hpo_storage_null_allowed():
    cfg = load_config()
    cfg["hpo"]["storage"] = None
    _validate_config(cfg)
