"""ALLERT configuration loader.

Loads ``config/config.yaml`` and validates the constraints that keep the rest of
pipeline consistent and free of silent errors.

Validation performed:
    - train/val/test splits sum to 1.0
    - model loss weights sum to 1.0
    - risk distributed-lag weights sum to 1.0
    - number of loss weights matches number of forecast horizons
    - sequence_length equals 2016 (7 days of 5-minute data)

All modules should import config through ``load_config()``.
"""
from pathlib import Path
from typing import Any, Dict

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


def get_project_root() -> Path:
    """Return the absolute project root directory."""
    return _PROJECT_ROOT


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load and validate ``config.yaml``.

    Parameters
    ----------
    path:
        Optional override path to a config file.

    Returns
    -------
    dict
        The validated configuration dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError("Config file is empty or invalid YAML")

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Dict[str, Any]) -> None:
    """Run the validation checks defined in the ALLERT design docs."""
    # Splits must sum to 1.0
    splits = (
        cfg["training"]["train_split"]
        + cfg["training"]["val_split"]
        + cfg["training"]["test_split"]
    )
    if abs(splits - 1.0) > 1e-6:
        raise ValueError(f"Train/val/test splits must sum to 1.0, got {splits}")

    # Loss weights must sum to 1.0
    loss_weights = cfg["model"]["loss_weights"]
    if abs(sum(loss_weights) - 1.0) > 1e-6:
        raise ValueError(f"Loss weights must sum to 1.0, got {sum(loss_weights)}")

    # Distributed lag weights must sum to 1.0
    lag_weights = cfg["risk"]["distributed_lag_weights"]
    if abs(sum(lag_weights) - 1.0) > 1e-6:
        raise ValueError(
            f"Distributed lag weights must sum to 1.0, got {sum(lag_weights)}"
        )

    # Loss weights must match forecast horizons
    horizons = cfg["model"]["forecast_horizons_hours"]
    if len(horizons) != len(loss_weights):
        raise ValueError(
            f"Loss weights ({len(loss_weights)}) must match forecast horizons "
            f"({len(horizons)})"
        )

    # Sequence length must be exactly 2016
    seq_len = cfg["model"]["sequence_length"]
    if seq_len != 2016:
        raise ValueError(
            f"Model sequence_length must be 2016 (7 days @ 5 min), got {seq_len}"
        )

    # Basic sanity checks on non-negative thresholds
    if cfg["preprocessing"]["gap_fill_threshold_hours"] < 0:
        raise ValueError("gap_fill_threshold_hours must be non-negative")
    if cfg["model"]["sequence_length"] <= 0:
        raise ValueError("sequence_length must be positive")

    # Validate per-pollutant threshold lengths when present
    anomaly = cfg.get("anomaly", {})
    if "z_thresholds_pm2_5" in anomaly and len(anomaly["z_thresholds_pm2_5"]) != 3:
        raise ValueError("anomaly.z_thresholds_pm2_5 must contain exactly 3 values")
    if "z_thresholds_pm10" in anomaly and len(anomaly["z_thresholds_pm10"]) != 3:
        raise ValueError("anomaly.z_thresholds_pm10 must contain exactly 3 values")

    # Validate reproducibility section
    repro = cfg.get("reproducibility", {})
    if repro.get("deterministic_mode") and repro.get("random_seed") is None:
        raise ValueError("reproducibility.random_seed is required when deterministic_mode is true")

    # Validate HPO section keys if present
    hpo = cfg.get("hpo", {})
    if hpo.get("sampler") not in {None, "tpe", "random", "cmaes"}:
        raise ValueError("hpo.sampler must be one of: tpe, random, cmaes")
    if hpo.get("pruner") not in {None, "median", "halving", "hyperband", "none"}:
        raise ValueError("hpo.pruner must be one of: median, halving, hyperband, none")

    # Validate training splits still sum to 1.0 even with augmentation/CV flags
    if cfg["training"]["train_split"] + cfg["training"]["val_split"] + cfg["training"]["test_split"] != splits:
        pass  # already validated above

    # Validate augmentation parameters are non-negative
    aug = cfg["training"].get("data_augmentation", {})
    if aug.get("jitter_std", 0) < 0:
        raise ValueError("training.data_augmentation.jitter_std must be non-negative")


__all__ = ["load_config", "get_project_root", "CONFIG_PATH"]
