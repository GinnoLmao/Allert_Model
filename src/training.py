"""ALLERT training utilities.

This module keeps the data-augmentation, walk-forward split, and curriculum
helpers that are framework-agnostic. The actual PyTorch training loop now lives
in ``src.pytorch_training``. Imports from ``src.training`` continue to work via
the re-exports below.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
def apply_jitter(X: np.ndarray, std: float, seed: Optional[int] = None) -> np.ndarray:
    """Add Gaussian noise to feature values."""
    if std <= 0:
        return X
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, std, size=X.shape)
    return X + noise


def apply_window_slice(
    X: np.ndarray, y: np.ndarray, prob: float, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """With probability ``prob``, slice each sequence to a contiguous sub-window."""
    if prob <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    n, seq_len, n_features = X.shape
    out_X = np.empty_like(X)
    out_y = y.copy()
    for i in range(n):
        if rng.random() < prob:
            start = rng.integers(0, seq_len // 4)
            end = rng.integers(3 * seq_len // 4, seq_len)
            out_X[i] = X[i].copy()
            out_X[i, :start] = 0.0
            out_X[i, end:] = 0.0
        else:
            out_X[i] = X[i]
    return out_X, out_y


def apply_mixup(
    X: np.ndarray, y: np.ndarray, alpha: float, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply mixup augmentation if alpha > 0."""
    if alpha <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    n = len(X)
    lam = rng.beta(alpha, alpha, size=n)
    lam = np.asarray(lam, dtype=X.dtype).reshape(-1, 1, 1)
    indices = rng.permutation(n)
    X_mixed = lam * X + (1 - lam) * X[indices]
    y_mixed = lam[:, :, 0] * y + (1 - lam[:, :, 0]) * y[indices]
    return X_mixed, y_mixed


def augment_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    aug_cfg: Dict[str, Any],
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply configured augmentation pipeline."""
    if not aug_cfg.get("enabled", False):
        return X_train, y_train
    X = apply_jitter(X_train, aug_cfg.get("jitter_std", 0.0), seed=seed)
    X, y = apply_window_slice(X, y_train, aug_cfg.get("window_slice_prob", 0.0), seed=seed)
    X, y = apply_mixup(X, y, aug_cfg.get("mixup_alpha", 0.0), seed=seed)
    return X, y


# ---------------------------------------------------------------------------
# Walk-forward CV splits
# ---------------------------------------------------------------------------
def walk_forward_splits(
    X: np.ndarray, cfg: Dict[str, Any]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate chronological walk-forward train/validation splits.

    Returns a list of (train_mask, val_mask) boolean arrays.
    """
    n_splits = cfg["training"]["walk_forward_cv"]["n_splits"]
    test_size_days = cfg["training"]["walk_forward_cv"]["test_size_days"]
    rph = 12  # 5-minute data
    test_size = test_size_days * rph * 24
    n = len(X)
    usable = n - test_size
    step = usable // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        val_end = i * step + test_size
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        train_mask[: i * step] = True
        val_mask[i * step : val_end] = True
        splits.append((train_mask, val_mask))
    return splits


# ---------------------------------------------------------------------------
# Curriculum learning
# ---------------------------------------------------------------------------
def build_curriculum_y(
    y: np.ndarray,
    cfg: Dict[str, Any],
    epoch: int,
) -> np.ndarray:
    """Mask out longer horizons during early curriculum epochs.

    Target layout: [pm2_5_6h, pm10_6h, pm2_5_12h, pm10_12h, pm2_5_24h, pm10_24h].
    """
    curriculum = cfg["training"].get("curriculum", {})
    if not curriculum.get("enabled", False):
        return y
    initial_h = curriculum.get("initial_horizon_hours", 6)
    final_h = curriculum.get("final_horizon_hours", 24)
    switch_epoch = curriculum.get("switch_epoch", 10)
    if epoch >= switch_epoch:
        return y

    frac = epoch / switch_epoch
    max_h = initial_h + frac * (final_h - initial_h)
    y_out = y.copy()
    if max_h < 12:
        y_out[:, 2:4] = np.nan
    if max_h < 24:
        y_out[:, 4:6] = np.nan
    return y_out


# ---------------------------------------------------------------------------
# Re-export PyTorch training loop (lazy to avoid circular imports)
# ---------------------------------------------------------------------------
import importlib as _importlib


def __getattr__(name: str):
    if name in {"train_model", "run_walk_forward_training", "save_checkpoint", "load_checkpoint", "get_checkpoint_dir"}:
        module = _importlib.import_module("src.pytorch_training")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "apply_jitter",
    "apply_window_slice",
    "apply_mixup",
    "augment_training_data",
    "walk_forward_splits",
    "build_curriculum_y",
    "train_model",
    "run_walk_forward_training",
    "save_checkpoint",
    "load_checkpoint",
    "get_checkpoint_dir",
]
