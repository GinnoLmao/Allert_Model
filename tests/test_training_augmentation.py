"""Tests for training augmentation and walk-forward CV helpers."""
import numpy as np
import pandas as pd

from src.training import (
    apply_jitter,
    apply_window_slice,
    apply_mixup,
    walk_forward_splits,
)


def test_apply_jitter_adds_noise():
    X = np.ones((10, 5, 3))
    X_aug = apply_jitter(X, std=0.1, seed=42)
    assert X_aug.shape == X.shape
    assert not np.allclose(X_aug, X)


def test_apply_jitter_zero_std_returns_copy():
    X = np.ones((10, 5, 3))
    X_aug = apply_jitter(X, std=0.0, seed=42)
    np.testing.assert_array_equal(X_aug, X)


def test_apply_window_slice_preserves_shape():
    X = np.ones((10, 20, 3))
    y = np.ones((10, 6))
    X_aug, y_aug = apply_window_slice(X, y, prob=1.0, seed=42)
    assert X_aug.shape == X.shape
    assert y_aug.shape == y.shape


def test_apply_mixup_returns_weighted_average():
    X = np.arange(24).reshape(4, 2, 3).astype(float)
    y = np.arange(24).reshape(4, 6).astype(float)
    X_aug, y_aug = apply_mixup(X, y, alpha=0.2, seed=42)
    assert X_aug.shape == X.shape
    assert y_aug.shape == y.shape
    # Values should stay within the convex hull of original values
    assert X_aug.min() >= X.min() - 1e-6
    assert X_aug.max() <= X.max() + 1e-6


def test_walk_forward_splits_nontrivial():
    cfg = {
        "training": {
            "walk_forward_cv": {"n_splits": 3, "test_size_days": 7},
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
        }
    }
    n = 30000
    # Only the length of X is used by walk_forward_splits; keep memory tiny.
    X = np.zeros((n, 1, 1))
    splits = walk_forward_splits(X, cfg)
    assert len(splits) == 3
    for train_mask, val_mask in splits:
        assert train_mask.sum() > 0
        assert val_mask.sum() > 0
        assert not (train_mask & val_mask).any()
