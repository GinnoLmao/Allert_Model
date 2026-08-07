"""Tests for evaluation helpers: persistence baseline and error-by-regime."""
import numpy as np
import pandas as pd

from src.evaluation import persistence_baseline, compute_error_by_regime


def test_persistence_baseline_shape_and_metrics():
    rng = np.random.default_rng(42)
    n_train = 100
    n_test = 50
    y_train = rng.uniform(10, 50, size=(n_train, 6))
    y_test = rng.uniform(10, 50, size=(n_test, 6))
    y_pred, metrics = persistence_baseline(y_train, y_test)
    assert y_pred.shape == y_test.shape
    assert "PM2_5" in metrics
    assert "PM10" in metrics


def test_compute_error_by_regime_keys():
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.uniform(10, 80, size=(n, 6))
    y_pred = y_true + rng.normal(0, 2, size=(n, 6))
    timestamps = pd.date_range("2026-01-01", periods=n, freq="5min").to_numpy()
    regimes = compute_error_by_regime(y_true, y_pred, timestamps)
    assert "hour_of_day" in regimes
    assert "day_of_week" in regimes
    assert "pm_level" in regimes
    for key in regimes:
        for val, stats in regimes[key].items():
            assert "n" in stats
            assert "mean_mae" in stats
