"""Tests for anomaly calibration helpers."""
import numpy as np

from src import anomaly_calibration as acal
from src.anomaly_detection import contingency_table


def test_get_z_thresholds_uses_per_pollutant_config():
    acfg = {
        "z_thresholds": [1.0, 2.0, 3.0],
        "z_thresholds_pm2_5": [2.0, 3.0, 4.0],
        "z_thresholds_pm10": [3.0, 4.0, 5.0],
    }
    assert acal.get_z_thresholds(acfg, "pm2_5") == [2.0, 3.0, 4.0]
    assert acal.get_z_thresholds(acfg, "pm10") == [3.0, 4.0, 5.0]


def test_get_velocity_threshold_uses_per_pollutant_config():
    acfg = {
        "velocity_threshold_ugm3_per_hour": 10.0,
        "velocity_threshold_pm2_5": 15.0,
        "velocity_threshold_pm10": 25.0,
    }
    assert acal.get_velocity_threshold(acfg, "pm2_5") == 15.0
    assert acal.get_velocity_threshold(acfg, "pm10") == 25.0


def test_grid_search_cusum_selects_high_f1_low_fpr():
    np.random.seed(42)
    # Create a baseline period, then a period where actual is above event threshold
    actual = np.ones(200) * 10.0
    forecast = np.ones(200) * 10.0
    actual[100:] = 25.0  # exceeds event_threshold of 15.0

    best, all_results = acal.grid_search_cusum(
        actual, forecast,
        k_multipliers=np.array([0.25, 0.5, 1.0]),
        h_multipliers=np.array([2.0, 4.0, 6.0]),
        max_fpr_pct=10.0,
        step_minutes=5,
        event_threshold=15.0,
        velocity_threshold=100.0,  # velocity should not trigger
    )
    assert best is not None
    assert "k_multiplier" in best
    assert "metrics" in best
    assert best["metrics"]["F1_Score"] > 0
    assert best["metrics"]["FalsePositiveRate_pct"] <= 10.0


def test_contingency_table_metrics():
    pred = np.array([True, True, False, False])
    truth = np.array([True, False, True, False])
    metrics = contingency_table(pred, truth)
    assert metrics["TP"] == 1
    assert metrics["FP"] == 1
    assert metrics["TN"] == 1
    assert metrics["FN"] == 1
