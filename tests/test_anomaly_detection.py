"""Tests for the ALLERT anomaly detection module."""
import numpy as np

from src.config_loader import load_config
from src import anomaly_detection as ad


def test_velocity_override_fires_above_threshold():
    cfg = load_config()
    actual = np.array([10.0, 10.0, 30.0])  # jump of 20 ug/m3 in 5 min -> 240 ug/m3/h
    velocity, trigger = ad.velocity_override(
        actual, cfg["sampling"]["transmission_interval_minutes"], cfg["anomaly"]["velocity_threshold_ugm3_per_hour"]
    )
    assert trigger[-1]
    assert velocity[-1] > cfg["anomaly"]["velocity_threshold_ugm3_per_hour"]


def test_velocity_override_does_not_fire_below_threshold():
    cfg = load_config()
    actual = np.array([10.0, 10.0, 11.0])  # jump of 1 ug/m3 in 5 min -> 12 ug/m3/h
    velocity, trigger = ad.velocity_override(
        actual, cfg["sampling"]["transmission_interval_minutes"], cfg["anomaly"]["velocity_threshold_ugm3_per_hour"]
    )
    assert not trigger[-1]


def test_cusum_detects_sustained_drift():
    cfg = load_config()
    residuals = np.ones(100) * 0.1
    residuals[50:] = 5.0
    alerts, k, h = ad.cusum_detector(
        residuals,
        cfg["anomaly"]["cusum_k_multiplier"],
        cfg["anomaly"]["cusum_h_multiplier"],
    )
    assert alerts[50:].sum() > 0
