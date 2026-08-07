"""Tests for the ALLERT feature engineering module."""
import numpy as np
import pandas as pd

from src.config_loader import load_config
from src import feature_engineering as fe


def _build_df(n=1000):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "pm2_5": np.linspace(10, 50, n),
            "pm10": np.linspace(20, 80, n),
            "temperature": 28.0,
            "humidity": 75.0,
        },
        index=idx,
    )


def test_feature_count():
    cfg = load_config()
    df = _build_df()
    features = fe.build_feature_matrix(df, cfg)
    assert len(features.columns) == 177


def test_lag_72h_offset():
    cfg = load_config()
    df = _build_df(n=1000)
    features = fe.build_feature_matrix(df, cfg)
    steps = 72 * 12
    valid = features["pm2_5_lag_72h"].dropna()
    expected = df["pm2_5"].shift(steps).loc[valid.index]
    assert (valid - expected).abs().max() < 1e-9


def test_cyclical_continuity_across_midnight():
    cfg = load_config()
    idx = pd.date_range("2026-01-01 23:55:00", periods=3, freq="5min")
    df = pd.DataFrame({"pm2_5": [1, 2, 3], "pm10": [2, 3, 4], "temperature": 25, "humidity": 60}, index=idx)
    cyc = fe.add_cyclical_time_features(df, cfg)
    # hour 23 and hour 0 should be close in cosine space
    cos_last = cyc.loc[idx[0], "hour_cos"]
    cos_first = cyc.loc[idx[1], "hour_cos"]
    assert abs(cos_last - cos_first) < 0.1
