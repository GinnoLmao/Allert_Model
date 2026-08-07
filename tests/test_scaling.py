"""Tests for scaling and chronological split utilities."""
import numpy as np
import pandas as pd

from src.config_loader import load_config
from src import scaling


def _make_df(n=100):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {"feat1": np.arange(n, dtype=float), "feat2": np.arange(n, dtype=float) * 2},
        index=idx,
    )


def test_chronological_split_preserves_order():
    cfg = load_config()
    df = _make_df(100)
    train, val, test = scaling.chronological_split(df, cfg)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_scaler_fit_train_only():
    cfg = load_config()
    df = _make_df(100)
    train, val, test = scaling.chronological_split(df, cfg)
    scaler = scaling.FeatureScaler("minmax")
    scaler.fit(train, ["feat1", "feat2"])

    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    # MinMax scaling maps training data to [0, 1]
    assert train_scaled["feat1"].min() == 0.0
    assert train_scaled["feat1"].max() == 1.0
    # Test data must not be scaled using its own min/max
    assert not (test_scaled["feat1"].min() == 0.0 and test_scaled["feat1"].max() == 1.0)
