"""Tests for target scaling helpers."""
import numpy as np
import pandas as pd

from src.target_scaling import TargetScaler, TARGET_COLUMNS


def _make_df(n=100):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "pm2_5_target": np.linspace(10, 50, n),
            "pm10_target": np.linspace(20, 100, n),
        },
        index=idx,
    )


def test_target_scaler_identity_when_not_scaled():
    df = _make_df()
    scaler = TargetScaler("minmax")
    scaler.fit(df, TARGET_COLUMNS)
    scaler.scaled = False
    arr = df[TARGET_COLUMNS].to_numpy()
    inv = scaler.inverse_transform(arr)
    np.testing.assert_allclose(inv, arr)


def test_target_scaler_inverts_scaled_values():
    df = _make_df()
    scaler = TargetScaler("minmax")
    scaler.fit(df, TARGET_COLUMNS)
    scaler.scaled = True
    arr = df[TARGET_COLUMNS].to_numpy()
    transformed = scaler.transform(arr)
    inv = scaler.inverse_transform(transformed)
    np.testing.assert_allclose(inv, arr, rtol=1e-4)


def test_target_scaler_preserves_target_column_names():
    df = _make_df()
    scaler = TargetScaler("minmax")
    scaler.fit(df)
    assert scaler.target_cols == TARGET_COLUMNS
