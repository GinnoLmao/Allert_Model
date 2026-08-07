"""Tests for the ALLERT preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.config_loader import load_config
from src import preprocessing as pp


def _synthetic_df(n=100):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    df = pd.DataFrame(
        {
            "pm2_5": np.linspace(10, 30, n),
            "pm10": np.linspace(20, 50, n),
            "temperature": np.linspace(25, 30, n),
            "humidity": np.linspace(60, 80, n),
        },
        index=idx,
    )
    return df


def test_load_raw_data_normalizes_columns():
    cfg = load_config()
    df = pp.load_raw_data(cfg)
    assert "temperature" in df.columns
    assert "humidity" in df.columns
    assert "temperature_c" not in df.columns
    assert "humidity_percent" not in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)


def test_validate_ranges_flags_out_of_range():
    cfg = load_config()
    df = _synthetic_df()
    df.loc[df.index[5], "pm2_5"] = 2000
    df = pp.validate_ranges(df, cfg)
    assert df.loc[df.index[5], "quality_flag"] == "OUT_OF_RANGE"
    assert (df["quality_flag"] != "OK").sum() == 1


def test_detect_outliers_flags_sensor_error():
    cfg = load_config()
    df = _synthetic_df(n=500)
    # Inject a clear outlier after enough history for rolling stats
    df.loc[df.index[300], "pm2_5"] = 500
    df = pp.validate_ranges(df, cfg)
    df = pp.detect_outliers(df, cfg)
    assert df.loc[df.index[300], "quality_flag"] == "SENSOR_ERROR"


def test_classify_and_fill_gaps_flags_long_gap():
    cfg = load_config()
    df = _synthetic_df(n=50)
    # Drop 10 rows -> 50-minute gap (> 2h threshold? no, 50 min < 2h, so interpolated)
    df_with_gap = df.drop(df.index[20:30])
    df_filled = pp.classify_and_fill_gaps(df_with_gap, cfg)
    assert "imputation_flag" in df_filled.columns
    assert (df_filled["imputation_flag"] == "ORIGINAL").sum() == 40
    assert (df_filled["imputation_flag"] == "INTERPOLATED").sum() == 10


def test_dataset_statistics_schema():
    cfg = load_config()
    df, stats = pp.preprocess(cfg)
    required = [
        "collection_duration_days",
        "total_raw_5min_records",
        "availability_rate_pct",
        "gap_summary",
        "quality_flags",
        "imputation_flags",
        "per_variable_availability",
    ]
    for key in required:
        assert key in stats
