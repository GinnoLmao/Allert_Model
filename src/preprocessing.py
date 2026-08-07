"""ALLERT data preprocessing pipeline.

This module cleans the raw 5-minute sensor CSV and produces an analysis-ready
DataFrame while preserving data-provenance flags. Flagged rows are **kept** in
the output so that sequence generation can exclude contaminated windows.

Pipeline steps:
    1. Load raw CSV and normalize column names.
    2. Parse / sort / deduplicate timestamps.
    3. Validate physical ranges and flag ``OUT_OF_RANGE``.
    4. Validate ``PM10 >= PM2.5`` and flag ``SENSOR_ERROR``.
    5. Rolling Z-score outlier detection and flag ``SENSOR_ERROR``.
    6. Reindex to regular 5-minute grid, classify gaps, interpolate short gaps.
    7. Rolling median filter for noise reduction.
    8. Export dataset statistics (Table 4.7).

Provenance flags:
    - quality_flag: OK | SENSOR_ERROR | OUT_OF_RANGE
    - imputation_flag: ORIGINAL | INTERPOLATED | GAP

Thesis traceability:
    - Chapter 3.2, "Data Infrastructure and Preprocessing"
    - Chapter 4, Table 4.7 (Data Quality & Preprocessing Metrics)
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config_loader import get_project_root


# Map from the raw CSV column names to the canonical names used everywhere else.
RAW_COLUMN_MAP = {
    "temperature_c": "temperature",
    "humidity_percent": "humidity",
}

SENSOR_COLUMNS = ["pm2_5", "pm10", "temperature", "humidity"]


# ---------------------------------------------------------------------------
# Loading and basic cleaning
# ---------------------------------------------------------------------------
def load_raw_data(cfg: Dict[str, Any], subset_days: Optional[int] = None) -> pd.DataFrame:
    """Load the raw CSV, normalize columns, parse timestamps and sort.

    Parameters
    ----------
    subset_days:
        If provided, keep only the most recent ``subset_days`` of data.
        Useful for fast correctness runs on a CPU.
    """
    raw_path = Path(cfg["paths"]["data_raw"])
    if not raw_path.is_absolute():
        raw_path = get_project_root() / raw_path

    df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    df = df.rename(columns=RAW_COLUMN_MAP)

    if "timestamp" not in df.columns:
        raise ValueError("Raw data must contain a 'timestamp' column")

    # Ensure canonical sensor columns exist
    for col in SENSOR_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in raw data")

    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if subset_days is not None:
        records_per_day = int(cfg["sampling"]["records_per_day"])
        n_keep = subset_days * records_per_day
        if len(df) > n_keep:
            df = df.iloc[-n_keep:]

    # Reject future timestamps
    now = pd.Timestamp.now(tz=df.index.tz)
    if (df.index > now).any():
        raise ValueError("Future timestamps detected in raw data")

    return df


# ---------------------------------------------------------------------------
# Range validation and sensor-consistency checks
# ---------------------------------------------------------------------------
def validate_ranges(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Flag values outside the physical ranges defined in config."""
    df = df.copy()
    if "quality_flag" not in df.columns:
        df["quality_flag"] = "OK"

    limits = cfg["preprocessing"]["range_limits"]
    for col, (lo, hi) in limits.items():
        if col not in df.columns:
            continue
        out_of_range = (df[col] < lo) | (df[col] > hi)
        # Do not overwrite an existing non-OK flag
        mask = out_of_range & (df["quality_flag"] == "OK")
        df.loc[mask, "quality_flag"] = "OUT_OF_RANGE"

    if cfg["preprocessing"].get("validate_pm10_geq_pm25", True):
        inconsistent = df["pm10"] < df["pm2_5"]
        mask = inconsistent & (df["quality_flag"] == "OK")
        df.loc[mask, "quality_flag"] = "SENSOR_ERROR"

    return df


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------
def detect_outliers(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Flag statistical outliers using a rolling Z-score over a 24-hour window."""
    df = df.copy()
    if "quality_flag" not in df.columns:
        df["quality_flag"] = "OK"

    records_per_day = int(cfg["sampling"]["records_per_day"])
    window_steps = cfg["preprocessing"]["outlier_zscore_window_hours"] * records_per_day // 24
    z_thresh = cfg["preprocessing"]["outlier_zscore_threshold"]

    for col in ["pm2_5", "pm10"]:
        if col not in df.columns:
            continue
        rolling_mean = (
            df[col]
            .rolling(window=window_steps, min_periods=1)
            .mean()
        )
        rolling_std = (
            df[col]
            .rolling(window=window_steps, min_periods=1)
            .std()
            .replace(0, np.nan)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            z_score = (df[col] - rolling_mean) / rolling_std

        outlier = z_score.abs() > z_thresh
        mask = outlier & (df["quality_flag"] == "OK")
        df.loc[mask, "quality_flag"] = "SENSOR_ERROR"

    return df


# ---------------------------------------------------------------------------
# Gap classification and interpolation
# ---------------------------------------------------------------------------
def _infer_frequency(df: pd.DataFrame) -> pd.Timedelta:
    """Infer the dominant sampling interval from the index."""
    median_delta = df.index.to_series().diff().median()
    if pd.isna(median_delta) or median_delta == pd.Timedelta(0):
        raise ValueError("Cannot infer sampling frequency from index")
    return median_delta


def classify_and_fill_gaps(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Reindex to a regular grid, interpolate short gaps, and flag long gaps."""
    df = df.copy()
    if "imputation_flag" not in df.columns:
        df["imputation_flag"] = "ORIGINAL"

    median_delta = _infer_frequency(df)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=median_delta)

    original_index = set(df.index)
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    # Mark provenance for every row
    df["imputation_flag"] = "GAP"
    df.loc[df.index.isin(original_index), "imputation_flag"] = "ORIGINAL"

    # Ensure quality_flag exists for inserted rows
    if "quality_flag" not in df.columns:
        df["quality_flag"] = "OK"
    df["quality_flag"] = df["quality_flag"].fillna("OK")

    threshold_hours = cfg["preprocessing"]["gap_fill_threshold_hours"]
    threshold_steps = max(1, int(pd.Timedelta(hours=threshold_hours) / median_delta))

    numeric_cols = [c for c in SENSOR_COLUMNS if c in df.columns]
    for col in numeric_cols:
        df[col] = df[col].interpolate(
            method="time", limit=threshold_steps, limit_direction="both"
        )

    # Newly-filled rows become INTERPOLATED; remaining gaps stay GAP
    interpolated = (df["imputation_flag"] == "GAP") & df[numeric_cols].notna().all(axis=1)
    df.loc[interpolated, "imputation_flag"] = "INTERPOLATED"

    return df


# ---------------------------------------------------------------------------
# Rolling median filter
# ---------------------------------------------------------------------------
def apply_rolling_median(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Apply a centered rolling median filter to the sensor channels."""
    df = df.copy()
    window = cfg["preprocessing"]["rolling_median_window"]
    numeric_cols = [c for c in SENSOR_COLUMNS if c in df.columns]
    df[numeric_cols] = (
        df[numeric_cols]
        .rolling(window=window, min_periods=1, center=True)
        .median()
    )
    return df


# ---------------------------------------------------------------------------
# Dataset statistics (Table 4.7)
# ---------------------------------------------------------------------------
def _gap_summary(df: pd.DataFrame, threshold_hours: float) -> Dict[str, Any]:
    """Count gaps from the original (non-interpolated) timestamps."""
    original = df[df["imputation_flag"] == "ORIGINAL"]
    if len(original) <= 1:
        return {
            "total_gaps_detected": 0,
            "gaps_under_2h_interpolated": 0,
            "gaps_2h_or_more_flagged": 0,
        }

    diffs = original.index.to_series().diff().dt.total_seconds().fillna(0)
    gap_mask = diffs > 300  # more than one 5-minute interval
    gaps_sec = diffs[gap_mask]

    under_2h = int((gaps_sec <= threshold_hours * 3600).sum())
    over_2h = int((gaps_sec > threshold_hours * 3600).sum())

    return {
        "total_gaps_detected": int(gap_mask.sum()),
        "gaps_under_2h_interpolated": under_2h,
        "gaps_2h_or_more_flagged": over_2h,
    }


def compute_dataset_statistics(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the dataset-quality summary required for Table 4.7."""
    start = df.index.min()
    end = df.index.max()
    duration_days = float((end - start).total_seconds() / 86400.0)

    total_raw_5min_records = int(len(df))
    expected_5min_records = int((duration_days * 24 * 60) / 5) + 1
    availability_rate_pct = round(
        100.0 * total_raw_5min_records / expected_5min_records, 2
    )

    quality_counts = df["quality_flag"].value_counts().to_dict()
    imputation_counts = df["imputation_flag"].value_counts().to_dict()

    per_variable = {}
    for col in SENSOR_COLUMNS:
        if col not in df.columns:
            continue
        valid = int(df[col].notna().sum())
        per_variable[col] = {
            "valid_records": valid,
            "availability_pct": round(100.0 * valid / total_raw_5min_records, 2),
        }

    return {
        "collection_duration_days": round(duration_days, 2),
        "collection_start": start.strftime("%Y-%m-%d"),
        "collection_end": end.strftime("%Y-%m-%d"),
        "total_raw_5min_records": total_raw_5min_records,
        "expected_5min_records": expected_5min_records,
        "availability_rate_pct": availability_rate_pct,
        "gap_summary": _gap_summary(df, cfg["preprocessing"]["gap_fill_threshold_hours"]),
        "quality_flags": {k: int(v) for k, v in quality_counts.items()},
        "imputation_flags": {k: int(v) for k, v in imputation_counts.items()},
        "per_variable_availability": per_variable,
    }


# ---------------------------------------------------------------------------
# Orchestration and export
# ---------------------------------------------------------------------------
def preprocess(cfg: Dict[str, Any], subset_days: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the full preprocessing pipeline and return the processed DataFrame + stats."""
    df = load_raw_data(cfg, subset_days=subset_days)
    df = validate_ranges(df, cfg)
    df = detect_outliers(df, cfg)
    df = classify_and_fill_gaps(df, cfg)
    df = apply_rolling_median(df, cfg)

    # Final chronological integrity check
    if not df.index.is_monotonic_increasing:
        raise ValueError("Preprocessed DataFrame is not chronologically sorted")
    if df.index.duplicated().any():
        raise ValueError("Duplicate timestamps remain after preprocessing")

    stats = compute_dataset_statistics(df, cfg)
    return df, stats


def save_processed_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> Path:
    """Save the preprocessed DataFrame to ``data/processed/preprocessed_5min.csv``."""
    out_path = get_project_root() / "data" / "processed" / "preprocessed_5min.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    return out_path


def export_dataset_statistics(stats: Dict[str, Any], cfg: Dict[str, Any]) -> Path:
    """Export Table 4.7 to ``results/tables/dataset_statistics.json``."""
    import json

    out_dir = Path(cfg["paths"]["tables_dir"])
    if not out_dir.is_absolute():
        out_dir = get_project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_statistics.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return out_path


__all__ = [
    "load_raw_data",
    "validate_ranges",
    "detect_outliers",
    "classify_and_fill_gaps",
    "apply_rolling_median",
    "compute_dataset_statistics",
    "preprocess",
    "save_processed_data",
    "export_dataset_statistics",
]
