"""ALLERT feature engineering module.

Builds the input matrix from the preprocessed 5-minute sensor data.
All parameters (lag offsets, rolling windows, velocity horizons, cyclical
features, exposure weights, ratio/interaction features) are read from ``config.yaml``.

Feature groups:
    - Raw inputs (pm2_5, pm10, temperature, humidity)
    - Lag features for PM2.5/PM10 at short and long horizons
    - Rolling mean/std/min/max for multiple windows per pollutant
    - Velocity and acceleration (rate-of-change and second difference)
    - PM10/PM2.5 ratio features (current and lagged)
    - Cross-variable interactions (temperature × humidity, weather × PM)
    - Cyclical encodings (hour, dayofweek, month)
    - Cumulative exposure for PM2.5/PM10
    - Spike-lag, spike-rate, and rapid-change threshold indicators

Thesis traceability:
    - Chapter 3.2, "Feature Engineering and Normalization"
    - Chapter 4, Table 4.8 and Fig. 4.5–4.7
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd


SENSOR_COLUMNS = ["pm2_5", "pm10", "temperature", "humidity"]


# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------
def _records_per_hour(df: pd.DataFrame) -> int:
    """Infer records per hour from the DatetimeIndex median interval."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by a DatetimeIndex")
    median_delta = df.index.to_series().diff().median()
    if pd.isna(median_delta) or median_delta == pd.Timedelta(0):
        raise ValueError("Cannot infer sampling frequency")
    records_per_hour = int(round(pd.Timedelta(hours=1) / median_delta))
    return max(1, records_per_hour)


def _hours_to_steps(df: pd.DataFrame, hours: float) -> int:
    return int(round(hours * _records_per_hour(df)))


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
def add_cyclical_time_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add sine/cosine encodings for the cyclical time features in config."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by a DatetimeIndex")

    cyclical = cfg["features"]["cyclical_features"]
    idx = df.index
    out = pd.DataFrame(index=idx)

    # Mapping from config names to pandas accessors and their periods.
    mapping = {
        "hour": (idx.hour, 24),
        "dayofweek": (idx.dayofweek, 7),
        "month": (idx.month - 1, 12),
    }

    for name in cyclical:
        if name not in mapping:
            raise ValueError(f"Unsupported cyclical feature: {name}")
        values, period = mapping[name]
        out[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        out[f"{name}_cos"] = np.cos(2 * np.pi * values / period)

    return out


def compute_lag_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Create lag features for PM2.5 and PM10 at the configured horizons."""
    pollutants = cfg["features"]["target_variables"]
    lags = cfg["features"]["lag_hours"]
    out = pd.DataFrame(index=df.index)

    for col in pollutants:
        if col not in df.columns:
            continue
        for h in lags:
            steps = _hours_to_steps(df, h)
            out[f"{col}_lag_{h}h"] = df[col].shift(steps)

    return out


def compute_rolling_stats(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Create rolling mean/std/min/max summaries for PM2.5 and PM10."""
    pollutants = cfg["features"]["target_variables"]
    windows = cfg["features"]["rolling_windows_hours"]
    out = pd.DataFrame(index=df.index)

    for h in windows:
        steps = _hours_to_steps(df, h)
        for col in pollutants:
            if col not in df.columns:
                continue
            roll = df[col].rolling(window=steps, min_periods=1)
            out[f"{col}_mean_{h}h"] = roll.mean()
            out[f"{col}_std_{h}h"] = roll.std()
            # Skip min/max for the 7-day window to keep dimensionality manageable
            if h != 168:
                out[f"{col}_min_{h}h"] = roll.min()
                out[f"{col}_max_{h}h"] = roll.max()

    return out


def compute_velocity_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Create hourly rate-of-change features for PM2.5 and PM10."""
    pollutants = cfg["features"]["target_variables"]
    horizons = cfg["features"]["velocity_hours"]
    out = pd.DataFrame(index=df.index)

    for col in pollutants:
        if col not in df.columns:
            continue
        for h in horizons:
            steps = _hours_to_steps(df, h)
            out[f"{col}_vel_{h}h"] = (df[col] - df[col].shift(steps)) / float(h)

    return out


def compute_acceleration_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Create second-difference (acceleration) features for PM2.5 and PM10."""
    pollutants = cfg["features"].get("target_variables", ["pm2_5", "pm10"])
    horizons = cfg["features"].get("acceleration_hours", [1, 3, 6, 12])
    out = pd.DataFrame(index=df.index)
    for col in pollutants:
        if col not in df.columns:
            continue
        for h in horizons:
            steps = _hours_to_steps(df, h)
            # v(t) - v(t-h) where v(t) = x(t) - x(t-h)
            out[f"{col}_accel_{h}h"] = (
                (df[col] - df[col].shift(steps))
                - (df[col].shift(steps) - df[col].shift(2 * steps))
            ) / float(h)
    return out


def compute_ratio_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Create PM10/PM2.5 ratio features, current and lagged."""
    if "pm2_5" not in df.columns or "pm10" not in df.columns:
        return pd.DataFrame(index=df.index)
    lags = cfg["features"].get("ratio_lag_hours", [0, 1, 6, 24])
    out = pd.DataFrame(index=df.index)
    for h in lags:
        if h == 0:
            ratio = df["pm10"] / df["pm2_5"].replace(0, np.nan)
            out["pm_ratio_current"] = ratio
        else:
            steps = _hours_to_steps(df, h)
            ratio = df["pm10"].shift(steps) / df["pm2_5"].shift(steps).replace(0, np.nan)
            out[f"pm_ratio_lag_{h}h"] = ratio
    return out


def compute_interaction_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Create cross-variable interaction features (weather × PM)."""
    pollutants = cfg["features"].get("target_variables", ["pm2_5", "pm10"])
    interactions = cfg["features"].get("interactions", [])
    out = pd.DataFrame(index=df.index)
    if not interactions:
        # Default set: temperature × humidity, and each weather variable × each pollutant
        weather = ["temperature", "humidity"]
        interactions = [("temperature", "humidity")]
        for w in weather:
            for p in pollutants:
                if w in df.columns and p in df.columns:
                    interactions.append((w, p))
    for a, b in interactions:
        if a in df.columns and b in df.columns:
            out[f"{a}_x_{b}"] = df[a] * df[b]
    return out


def compute_ema_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Create exponentially-weighted moving average features for pollutants."""
    pollutants = cfg["features"].get("target_variables", ["pm2_5", "pm10"])
    spans = cfg["features"].get("ema_windows_hours", [1, 6, 24])
    out = pd.DataFrame(index=df.index)
    for col in pollutants:
        if col not in df.columns:
            continue
        for h in spans:
            steps = _hours_to_steps(df, h)
            # Convert window in steps to span for pandas EWM (center-of-mass approximation)
            span = max(1.0, steps / 2.0)
            out[f"{col}_ema_{h}h"] = df[col].ewm(span=span, min_periods=1, adjust=False).mean()
    return out


def compute_spike_lag_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Binary indicators for recent high-PM exceedances.

    These give the LSTM explicit rare-event signals that are otherwise hard to
    extract from smoothed rolling statistics.
    """
    if not cfg["features"].get("add_spike_lag_features", False):
        return pd.DataFrame(index=df.index)

    pollutants = cfg["features"].get("target_variables", ["pm2_5", "pm10"])
    thresholds = {
        "pm2_5": cfg["features"].get("spike_thresholds_pm25", [35, 50]),
        "pm10": cfg["features"].get("spike_thresholds_pm10", [75, 100]),
    }
    lag_hours = cfg["features"].get("spike_lag_hours", [1, 3, 6, 24])

    out = pd.DataFrame(index=df.index)
    for col in pollutants:
        if col not in df.columns or col not in thresholds:
            continue
        for thr in thresholds[col]:
            for h in lag_hours:
                steps = _hours_to_steps(df, h)
                out[f"{col}_spike_gt{thr}_{h}h"] = (
                    (df[col].shift(steps) > thr).astype(float)
                )
    return out


def compute_spike_rate_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Fraction of time a pollutant exceeded a threshold over recent windows."""
    if not cfg["features"].get("add_spike_rate_features", False):
        return pd.DataFrame(index=df.index)

    pollutants = cfg["features"].get("target_variables", ["pm2_5", "pm10"])
    thresholds = {
        "pm2_5": cfg["features"].get("spike_thresholds_pm25", [35, 50]),
        "pm10": cfg["features"].get("spike_thresholds_pm10", [75, 100]),
    }
    windows = cfg["features"].get("spike_rate_windows_hours", [6, 24, 72])

    out = pd.DataFrame(index=df.index)
    for col in pollutants:
        if col not in df.columns or col not in thresholds:
            continue
        for thr in thresholds[col]:
            flag = (df[col] > thr).astype(float)
            for h in windows:
                steps = _hours_to_steps(df, h)
                out[f"{col}_spikerate_gt{thr}_{h}h"] = flag.rolling(
                    window=steps, min_periods=1
                ).mean()
    return out


def compute_roc_threshold_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Binary flags for rapid PM increases."""
    if not cfg["features"].get("add_roc_threshold_features", False):
        return pd.DataFrame(index=df.index)

    pollutants = cfg["features"].get("target_variables", ["pm2_5", "pm10"])
    thresholds = cfg["features"].get("roc_thresholds_ugm3_per_hour", [5, 10, 15])
    horizons = cfg["features"].get("roc_hours", [0.5, 1, 3])

    out = pd.DataFrame(index=df.index)
    for col in pollutants:
        if col not in df.columns:
            continue
        for h in horizons:
            steps = _hours_to_steps(df, h)
            roc = (df[col] - df[col].shift(steps)) / float(h)
            for thr in thresholds:
                out[f"{col}_roc_gt{thr}_{h}h"] = (roc > thr).astype(float)
    return out


def compute_cumulative_exposure(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Create the 72-hour distributed-lag cumulative exposure features."""
    pollutants = cfg["features"]["target_variables"]
    weights = cfg["risk"]["distributed_lag_weights"]
    hours = cfg["risk"]["distributed_lag_hours"]
    out = pd.DataFrame(index=df.index)

    if len(weights) != len(hours):
        raise ValueError(
            "distributed_lag_weights and distributed_lag_hours must have the same length"
        )

    for col in pollutants:
        if col not in df.columns:
            continue
        exposure = None
        for w, h in zip(weights, hours):
            steps = _hours_to_steps(df, h)
            shifted = df[col] if h == 0 else df[col].shift(steps)
            exposure = shifted * w if exposure is None else exposure + shifted * w
        out[f"{col}_cum_72h"] = exposure

    return out


# ---------------------------------------------------------------------------
# Full feature matrix
# ---------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Build the complete ~56-feature matrix.

    Parameters
    ----------
    df:
        Preprocessed DataFrame indexed by timestamp. May contain non-numeric
        provenance columns (e.g. ``quality_flag``, ``imputation_flag``); these
        are ignored.
    cfg:
        Loaded configuration dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only engineered numeric features, indexed by
        timestamp. Rows with NaNs are preserved so callers can decide how to
        handle them.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a timestamp column")

    features = pd.DataFrame(index=df.index)

    # Raw inputs
    raw_inputs = cfg["features"]["raw_inputs"]
    for col in raw_inputs:
        if col in df.columns:
            features[col] = df[col]

    # Engineered features
    features = pd.concat(
        [
            features,
            compute_lag_features(df, cfg),
            compute_rolling_stats(df, cfg),
            compute_velocity_features(df, cfg),
            compute_acceleration_features(df, cfg),
            compute_ratio_features(df, cfg),
            compute_interaction_features(df, cfg),
            compute_ema_features(df, cfg),
            add_cyclical_time_features(df, cfg),
            compute_cumulative_exposure(df, cfg),
            compute_spike_lag_features(df, cfg),
            compute_spike_rate_features(df, cfg),
            compute_roc_threshold_features(df, cfg),
        ],
        axis=1,
    )

    # Drop any non-numeric columns that may have slipped through
    features = features.select_dtypes(include=[np.number])
    return features


def get_feature_column_names(features_df: pd.DataFrame) -> List[str]:
    """Return the ordered list of feature column names."""
    return list(features_df.columns)


__all__ = [
    "add_cyclical_time_features",
    "compute_lag_features",
    "compute_rolling_stats",
    "compute_velocity_features",
    "compute_acceleration_features",
    "compute_ratio_features",
    "compute_interaction_features",
    "compute_ema_features",
    "compute_spike_lag_features",
    "compute_spike_rate_features",
    "compute_roc_threshold_features",
    "compute_cumulative_exposure",
    "build_feature_matrix",
    "get_feature_column_names",
]
