"""ALLERT scaling and chronological split utilities.

Implements the two critical requirements from the design docs:

1. Chronological train/val/test split — no random shuffling.
2. Scaler fitted **only** on the training partition; the fitted scaler is then
   applied to validation and test to prevent data leakage.

Artifacts produced:
    - ``model/scaler.joblib``
    - ``data/splits/train_val_test_indices.json``
    - ``data/processed/features_5min_scaled.csv``
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config_loader import get_project_root
from src.target_scaling import TARGET_COLUMNS, TargetScaler


def _safe_remove(path: Path) -> None:
    """Remove a file if it exists, ignoring errors (e.g. Windows lock races)."""
    try:
        if path.exists():
            os.remove(path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------
def chronological_split(
    df: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a chronologically sorted DataFrame into train/val/test."""
    n = len(df)
    train_frac = cfg["training"]["train_split"]
    val_frac = cfg["training"]["val_split"]

    train_end = int(np.floor(n * train_frac))
    val_end = int(np.floor(n * (train_frac + val_frac)))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


# ---------------------------------------------------------------------------
# Scaler wrapper
# ---------------------------------------------------------------------------
class FeatureScaler:
    """Train-only scaler wrapper for DataFrames.

    Supports ``minmax`` and ``standard`` scaling via the config. The scaler is
    fitted on the training split and then applied to validation and test.
    """

    def __init__(self, scaler_type: str = "minmax"):
        if scaler_type not in {"minmax", "standard"}:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")
        self.scaler_type = scaler_type
        self.scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
        self.feature_cols: List[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "FeatureScaler":
        """Fit the scaler on the training data only."""
        self.feature_cols = list(feature_cols)
        values = df[self.feature_cols].to_numpy(dtype="float64")
        self.scaler.fit(values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame using the already-fitted scaler."""
        if not self.feature_cols:
            raise NotFittedError("FeatureScaler is not fitted. Call `fit()` first.")
        df = df.copy()
        values = df[self.feature_cols].to_numpy(dtype="float64")
        df[self.feature_cols] = self.scaler.transform(values)
        return df

    def fit_transform(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        """Fit on ``df`` and transform it (training use only)."""
        self.fit(df, feature_cols)
        return self.transform(df)

    def save(self, path: Path) -> None:
        """Persist the fitted scaler and feature column list."""
        if not self.feature_cols:
            raise NotFittedError("Nothing to save; call `fit()` first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        _safe_remove(path)
        joblib.dump(
            {"scaler": self.scaler, "feature_cols": self.feature_cols, "type": self.scaler_type},
            path,
        )

    def load(self, path: Path) -> "FeatureScaler":
        """Load a previously saved scaler."""
        payload = joblib.load(path)
        self.scaler = payload["scaler"]
        self.feature_cols = payload["feature_cols"]
        self.scaler_type = payload.get("type", "minmax")
        return self


# ---------------------------------------------------------------------------
# Build and save scaled dataset
# ---------------------------------------------------------------------------
def build_scaled_dataset(
    cfg: Dict[str, Any],
    preprocessed_path: Path = None,
    feature_cols: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureScaler, TargetScaler]:
    """Build train/val/test scaled feature sets.

    Returns the full scaled DataFrame (concatenated in chronological order),
    the three split DataFrames, the fitted feature scaler, and the fitted target
    scaler. The full DataFrame preserves ``pm2_5``, ``pm10``, ``quality_flag``
    and ``imputation_flag`` needed by the sequence builder.

    Target columns (``pm2_5_target``, ``pm10_target``) are left in original
    µg/m³ units unless ``features.scale_targets`` is set to true. In either
    case, a ``TargetScaler`` is saved so evaluation can invert predictions.
    """
    from src import feature_engineering as fe

    if preprocessed_path is None:
        preprocessed_path = get_project_root() / "data" / "processed" / "preprocessed_5min.csv"

    df = pd.read_csv(preprocessed_path, parse_dates=["timestamp"], index_col="timestamp")

    # Build full feature matrix on the observed history
    features = fe.build_feature_matrix(df, cfg)

    if feature_cols is None:
        feature_cols = fe.get_feature_column_names(features)
    else:
        # Ensure requested columns exist and are in the right order
        missing = [c for c in feature_cols if c not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        features = features[feature_cols]

    # Attach raw target columns and provenance flags needed downstream.
    # The raw PM columns are kept as targets; the versions inside ``features``
    # are scaled by the fitted scaler and used as model inputs.
    extra_frames = []
    pm_target = pd.DataFrame(index=df.index)
    for col in ["pm2_5", "pm10"]:
        if col in df.columns:
            pm_target[f"{col}_target"] = df[col]
    if not pm_target.empty:
        extra_frames.append(pm_target)

    flag_cols = [c for c in ["quality_flag", "imputation_flag"] if c in df.columns]
    if flag_cols:
        extra_frames.append(df[flag_cols])

    combined = pd.concat([features] + extra_frames, axis=1)

    # Ensure target columns exist for downstream modules
    for tcol in TARGET_COLUMNS:
        if tcol not in combined.columns:
            combined[tcol] = np.nan

    # Split chronologically on the full DataFrame
    train_df, val_df, test_df = chronological_split(combined, cfg)

    # Fit scaler **only** on training features
    scaler_type = cfg["features"].get("scaler_type", "minmax")
    scaler = FeatureScaler(scaler_type)
    scaler.fit(train_df, feature_cols)

    # Fit target scaler on training targets (identity if scale_targets is false)
    target_scaler = TargetScaler(scaler_type)
    target_scaler.fit(train_df, TARGET_COLUMNS)
    target_scaler.scaled = bool(cfg["features"].get("scale_targets", False))

    # Transform all splits with the training-fitted scaler
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # Optionally scale targets in-place so sequence builder sees scaled targets
    if target_scaler.scaled:
        for split_df in (train_scaled, val_scaled, test_scaled):
            split_df[TARGET_COLUMNS] = target_scaler.scaler.transform(
                split_df[TARGET_COLUMNS].to_numpy(dtype="float64")
            )

    # Reassemble chronologically
    full_scaled = pd.concat([train_scaled, val_scaled, test_scaled], axis=0)
    full_scaled = full_scaled.sort_index()

    # Persist target scaler
    root = get_project_root()
    model_dir = root / cfg["paths"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    target_scaler.save(model_dir / "target_scaler.joblib")

    return full_scaled, train_scaled, val_scaled, test_scaled, scaler, target_scaler


def save_split_indices(
    n_total: int, train_df: pd.DataFrame, val_df: pd.DataFrame, out_path: Path
) -> Path:
    """Save the chronological split boundaries as JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indices = {
        "n_total": n_total,
        "train_end": int(len(train_df)),
        "val_end": int(len(train_df) + len(val_df)),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(indices, f, indent=2)
    return out_path


__all__ = [
    "chronological_split",
    "FeatureScaler",
    "build_scaled_dataset",
    "save_split_indices",
]
