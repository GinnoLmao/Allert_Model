"""ALLERT sliding-window sequence builder.

Converts the scaled feature DataFrame into 3D tensors for the Bi-LSTM:
    X: (N, 2016, F)
    y: (N, 6)

Target ordering:
    [pm2_5_6h, pm10_6h, pm2_5_12h, pm10_12h, pm2_5_24h, pm10_24h]

Sequences are generated with stride=1. Any window containing a ``GAP``-flagged
row or NaN values is skipped. Because the full tensor can be very large, the
builder writes ``X`` to a memory-mapped file and returns the memmap array, while
``y`` is kept in memory (it is small). The loader also remains backward-
compatible with the older ``results/sequences.npz`` artifact so legacy scripts
can still run against historical outputs.

Thesis traceability:
    - Chapter 3.2, "Sample Size and Dataset Construction"
    - Chapter 4, Table 4.8
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config_loader import get_project_root


TARGET_ORDER = [
    ("pm2_5_target", 6),
    ("pm10_target", 6),
    ("pm2_5_target", 12),
    ("pm10_target", 12),
    ("pm2_5_target", 24),
    ("pm10_target", 24),
]


def _records_per_hour(df: pd.DataFrame) -> int:
    median_delta = df.index.to_series().diff().median()
    if pd.isna(median_delta) or median_delta == pd.Timedelta(0):
        raise ValueError("Cannot infer sampling frequency")
    return max(1, int(round(pd.Timedelta(hours=1) / median_delta)))


def _load_feature_cols(cfg: Dict[str, Any]) -> List[str]:
    """Load the canonical feature column list from ``model/feature_cols.json``."""
    path = Path(cfg["paths"]["model_dir"]) / "feature_cols.json"
    if not path.is_absolute():
        path = get_project_root() / path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sequences(
    cfg: Dict[str, Any],
    scaled_path: Path = None,
    feature_cols: List[str] = None,
    out_dir: Path = None,
    stride: int = 1,
) -> Tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build sliding-window sequences from the scaled feature CSV.

    Parameters
    ----------
    cfg:
        Loaded configuration.
    scaled_path:
        Path to ``features_5min_scaled.csv``. Defaults to config path.
    feature_cols:
        Ordered list of feature columns for X. Defaults to ``feature_cols.json``.
    out_dir:
        Directory to write sequence artifacts. Defaults to ``results/``.

    Returns
    -------
    X : np.memmap
        Shape ``(N, seq_len, n_features)``.
    y : np.ndarray
        Shape ``(N, 6)``.
    timestamps : np.ndarray
        Prediction-start timestamps for each window.
    train_mask, val_mask, test_mask : np.ndarray
        Boolean masks assigning each sequence to a split.
    meta : dict
        Metadata with shapes, paths, and split counts.
    """
    if scaled_path is None:
        scaled_path = get_project_root() / "data" / "processed" / "features_5min_scaled.csv"
    if out_dir is None:
        out_dir = get_project_root() / "results"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scaled_path, parse_dates=["timestamp"], index_col="timestamp")

    if feature_cols is None:
        feature_cols = _load_feature_cols(cfg)

    # Ensure all requested feature columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in scaled CSV: {missing}")

    target_cols = [t[0] for t in TARGET_ORDER]
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in scaled CSV: {missing_targets}")

    if "imputation_flag" not in df.columns:
        raise ValueError("Scaled CSV must contain 'imputation_flag' for GAP exclusion")

    seq_len = cfg["model"]["sequence_length"]
    rph = _records_per_hour(df)
    horizons = cfg["model"]["forecast_horizons_hours"]
    horizon_steps = [h * rph for h in horizons]
    max_horizon = max(horizon_steps)

    n = len(df)
    if n < seq_len + max_horizon:
        raise ValueError(
            f"Data too short (n={n}) for seq_len={seq_len} and max_horizon={max_horizon}"
        )

    # Split boundaries in row space
    train_end = int(np.floor(n * cfg["training"]["train_split"]))
    val_end = int(np.floor(n * (cfg["training"]["train_split"] + cfg["training"]["val_split"])))

    feature_array = df[feature_cols].to_numpy(dtype="float32")
    imputation_flags = df["imputation_flag"].to_numpy()
    timestamps = df.index.to_numpy()

    # Precompute target values for every possible prediction-start index and horizon.
    # target_matrix[end, i] is the value of TARGET_ORDER[i] at index end + horizon - 1.
    target_matrix = np.full((n, 6), np.nan, dtype="float32")
    for col_i, (target_col, h) in enumerate(TARGET_ORDER):
        horizon = horizon_steps[horizons.index(h)]
        shifted = df[target_col].shift(-(horizon - 1))
        target_matrix[:, col_i] = shifted.to_numpy(dtype="float32")

    # First pass: count valid windows
    last_start = n - seq_len - max_horizon + 1
    valid_count = 0
    valid_starts = []
    print(f"Scanning {last_start} possible windows (seq_len={seq_len}, max_horizon={max_horizon})...")

    for start in range(0, last_start, stride):
        end = start + seq_len
        if (imputation_flags[start:end] == "GAP").any():
            continue
        if (imputation_flags[end : end + max_horizon] == "GAP").any():
            continue
        if np.isnan(feature_array[start:end]).any():
            continue
        if np.isnan(target_matrix[end]).any():
            continue
        valid_count += 1
        valid_starts.append(start)

    if valid_count == 0:
        raise RuntimeError("No valid sequences generated — check input data and flags")

    print(f"Found {valid_count} valid sequences; allocating memmaps...")

    # Allocate memmap files (remove stale files first to avoid Windows lock/size conflicts)
    X_path = out_dir / "sequences_X.dat"
    y_path = out_dir / "sequences_y.dat"
    for p in (X_path, y_path):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    X_mm = np.memmap(str(X_path), dtype="float32", mode="w+", shape=(valid_count, seq_len, len(feature_cols)))
    y_mm = np.memmap(str(y_path), dtype="float32", mode="w+", shape=(valid_count, 6))
    ts_arr = np.empty(valid_count, dtype="datetime64[ns]")
    end_idx_arr = np.empty(valid_count, dtype=np.int64)

    # Second pass: fill arrays
    for write_i, start in enumerate(valid_starts):
        end = start + seq_len
        X_mm[write_i] = feature_array[start:end]
        y_mm[write_i] = target_matrix[end]
        ts_arr[write_i] = timestamps[end]
        end_idx_arr[write_i] = end

    X_mm.flush()
    y_mm.flush()

    # Load read-only memmaps to return
    X_read = np.memmap(str(X_path), dtype="float32", mode="r", shape=(valid_count, seq_len, len(feature_cols)))
    y_read = np.memmap(str(y_path), dtype="float32", mode="r", shape=(valid_count, 6))

    # Build chronological split masks based on target-row boundaries
    end_target_idx = end_idx_arr + max_horizon - 1
    train_mask = end_target_idx < train_end
    val_mask = (end_idx_arr >= train_end) & (end_target_idx < val_end)
    test_mask = (end_idx_arr >= val_end) & (end_target_idx < n)

    meta = {
        "X_shape": [int(valid_count), int(seq_len), int(len(feature_cols))],
        "y_shape": [int(valid_count), 6],
        "n_features": int(len(feature_cols)),
        "seq_len": int(seq_len),
        "horizon_steps": horizon_steps,
        "train_count": int(train_mask.sum()),
        "val_count": int(val_mask.sum()),
        "test_count": int(test_mask.sum()),
        "X_path": str(X_path),
        "y_path": str(y_path),
    }

    # Save lightweight metadata and split masks
    with open(out_dir / "sequences_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    np.save(out_dir / "sequences_train_mask.npy", train_mask)
    np.save(out_dir / "sequences_val_mask.npy", val_mask)
    np.save(out_dir / "sequences_test_mask.npy", test_mask)
    np.save(out_dir / "sequences_timestamps.npy", ts_arr)

    return X_read, np.array(y_read), ts_arr, train_mask, val_mask, test_mask, meta


def load_sequences(out_dir: Path) -> Tuple[np.memmap, np.memmap, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reload previously built sequence artifacts and split masks.

    The current pipeline stores sequence data as memmaps plus metadata files.
    For compatibility, this function also reads the legacy ``sequences.npz``
    container when the newer metadata is not present.
    """
    out_dir = Path(out_dir)
    meta_path = out_dir / "sequences_meta.json"

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        X = np.memmap(meta["X_path"], dtype="float32", mode="r", shape=tuple(meta["X_shape"]))
        y = np.memmap(meta["y_path"], dtype="float32", mode="r", shape=tuple(meta["y_shape"]))
        train_mask = np.load(out_dir / "sequences_train_mask.npy")
        val_mask = np.load(out_dir / "sequences_val_mask.npy")
        test_mask = np.load(out_dir / "sequences_test_mask.npy")
        timestamps = np.load(out_dir / "sequences_timestamps.npy")
        return X, y, timestamps, train_mask, val_mask, test_mask

    legacy_path = out_dir / "sequences.npz"
    if not legacy_path.exists():
        raise FileNotFoundError(
            f"No sequence artifacts found in {out_dir}; expected {meta_path.name} or {legacy_path.name}"
        )

    data = np.load(str(legacy_path), allow_pickle=True)
    X = data["X"]
    y = data["y"]
    timestamps = data["timestamps"]
    timestamps = np.array([pd.to_datetime(t) for t in timestamps])

    n = len(X)
    train_end = int(np.floor(n * 0.70))
    val_end = int(np.floor(n * 0.85))
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True
    return X, y, timestamps, train_mask, val_mask, test_mask


__all__ = [
    "build_sequences",
    "load_sequences",
    "TARGET_ORDER",
]
