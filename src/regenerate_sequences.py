"""
Regenerate `results/sequences.npz` containing X and combined PM2.5+PM10 targets.

Run from project root:
        python -m src.regenerate_sequences

This script:
- Prefers the scaled 5-minute feature table `results/features_5min_scaled.csv`.
- Uses `model/feature_cols.json` for feature ordering (auto-syncs if missing or mismatched).
- Uses `src.preprocessing.generate_sequences()` to create windows for `pm2_5`.
- Extracts 6h/12h/24h targets (3 horizons) for PM2.5 and PM10.
- Saves `results/sequences.npz` with arrays `X`, `y` (6 targets) and `timestamps`.
"""
from pathlib import Path
import argparse
import json
import sys
from typing import List

import numpy as np
import pandas as pd


def regenerate_sequences(workspace: Path, seq_len: int = 2016, horizon: int = 288, stride: int = 1, assume_start: str = None, freq: str = '5T'):
    results = workspace / "results"
    # prefer 5-minute native features file; fall back to hourly legacy file if absent
    features_csv = results / "features_5min_scaled.csv"
    if not features_csv.exists():
        features_csv = results / "features_hourly_scaled.csv"
    feature_json = workspace / "model" / "feature_cols.json"
    out_npz = results / "sequences.npz"

    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")
    if not feature_json.exists():
        feature_json.parent.mkdir(parents=True, exist_ok=True)
        feature_json.write_text("[]", encoding="utf-8")

    # load scaled features (already scaled)
    df = pd.read_csv(str(features_csv), index_col=0, parse_dates=True)
    # if the CSV lacked a DatetimeIndex and the first column was treated as the index,
    # re-read without `index_col` so feature columns like 'pm2_5' remain available.
    if 'pm2_5' not in df.columns:
        df = pd.read_csv(str(features_csv), index_col=None)

    with open(str(feature_json), 'r', encoding='utf-8') as f:
        feature_cols: List[str] = json.load(f)

    # basic sanity check: feature cols present in df
    missing = [c for c in feature_cols if c not in df.columns]
    if missing or len(feature_cols) == 0:
        feature_cols = list(df.columns)
        with open(str(feature_json), 'w', encoding='utf-8') as f:
            json.dump(feature_cols, f, indent=2)
        print("Updated feature_cols.json to match features CSV")

    # ensure numeric features use smaller dtype to reduce memory usage during windowing
    numeric_cols = [c for c in feature_cols if c in df.columns]
    for c in ("pm2_5", "pm10"):
        if c in df.columns and c not in numeric_cols:
            numeric_cols.append(c)
    if numeric_cols:
        try:
            df[numeric_cols] = df[numeric_cols].astype('float32')
        except Exception:
            # fallback: attempt coercion column-by-column
            for c in numeric_cols:
                try:
                    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
                except Exception:
                    pass

    # lazy import preprocessing to avoid package import issues when executed as module
    from src import preprocessing as pp

    # infer steps per hour from index frequency; if index is not datetime, optionally reconstruct
    median_delta = df.index.to_series().diff().median()
    # convert to seconds safely (handles numpy.timedelta64 and pd.Timedelta)
    try:
        median_seconds = pd.to_timedelta(median_delta).total_seconds()
    except Exception:
        median_seconds = 0

    if pd.isna(median_delta) or median_seconds == 0:
        if assume_start is not None:
            print(f"Index not datetime-like or zero-delta — constructing DatetimeIndex from {assume_start} with freq={freq}")
            # Try date_range with the provided freq; be tolerant of pandas frequency aliases
            try:
                idx = pd.date_range(start=pd.to_datetime(assume_start), periods=len(df), freq=freq)
            except Exception:
                # common alias: 'T' (minutes) may not be recognized in some pandas builds; try 'min'
                try:
                    freq2 = freq.replace('T', 'min')
                    idx = pd.date_range(start=pd.to_datetime(assume_start), periods=len(df), freq=freq2)
                except Exception:
                    # fallback: if freq is like '5T' or '5min', extract integer minutes
                    try:
                        import re
                        m = re.match(r"^(\d+)([A-Za-z]+)$", freq)
                        if m and m.group(2).lower() in ('t', 'min'):
                            minutes = int(m.group(1))
                            idx = pd.date_range(start=pd.to_datetime(assume_start), periods=len(df), freq=pd.offsets.Minute(minutes))
                        else:
                            raise
                    except Exception as e:
                        raise RuntimeError(f"Failed to construct DatetimeIndex with freq='{freq}': {e}") from e
            df.index = idx
            median_delta = df.index.to_series().diff().median()
            median_seconds = pd.to_timedelta(median_delta).total_seconds()
        else:
            raise RuntimeError("Cannot infer sampling frequency from features index; provide --assume-start to reconstruct an index")

    steps_per_hour = int(round(3600.0 / median_seconds))
    steps_per_hour = max(1, steps_per_hour)
    horizon_hours = [6, 12, 24]
    horizon_steps = [h * steps_per_hour for h in horizon_hours]
    max_horizon = max(horizon_steps)

    print(
        f"Generating sequences with seq_len={seq_len}, max_horizon={max_horizon} steps, "
        f"stride={stride}, steps_per_hour={steps_per_hour}"
    )
    
    # Two-pass memmap writer to avoid allocating the entire dataset in RAM
    n_features = len(feature_cols)
    last_start = len(df) - max_horizon - seq_len + 1
    if last_start <= 0:
        raise RuntimeError(f"DataFrame too short (len={len(df)}) for seq_len={seq_len}, horizon={max_horizon}")

    # First pass: count valid windows
    valid_count = 0
    scan_windows = 0
    for start in range(0, last_start, stride):
        scan_windows += 1
        if scan_windows % 5000 == 0:
            print(f"  Scanned {scan_windows} windows...")
        end = start + seq_len
        try:
            x_win = df[feature_cols].iloc[start:end].to_numpy(dtype='float32', copy=False)
            y_pm25_win = df['pm2_5'].iloc[end:end + max_horizon].to_numpy(dtype='float32', copy=False)
            y_pm10_win = df['pm10'].iloc[end:end + max_horizon].to_numpy(dtype='float32', copy=False)
        except Exception:
            continue
        if len(x_win) < seq_len or len(y_pm25_win) < max_horizon or len(y_pm10_win) < max_horizon:
            continue
        if np.isnan(x_win).any() or np.isnan(y_pm25_win).any() or np.isnan(y_pm10_win).any():
            continue
        if 'imputation_flag' in df.columns:
            flags = df['imputation_flag'].iloc[start:end + max_horizon]
            if (flags == "GAP").any():
                continue
        valid_count += 1

    if valid_count == 0:
        raise RuntimeError("No sequences generated — check input features and parameters")

    print(f"Found {valid_count} valid sequences; allocating memmaps...")

    # Allocate memmap files in results dir
    X_path = results / 'sequences_X.dat'
    y_pm25_path = results / 'sequences_y_pm25.dat'
    y_pm10_path = results / 'sequences_y_pm10.dat'
    X_mm = np.memmap(str(X_path), dtype='float32', mode='w+', shape=(valid_count, seq_len, n_features))
    y_pm25_mm = np.memmap(str(y_pm25_path), dtype='float32', mode='w+', shape=(valid_count, max_horizon))
    y_pm10_mm = np.memmap(str(y_pm10_path), dtype='float32', mode='w+', shape=(valid_count, max_horizon))
    timestamps_arr = np.empty(valid_count, dtype='datetime64[ns]')

    # Second pass: fill memmaps
    write_i = 0
    scan_windows = 0
    for start in range(0, last_start, stride):
        scan_windows += 1
        if scan_windows % 5000 == 0:
            print(f"  Processed {scan_windows} windows (second pass)...")
        end = start + seq_len
        try:
            x_win = df[feature_cols].iloc[start:end].to_numpy(dtype='float32', copy=False)
            y_pm25_win = df['pm2_5'].iloc[end:end + max_horizon].to_numpy(dtype='float32', copy=False)
            y_pm10_win = df['pm10'].iloc[end:end + max_horizon].to_numpy(dtype='float32', copy=False)
        except Exception:
            continue
        if len(x_win) < seq_len or len(y_pm25_win) < max_horizon or len(y_pm10_win) < max_horizon:
            continue
        if np.isnan(x_win).any() or np.isnan(y_pm25_win).any() or np.isnan(y_pm10_win).any():
            continue
        if 'imputation_flag' in df.columns:
            flags = df['imputation_flag'].iloc[start:end + max_horizon]
            if (flags == "GAP").any():
                continue

        X_mm[write_i, :, :] = x_win
        y_pm25_mm[write_i, :] = y_pm25_win
        y_pm10_mm[write_i, :] = y_pm10_win
        timestamps_arr[write_i] = df.index[end].to_datetime64()
        write_i += 1

    if write_i != valid_count:
        print(f"Warning: expected {valid_count} valid windows but wrote {write_i}")

    # extract 6h/12h/24h targets (index is step-1)
    idxs = [h - 1 for h in horizon_steps]
    pm25_targets = y_pm25_mm[:, idxs]
    pm10_targets = y_pm10_mm[:, idxs]
    y_combined = np.column_stack([
        pm25_targets[:, 0], pm10_targets[:, 0],
        pm25_targets[:, 1], pm10_targets[:, 1],
        pm25_targets[:, 2], pm10_targets[:, 2],
    ])

    # prepare memmap-backed X for saving
    X_mm_read = np.memmap(str(X_path), dtype='float32', mode='r', shape=(valid_count, seq_len, n_features))
    timestamps_valid = timestamps_arr.copy()

    # backup existing sequences.npz
    if out_npz.exists():
        backup = results / 'sequences.backup.npz'
        # If a previous backup already exists on Windows, remove it first.
        if backup.exists():
            try:
                backup.unlink()
            except Exception:
                # If unlink fails (permissions?), rotate to a timestamped backup name
                import time
                backup = results / f"sequences.backup.{int(time.time())}.npz"
        print(f"Backing up existing {out_npz} -> {backup}")
        out_npz.rename(backup)

    # Save final NPZ using the memmap as the X input (np.savez will read from memmap)
    try:
        np.savez(str(out_npz), X=X_mm_read, y=y_combined, timestamps=timestamps_valid)
        print(f"Wrote {out_npz}: X.shape=({valid_count}, {seq_len}, {n_features}), y.shape={y_combined.shape}")
    finally:
        # clean up temporary memmap files
        try:
            del X_mm_read
            del X_mm
            del y_pm25_mm
            del y_pm10_mm
        except Exception:
            pass
        for p in (X_path, y_pm25_path, y_pm10_path):
            try:
                p.unlink()
            except Exception:
                pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Regenerate sequences.npz with PM2.5+PM10 targets")
    p.add_argument('--workspace', type=str, default='.', help='Project workspace root (default: current dir)')
    p.add_argument('--seq-len', type=int, default=2016)
    p.add_argument('--horizon', type=int, default=288)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--assume-start', type=str, default=None,
                   help='If features CSV lacks a DatetimeIndex, construct one starting at this ISO datetime')
    p.add_argument('--freq', type=str, default='5T', help='Frequency string used with --assume-start (default: 5T)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ws = Path(args.workspace).resolve()
    regenerate_sequences(
        ws,
        seq_len=args.seq_len,
        horizon=args.horizon,
        stride=args.stride,
        assume_start=args.assume_start,
        freq=args.freq,
    )


if __name__ == '__main__':
    main()
