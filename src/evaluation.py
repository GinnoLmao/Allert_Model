"""ALLERT evaluation and baseline utilities.

Provides:
    - Per-pollutant, per-horizon forecasting metrics (MAE, RMSE, MAPE, R²)
    - PyTorch Bi-LSTM evaluation on the held-out test set
    - ARIMA(2,1,2) baseline benchmark
    - Training-convergence and Bi-LSTM-vs-ARIMA visualization figures

Exported artifacts:
    - ``results/tables/table_4_8_bilstm.json``
    - ``results/tables/arima_results.json``
    - ``results/visualizations/fig_4_5a_training_convergence.png``
    - ``results/visualizations/fig_4_5b_bilstm_vs_arima.png``

Thesis traceability:
    - Chapter 3.2, "Technical System Validation (ISO/IEC 25010)"
    - Chapter 4, Table 4.8 and Fig. 4.5
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config_loader import get_project_root
from src.pytorch_model import build_bilstm, DEVICE, load_model_with_config
from src.target_scaling import load_target_scaler


TARGET_LABELS = [
    ("PM2_5", "6h"),
    ("PM10", "6h"),
    ("PM2_5", "12h"),
    ("PM10", "12h"),
    ("PM2_5", "24h"),
    ("PM10", "24h"),
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100.0
    return float(mape) if np.isfinite(mape) else float("nan")


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan"), "R2": float("nan")}

    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mape = _safe_mape(yt, yp)
    try:
        r2 = r2_score(yt, yp)
    except Exception:
        r2 = float("nan")

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
        "R2": round(float(r2), 4),
    }


def _predict_in_batches(
    model,
    X: np.ndarray,
    batch_size: int,
    return_spikes: bool = False,
    indices: np.ndarray = None,
) -> np.ndarray:
    """Run PyTorch model inference in batches to avoid OOM.

    If ``indices`` is provided, only those rows are predicted, which keeps the
    full array on disk as a memory-mapped object. If ``indices`` is None, all
    rows of ``X`` are predicted.

    If the model returns a tuple (regression, spike_logits), only the regression
    output is returned by default. Set ``return_spikes=True`` to get the spike
    probability predictions instead.
    """
    if indices is None:
        indices = np.arange(len(X))
    preds = []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            idx_batch = indices[i : i + batch_size]
            xb = torch.from_numpy(np.asarray(X[idx_batch], dtype=np.float32)).to(DEVICE)
            out = model(xb)
            if isinstance(out, tuple):
                pred = out[1] if return_spikes else out[0]
            else:
                pred = out
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute nested metrics for the 6 target columns."""
    if y_true.shape[1] != 6 or y_pred.shape[1] != 6:
        raise ValueError("y_true and y_pred must have shape (N, 6)")

    metrics = {"PM2_5": {}, "PM10": {}}
    for i, (pollutant, horizon) in enumerate(TARGET_LABELS):
        metrics[pollutant][horizon] = _metric_row(y_true[:, i], y_pred[:, i])

    # Overall mean across the 6 targets
    all_mae = [metrics[p][h]["MAE"] for p, h in TARGET_LABELS]
    all_rmse = [metrics[p][h]["RMSE"] for p, h in TARGET_LABELS]
    all_mape = [metrics[p][h]["MAPE"] for p, h in TARGET_LABELS]
    all_r2 = [metrics[p][h]["R2"] for p, h in TARGET_LABELS]

    metrics["overall"] = {
        "MAE": round(float(np.nanmean(all_mae)), 4),
        "RMSE": round(float(np.nanmean(all_rmse)), 4),
        "MAPE": round(float(np.nanmean(all_mape)), 4),
        "R2": round(float(np.nanmean(all_r2)), 4),
    }
    return metrics


# ---------------------------------------------------------------------------
# ARIMA baseline
# ---------------------------------------------------------------------------
def arima_forecast_series(
    train_series: np.ndarray, test_length: int, order: Tuple[int, int, int] = (2, 1, 2)
) -> np.ndarray:
    """Fit ARIMA on ``train_series`` and forecast ``test_length`` steps."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:
        raise RuntimeError("statsmodels is required for the ARIMA baseline") from exc

    # Drop leading NaNs and interpolate any internal missing values
    series = pd.Series(train_series).dropna()
    if len(series) < 10:
        return np.full(test_length, np.nan)

    try:
        model = ARIMA(series, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=test_length)
        return np.asarray(forecast, dtype=float)
    except Exception:
        # Fallback to persistence forecast
        return np.full(test_length, float(series.iloc[-1]))


def arima_baseline(
    y_train: np.ndarray, y_test: np.ndarray, order: Tuple[int, int, int] = (2, 1, 2)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute ARIMA(2,1,2) forecasts for each of the 6 target columns."""
    y_pred = np.full_like(y_test, np.nan, dtype=float)
    for i in range(y_test.shape[1]):
        y_pred[:, i] = arima_forecast_series(y_train[:, i], y_test.shape[0], order=order)
    metrics = compute_metrics(y_test, y_pred)
    return y_pred, metrics


def persistence_baseline(
    y_train: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute a naive persistence baseline (last observed value repeated)."""
    last_values = np.nanmedian(y_train[-12:, :], axis=0)  # median of last hour
    y_pred = np.tile(last_values, (y_test.shape[0], 1))
    metrics = compute_metrics(y_test, y_pred)
    return y_pred, metrics


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_training_convergence(history, out_path: Path) -> Path:
    """Generate Fig 4.5a — training and validation loss convergence.

    Accepts either the Keras-style dict (``{"loss": [...], "val_loss": [...]}``)
    or the PyTorch-style list of per-epoch records.
    """
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))

    if isinstance(history, list):
        epochs = [h["epoch"] for h in history]
        train_loss = [h["loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        plt.plot(epochs, train_loss, label="Training loss", linewidth=2)
        plt.plot(epochs, val_loss, label="Validation loss", linewidth=2)
    else:
        plt.plot(history["loss"], label="Training loss", linewidth=2)
        if "val_loss" in history:
            plt.plot(history["val_loss"], label="Validation loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Composite MAE loss")
    plt.title("Fig 4.5a — Training Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def compute_error_by_regime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
) -> Dict[str, Any]:
    """Break MAE/MAPE down by hour-of-day, day-of-week, and PM level.

    Target layout: [pm2_5_6h, pm10_6h, pm2_5_12h, pm10_12h, pm2_5_24h, pm10_24h].
    """
    import pandas as pd

    ts = pd.to_datetime(timestamps)
    df = pd.DataFrame({
        "hour": ts.hour,
        "dayofweek": ts.dayofweek,
        "pm2_5_actual": y_true[:, 0],
    })

    def _pm_level(v):
        if v < 15:
            return "low"
        if v < 35:
            return "moderate"
        return "high"

    df["pm_level"] = df["pm2_5_actual"].apply(_pm_level)

    regimes = {}
    for name, group_key in [
        ("hour_of_day", "hour"),
        ("day_of_week", "dayofweek"),
        ("pm_level", "pm_level"),
    ]:
        per_regime = {}
        for val, sub in df.groupby(group_key):
            idx = sub.index.to_numpy()
            yt = y_true[idx]
            yp = y_pred[idx]
            all_mae = []
            all_mape = []
            for col in range(6):
                mask = ~np.isnan(yt[:, col])
                if mask.sum() == 0:
                    continue
                mae = float(np.mean(np.abs(yt[mask, col] - yp[mask, col])))
                with np.errstate(divide="ignore", invalid="ignore"):
                    mape = float(np.mean(np.abs((yt[mask, col] - yp[mask, col]) / np.where(yt[mask, col] == 0, np.nan, yt[mask, col]))) * 100.0)
                all_mae.append(mae)
                all_mape.append(mape if np.isfinite(mape) else float("nan"))
            per_regime[str(val)] = {
                "n": int(len(idx)),
                "mean_mae": round(float(np.nanmean(all_mae)), 4) if all_mae else None,
                "mean_mape": round(float(np.nanmean(all_mape)), 4) if all_mape else None,
            }
        regimes[name] = per_regime
    return regimes


def plot_error_by_regime(
    regimes: Dict[str, Any],
    out_path: Path,
) -> Path:
    """Generate Fig 4.8e — MAE by regime."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = [
        ("hour_of_day", "MAE by Hour of Day"),
        ("day_of_week", "MAE by Day of Week"),
        ("pm_level", "MAE by PM2.5 Level"),
    ]
    for ax, (key, title) in zip(axes, titles):
        data = regimes.get(key, {})
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        labels = list(data.keys())
        values = [data[k]["mean_mae"] for k in labels]
        ax.bar(labels, values, color="steelblue", edgecolor="white")
        ax.set_xlabel("Regime")
        ax.set_ylabel("Mean MAE (µg/m³)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Fig 4.8e — Error Analysis by Regime")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_bilstm_vs_arima(
    y_true: np.ndarray,
    y_pred_bilstm: np.ndarray,
    y_pred_arima: np.ndarray,
    out_path: Path,
    horizon_idx: int = 0,
) -> Path:
    """Generate Fig 4.5b — actual vs Bi-LSTM vs ARIMA for one target."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    actual = y_true[:, horizon_idx]
    pred_bilstm = y_pred_bilstm[:, horizon_idx]
    pred_arima = y_pred_arima[:, horizon_idx]

    plt.figure(figsize=(10, 4))
    n_show = min(len(actual), 500)
    x = np.arange(n_show)
    plt.plot(x, actual[:n_show], label="Actual", linewidth=1.5)
    plt.plot(x, pred_bilstm[:n_show], label="Bi-LSTM", linewidth=1.5)
    plt.plot(x, pred_arima[:n_show], label="ARIMA(2,1,2)", linewidth=1.5)
    plt.xlabel("Sample index")
    plt.ylabel("Concentration (µg/m³)")
    pollutant, horizon = TARGET_LABELS[horizon_idx]
    plt.title(f"Fig 4.5b — Bi-LSTM vs ARIMA ({pollutant} {horizon})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------------
def run_evaluation(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the trained Bi-LSTM and ARIMA baseline on the test set."""
    from src import sequence_builder as sb

    results_dir = Path(cfg["paths"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = get_project_root() / results_dir
    tables_dir = results_dir / "tables"
    viz_dir = results_dir / "visualizations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences and masks (keep X on disk as a memmap)
    X, y, timestamps, train_mask, val_mask, test_mask = sb.load_sequences(results_dir)
    test_indices = np.where(test_mask)[0]
    train_indices = np.where(train_mask)[0]
    y_test = y[test_indices]
    y_train = y[train_indices]

    # Load trained PyTorch model using the exact config it was trained with
    model_dir = Path(cfg["paths"]["model_dir"])
    if not model_dir.is_absolute():
        model_dir = get_project_root() / model_dir
    model_path = model_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    model, model_cfg = load_model_with_config(
        model_dir, (X.shape[1], X.shape[2]), fallback_cfg=cfg, device=DEVICE
    )

    # Load target scaler so metrics are reported in original µg/m³
    target_scaler = load_target_scaler(Path(cfg["paths"]["model_dir"]))
    unit_note = "original µg/m³" if not target_scaler.scaled else "µg/m³ after inverse transform"

    # Bi-LSTM predictions (batched by index to avoid materializing X_test)
    batch_size = cfg["model"]["batch_size"]
    y_pred_bilstm = _predict_in_batches(model, X, batch_size, indices=test_indices)
    bilstm_inv = target_scaler.inverse_transform(y_test)
    bilstm_pred_inv = target_scaler.inverse_transform(y_pred_bilstm)
    bilstm_metrics = compute_metrics(bilstm_inv, bilstm_pred_inv)

    # ARIMA baseline
    y_train_orig = target_scaler.inverse_transform(y_train)
    y_test_orig = target_scaler.inverse_transform(y_test)
    y_pred_arima, arima_metrics = arima_baseline(y_train_orig, y_test_orig)

    # Persistence baseline
    y_pred_persistence, persistence_metrics = persistence_baseline(y_train_orig, y_test_orig)

    # Export Table 4.8
    table_4_8 = {
        "model": "Bi-LSTM (PyTorch)",
        "units": unit_note,
        "metrics": bilstm_metrics,
    }
    with open(tables_dir / "table_4_8_bilstm.json", "w", encoding="utf-8") as f:
        json.dump(table_4_8, f, indent=2)

    # Export ARIMA results
    arima_results = {
        "model": "ARIMA(2,1,2)",
        "order": [2, 1, 2],
        "units": unit_note,
        "metrics": arima_metrics,
    }
    with open(tables_dir / "arima_results.json", "w", encoding="utf-8") as f:
        json.dump(arima_results, f, indent=2)

    # Export persistence baseline
    persistence_results = {
        "model": "Persistence",
        "units": unit_note,
        "metrics": persistence_metrics,
    }
    with open(tables_dir / "persistence_results.json", "w", encoding="utf-8") as f:
        json.dump(persistence_results, f, indent=2)

    # Training convergence figure
    training_results_path = Path(cfg["paths"]["model_dir"]) / "training_results.json"
    if not training_results_path.is_absolute():
        training_results_path = get_project_root() / training_results_path
    if training_results_path.exists():
        with open(training_results_path, "r", encoding="utf-8") as f:
            training_results = json.load(f)
        plot_training_convergence(training_results["history"], viz_dir / "fig_4_5a_training_convergence.png")

    # Bi-LSTM vs ARIMA figure
    plot_bilstm_vs_arima(
        bilstm_inv, bilstm_pred_inv, y_pred_arima,
        viz_dir / "fig_4_5b_bilstm_vs_arima.png",
        horizon_idx=0,
    )

    # Error by regime (Table 4.8c / Fig 4.8e)
    regimes = compute_error_by_regime(bilstm_inv, bilstm_pred_inv, timestamps[test_mask])
    with open(tables_dir / "table_4_8c_error_by_regime.json", "w", encoding="utf-8") as f:
        json.dump(regimes, f, indent=2)
    plot_error_by_regime(regimes, viz_dir / "fig_4_8e_error_by_regime.png")

    return {
        "bilstm_metrics": bilstm_metrics,
        "arima_metrics": arima_metrics,
        "regimes": regimes,
        "tables_dir": str(tables_dir),
        "viz_dir": str(viz_dir),
    }


__all__ = [
    "compute_metrics",
    "arima_baseline",
    "plot_training_convergence",
    "plot_bilstm_vs_arima",
    "run_evaluation",
]
