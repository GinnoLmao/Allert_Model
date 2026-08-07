"""ALLERT dual-mode anomaly detection pipeline.

Implements the three integrated detectors from Chapter 3.2:
    1. Rolling Z-score residual monitor
    2. Velocity override
    3. CUSUM drift detector

All thresholds come from ``config.yaml``. The module also builds contingency
tables against a synthetic ground-truth label and exports the artifacts
required for Chapter 4.

Exported artifacts:
    - ``results/tables/table_4_9_anomaly.json``
    - ``results/tables/table_4_10_cusum.json``
    - ``results/visualizations/fig_4_6_residual_distribution.png``

Thesis traceability:
    - Chapter 3.2, "Integrated Anomaly Detection"
    - Chapter 4, Table 4.9, Table 4.10, Fig. 4.6
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config_loader import get_project_root
from src import anomaly_calibration as acal


def _records_per_hour_from_timestamps(timestamps: pd.DatetimeIndex) -> int:
    median_delta = pd.Series(timestamps).diff().median()
    if pd.isna(median_delta) or median_delta == pd.Timedelta(0):
        return 12  # default 5-minute data
    return max(1, int(round(pd.Timedelta(hours=1) / median_delta)))


# ---------------------------------------------------------------------------
# Z-score residual monitor
# ---------------------------------------------------------------------------
def compute_z_score_residuals(
    actual: np.ndarray,
    forecast: np.ndarray,
    rolling_window: int,
) -> Tuple[np.ndarray, float, float]:
    """Compute standardized residuals using a rolling 24-hour window."""
    residuals = np.abs(actual - forecast)
    s = pd.Series(residuals)
    roll_mu = s.rolling(window=rolling_window, min_periods=1).mean().to_numpy()
    roll_sigma = s.rolling(window=rolling_window, min_periods=1).std().fillna(0).to_numpy()
    roll_sigma = np.where(roll_sigma == 0, 1e-9, roll_sigma)
    z_scores = (residuals - roll_mu) / roll_sigma
    return z_scores, float(np.mean(residuals)), float(np.std(residuals))


def classify_z_tiers(
    z_scores: np.ndarray, thresholds: List[float]
) -> Dict[str, np.ndarray]:
    """Return boolean masks for each Z-score tier."""
    return {
        "Spike_Watch": (z_scores >= thresholds[0]) & (z_scores < thresholds[1]),
        "Spike_Warning": (z_scores >= thresholds[1]) & (z_scores < thresholds[2]),
        "Spike_Emergency": z_scores >= thresholds[2],
    }


# ---------------------------------------------------------------------------
# Velocity override
# ---------------------------------------------------------------------------
def velocity_override(
    actual: np.ndarray,
    step_minutes: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute hourly PM velocity and flag emergency overrides."""
    dt_hr = step_minutes / 60.0
    velocity = np.zeros_like(actual)
    velocity[1:] = (actual[1:] - actual[:-1]) / dt_hr
    trigger = velocity > threshold
    return velocity, trigger


# ---------------------------------------------------------------------------
# CUSUM drift detector
# ---------------------------------------------------------------------------
def cusum_detector(
    residuals: np.ndarray,
    k_multiplier: float,
    h_multiplier: float,
) -> Tuple[np.ndarray, float, float]:
    """CUSUM control chart with reset-after-alert policy."""
    sigma = float(np.std(residuals)) if np.std(residuals) > 0 else 1.0
    k = k_multiplier * sigma
    h = h_multiplier * sigma

    S = 0.0
    alerts = np.zeros(len(residuals), dtype=bool)
    for i, e in enumerate(residuals):
        S = max(0.0, S + (e - k))
        if S > h:
            alerts[i] = True
            S = 0.0
    return alerts, k, h


# ---------------------------------------------------------------------------
# Ground truth labeling and contingency
# ---------------------------------------------------------------------------
def label_ground_truth(
    actual: np.ndarray,
    velocity: np.ndarray,
    pm_threshold: float,
    velocity_threshold: float,
) -> np.ndarray:
    """Label anomalous timesteps for evaluation."""
    return (actual > pm_threshold) | (velocity > velocity_threshold)


def contingency_table(pred_mask: np.ndarray, truth_mask: np.ndarray) -> Dict[str, Any]:
    """Return TP/FP/TN/FN and derived classification metrics."""
    tp = int(np.sum(pred_mask & truth_mask))
    fp = int(np.sum(pred_mask & ~truth_mask))
    tn = int(np.sum(~pred_mask & ~truth_mask))
    fn = int(np.sum(~pred_mask & truth_mask))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Sensitivity_pct": round(sensitivity * 100, 2),
        "Specificity_pct": round(specificity * 100, 2),
        "FalsePositiveRate_pct": round(fpr * 100, 2),
        "Precision_pct": round(precision * 100, 2),
        "F1_Score": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# CUSUM lag analysis (Table 4.10)
# ---------------------------------------------------------------------------
def _find_event_boundaries(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return (start, end) indices of contiguous True regions."""
    boundaries = []
    in_event = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_event:
            in_event = True
            start = i
        elif not val and in_event:
            in_event = False
            boundaries.append((start, i - 1))
    if in_event:
        boundaries.append((start, len(mask) - 1))
    return boundaries


def cusum_lag_analysis(
    timestamps: np.ndarray,
    ground_truth: np.ndarray,
    z_detect: np.ndarray,
    cusum_alert: np.ndarray,
    step_minutes: int,
) -> List[Dict[str, Any]]:
    """Build CUSUM detection-lag records from labeled anomaly events."""
    events = _find_event_boundaries(ground_truth)
    scenarios = []
    for idx, (onset, _) in enumerate(events[:20], start=1):  # cap at 20 events
        cusum_detections = np.where(cusum_alert[onset:])[0]
        z_detections = np.where(z_detect[onset:])[0]

        cusum_delay = int(cusum_detections[0] * step_minutes) if len(cusum_detections) > 0 else None
        z_delay = int(z_detections[0] * step_minutes) if len(z_detections) > 0 else None

        scenarios.append({
            "scenario_id": idx,
            "anomaly_type": "gradual_drift" if cusum_delay is not None and (z_delay is None or cusum_delay < z_delay) else "sudden_spike",
            "onset_timestamp": pd.Timestamp(timestamps[onset]).isoformat(),
            "cusum_detection_timestamp": pd.Timestamp(timestamps[onset + cusum_detections[0]]).isoformat() if len(cusum_detections) > 0 else None,
            "z_score_detection_timestamp": pd.Timestamp(timestamps[onset + z_detections[0]]).isoformat() if len(z_detections) > 0 else None,
            "cusum_detection_delay_minutes": cusum_delay,
            "z_score_detection_delay_minutes": z_delay,
            "peak_z_score": None,
            "peak_cusum_S": None,
            "correctly_detected": len(cusum_detections) > 0,
        })
    return scenarios


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_cusum_decision(
    residuals: np.ndarray,
    alerts: np.ndarray,
    k: float,
    h: float,
    out_path: Path,
) -> Path:
    """Generate Fig 4.6b — CUSUM decision function with k/h lines."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    S = 0.0
    S_values = []
    for e in residuals:
        S = max(0.0, S + (e - k))
        S_values.append(S)

    plt.figure(figsize=(12, 4))
    x = np.arange(len(residuals))
    plt.plot(x, S_values, color="steelblue", linewidth=1.2, label="CUSUM S_t")
    plt.axhline(h, color="red", linestyle="--", linewidth=2, label=f"threshold h = {h:.2f}")
    plt.axhline(k, color="orange", linestyle="--", linewidth=1.5, label=f"k = {k:.2f}")
    alert_x = x[alerts]
    alert_y = np.array(S_values)[alerts]
    plt.scatter(alert_x, alert_y, color="red", s=20, zorder=5, label="alerts")
    plt.xlabel("Sample index")
    plt.ylabel("CUSUM statistic")
    plt.title("Fig 4.6b — CUSUM Decision Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_confusion_matrix(
    pred_mask: np.ndarray,
    truth_mask: np.ndarray,
    out_path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Generate a confusion-matrix heatmap."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tp = int(np.sum(pred_mask & truth_mask))
    fp = int(np.sum(pred_mask & ~truth_mask))
    tn = int(np.sum(~pred_mask & ~truth_mask))
    fn = int(np.sum(~pred_mask & truth_mask))
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Negative", "Predicted Positive"])
    ax.set_yticklabels(["Actual Negative", "Actual Positive"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_residual_distribution(
    residuals: np.ndarray,
    out_path: Path,
    n_lags: int = 40,
) -> Path:
    """Generate Fig 4.6 — residual histogram and autocorrelation."""
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import acf

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(residuals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Absolute residual (µg/m³)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("(a) Residual Distribution")
    axes[0].grid(True, alpha=0.3)

    # ACF
    try:
        acf_vals = acf(residuals, nlags=n_lags, fft=True, missing="drop")
        axes[1].bar(range(len(acf_vals)), acf_vals, color="coral", edgecolor="white")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Lag")
        axes[1].set_ylabel("Autocorrelation")
        axes[1].set_title("(b) Residual ACF")
        axes[1].grid(True, alpha=0.3)
    except Exception:
        axes[1].text(0.5, 0.5, "ACF computation failed", ha="center", va="center", transform=axes[1].transAxes)

    fig.suptitle("Fig 4.6 — Residual Distribution and Autocorrelation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
def run_anomaly_pipeline(
    cfg: Dict[str, Any],
    actual: np.ndarray,
    forecast: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    out_dir: Optional[Path] = None,
    pollutant: str = "pm2_5",
) -> Dict[str, Any]:
    """Run the full anomaly detection pipeline and export artifacts."""
    if out_dir is None:
        out_dir = Path(cfg["paths"]["results_dir"])
        if not out_dir.is_absolute():
            out_dir = get_project_root() / out_dir
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    viz_dir = out_dir / "visualizations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    acfg = cfg["anomaly"]
    step_minutes = cfg["sampling"]["transmission_interval_minutes"]
    rph = _records_per_hour_from_timestamps(pd.to_datetime(timestamps) if timestamps is not None else pd.date_range("2026-01-01", periods=len(actual), freq="5T"))
    rolling_window = acfg["residual_rolling_window_hours"] * rph

    z_thresholds = acal.get_z_thresholds(acfg, pollutant)
    velocity_threshold = acal.get_velocity_threshold(acfg, pollutant)
    event_threshold = acal.get_event_threshold(acfg, pollutant)

    residuals = np.abs(actual - forecast)
    z_scores, mu, sigma = compute_z_score_residuals(actual, forecast, rolling_window)
    z_tiers = classify_z_tiers(z_scores, z_thresholds)
    z_detect = z_scores >= z_thresholds[1]  # warning or emergency

    velocity, vel_trigger = velocity_override(
        actual, step_minutes, velocity_threshold
    )

    # Optional calibration on validation residuals (not test set)
    cal_result = acal.calibrate_anomaly_thresholds(
        cfg, actual, forecast, pollutant=pollutant, out_dir=tables_dir
    )
    k_multiplier = cal_result.get("chosen", {}).get("k_multiplier", acfg["cusum_k_multiplier"])
    h_multiplier = cal_result.get("chosen", {}).get("h_multiplier", acfg["cusum_h_multiplier"])

    cusum_alert, k, h = cusum_detector(
        residuals,
        k_multiplier,
        h_multiplier,
    )

    gt = label_ground_truth(
        actual,
        velocity,
        event_threshold,
        velocity_threshold,
    )

    results = {
        "pollutant": pollutant,
        "z_score": {
            "thresholds": z_thresholds,
            "global_mu": round(mu, 6),
            "global_sigma": round(sigma, 6),
            "contingency": contingency_table(z_detect, gt),
            "tier_counts": {k: int(np.sum(v)) for k, v in z_tiers.items()},
        },
        "velocity_override": {
            "threshold_ugm3_per_hour": velocity_threshold,
            "contingency": contingency_table(vel_trigger, gt),
            "trigger_count": int(np.sum(vel_trigger)),
        },
        "cusum": {
            "k": round(k, 6),
            "h": round(h, 6),
            "contingency": contingency_table(cusum_alert, gt),
            "alert_count": int(np.sum(cusum_alert)),
            "calibration": cal_result,
        },
        "ground_truth": {
            "event_threshold_ugm3": event_threshold,
            "velocity_threshold_ugm3_per_hour": velocity_threshold,
            "anomaly_count": int(np.sum(gt)),
            "total_samples": len(actual),
        },
    }

    # Table 4.9 (per-pollutant and primary copy)
    with open(tables_dir / f"table_4_9_anomaly_{pollutant}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if pollutant == "pm2_5":
        with open(tables_dir / "table_4_9_anomaly.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # Table 4.10 (per-pollutant and primary copy)
    scenarios = cusum_lag_analysis(
        timestamps if timestamps is not None else np.arange(len(actual)),
        gt,
        z_detect,
        cusum_alert,
        step_minutes,
    )
    with open(tables_dir / f"table_4_10_cusum_{pollutant}.json", "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)
    if pollutant == "pm2_5":
        with open(tables_dir / "table_4_10_cusum.json", "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2)

    # Fig 4.6
    plot_residual_distribution(residuals, viz_dir / "fig_4_6_residual_distribution.png")

    # Fig 4.6b
    plot_cusum_decision(residuals, cusum_alert, k, h, viz_dir / "fig_4_6b_cusum_decision.png")

    # Fig 4.6c
    plot_confusion_matrix(
        z_detect, gt, viz_dir / "fig_4_6c_confusion_matrix_z.png",
        title="Z-score Detector Confusion Matrix",
    )
    plot_confusion_matrix(
        cusum_alert, gt, viz_dir / "fig_4_6c_confusion_matrix_cusum.png",
        title="CUSUM Confusion Matrix",
    )

    return results


__all__ = [
    "compute_z_score_residuals",
    "velocity_override",
    "cusum_detector",
    "label_ground_truth",
    "contingency_table",
    "run_anomaly_pipeline",
]
