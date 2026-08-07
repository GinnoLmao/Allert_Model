"""ALLERT anomaly detection calibration.

Provides validation-based grid search for CUSUM ``k`` and ``h`` parameters
and per-pollutant threshold helpers. The calibration uses train/validation
residuals and a synthetic ground-truth label, avoiding the test set.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import anomaly_detection as ad


TARGET_POLLUTANTS = ["pm2_5", "pm10"]


def _get_thresholds(
    acfg: Dict[str, Any],
    pollutant: str,
    threshold_type: str,
) -> Any:
    """Return per-pollutant thresholds with global fallback.

    Parameters
    ----------
    acfg:
        ``cfg["anomaly"]``.
    pollutant:
        ``pm2_5`` or ``pm10``.
    threshold_type:
        ``z_thresholds`` or ``velocity_threshold_ugm3_per_hour``.
    """
    # For velocity thresholds, prefer short key (e.g., velocity_threshold_pm2_5)
    if threshold_type == "velocity_threshold_ugm3_per_hour":
        per_key = f"velocity_threshold_{pollutant}"
    else:
        per_key = f"{threshold_type}_{pollutant}"
    if per_key in acfg:
        return acfg[per_key]
    return acfg[threshold_type]


def get_z_thresholds(acfg: Dict[str, Any], pollutant: str = "pm2_5") -> List[float]:
    """Return Z-score thresholds for a pollutant."""
    return list(_get_thresholds(acfg, pollutant, "z_thresholds"))


def get_velocity_threshold(acfg: Dict[str, Any], pollutant: str = "pm2_5") -> float:
    """Return velocity threshold for a pollutant."""
    return float(_get_thresholds(acfg, pollutant, "velocity_threshold_ugm3_per_hour"))


def get_event_threshold(acfg: Dict[str, Any], pollutant: str = "pm2_5") -> float:
    """Return concentration event threshold for a pollutant."""
    key = f"{pollutant}_event_threshold_ugm3"
    if key in acfg:
        return float(acfg[key])
    # Fallback to PM2.5 threshold
    return float(acfg["pm25_event_threshold_ugm3"])


def grid_search_cusum(
    actual: np.ndarray,
    forecast: np.ndarray,
    k_multipliers: np.ndarray,
    h_multipliers: np.ndarray,
    max_fpr_pct: float = 10.0,
    step_minutes: int = 5,
    event_threshold: float = 15.0,
    velocity_threshold: float = 15.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Grid search CUSUM k/h on a validation residual set.

    Parameters
    ----------
    actual, forecast:
        Validation actuals and model forecasts in original units.
    k_multipliers, h_multipliers:
        Multipliers of residual sigma to sweep.
    max_fpr_pct:
        Maximum acceptable false-positive rate.
    step_minutes:
        Sampling interval.
    event_threshold, velocity_threshold:
        Ground-truth labeling thresholds.

    Returns
    -------
    tuple
        (best parameters dict, all results list).
    """
    residuals = np.abs(actual - forecast)
    velocity, _ = ad.velocity_override(actual, step_minutes, velocity_threshold)
    truth = ad.label_ground_truth(actual, velocity, event_threshold, velocity_threshold)

    best = None
    all_results = []
    for km in k_multipliers:
        for hm in h_multipliers:
            alerts, k, h = ad.cusum_detector(residuals, float(km), float(hm))
            metrics = ad.contingency_table(alerts, truth)
            entry = {
                "k_multiplier": float(km),
                "h_multiplier": float(hm),
                "k": round(k, 6),
                "h": round(h, 6),
                "metrics": metrics,
            }
            all_results.append(entry)
            if metrics["FalsePositiveRate_pct"] > max_fpr_pct:
                continue
            if best is None or metrics["F1_Score"] > best["metrics"]["F1_Score"]:
                best = entry

    if best is None:
        # If nothing satisfies FPR constraint, pick the one with lowest FPR
        best = min(all_results, key=lambda x: x["metrics"]["FalsePositiveRate_pct"])

    return best, all_results


def calibrate_anomaly_thresholds(
    cfg: Dict[str, Any],
    actual: np.ndarray,
    forecast: np.ndarray,
    pollutant: str = "pm2_5",
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Calibrate CUSUM thresholds for a single pollutant and export results.

    Uses the calibration grid defined in ``cfg["anomaly"]["calibration"]``.
    """
    acfg = cfg["anomaly"]
    cal = acfg.get("calibration", {})
    if not cal.get("enabled", False):
        return {
            "calibration_enabled": False,
            "k_multiplier": acfg["cusum_k_multiplier"],
            "h_multiplier": acfg["cusum_h_multiplier"],
        }

    k_multipliers = np.asarray(cal.get("k_multipliers", [0.5]))
    h_multipliers = np.asarray(cal.get("h_multipliers", [4.0]))
    max_fpr_pct = float(cal.get("max_fpr_pct", 10.0))
    step_minutes = cfg["sampling"]["transmission_interval_minutes"]
    event_threshold = get_event_threshold(acfg, pollutant)
    velocity_threshold = get_velocity_threshold(acfg, pollutant)

    best, all_results = grid_search_cusum(
        actual, forecast, k_multipliers, h_multipliers,
        max_fpr_pct=max_fpr_pct,
        step_minutes=step_minutes,
        event_threshold=event_threshold,
        velocity_threshold=velocity_threshold,
    )

    result = {
        "pollutant": pollutant,
        "calibration_enabled": True,
        "max_fpr_pct": max_fpr_pct,
        "chosen": best,
        "all_results": all_results,
        "rationale": (
            f"Selected k_multiplier={best['k_multiplier']}, h_multiplier={best['h_multiplier']} "
            f"maximizing F1 while keeping FPR <= {max_fpr_pct}% on validation residuals."
        ),
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"table_4_9b_anomaly_calibration_{pollutant}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


__all__ = [
    "get_z_thresholds",
    "get_velocity_threshold",
    "get_event_threshold",
    "grid_search_cusum",
    "calibrate_anomaly_thresholds",
]
