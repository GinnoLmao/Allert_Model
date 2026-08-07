"""ALLERT risk engine sensitivity analysis.

Perturbs the epidemiological coefficients by a configurable percentage and
reports the resulting change in absolute risk probability and tier distribution.
"""
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config_loader import get_project_root
from src.risk_engine import (
    add_p_risk_columns,
    add_risk_tier_columns,
    classify_risk_tier,
    compute_p_risk,
)


SENSITIVITY_COEFFICIENTS = [
    "pm2_5_beta",
    "pm10_beta",
    "temp_beta",
    "humidity_beta",
    "baseline_prevalence_adult",
    "baseline_prevalence_pediatric",
]


def _perturb_config(cfg: Dict[str, Any], coefficient: str, pct: float) -> Dict[str, Any]:
    """Return a config copy with ``coefficient`` changed by ``pct``."""
    new_cfg = deepcopy(cfg)
    base = new_cfg["risk"][coefficient]
    new_cfg["risk"][coefficient] = base * (1.0 + pct)
    return new_cfg


def _tier_distribution(tiers: pd.Series, cfg: Dict[str, Any], age_group: str) -> Dict[str, int]:
    """Count risk-tier occurrences."""
    tier_defs = cfg["risk"][f"{age_group}_tiers"]
    sorted_tiers = sorted(tier_defs.items(), key=lambda kv: kv[1])
    counts = {}
    total = len(tiers)
    for i, (name, lo) in enumerate(sorted_tiers):
        hi = sorted_tiers[i + 1][1] if i + 1 < len(sorted_tiers) else 10.0
        mask = (tiers >= lo) & (tiers < hi)
        counts[name] = {
            "count": int(mask.sum()),
            "pct": round(100.0 * int(mask.sum()) / total, 2) if total else 0.0,
        }
    return counts


def analyze_risk_sensitivity(
    cfg: Dict[str, Any],
    df: pd.DataFrame,
    coefficients: Optional[List[str]] = None,
    perturbation_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Perturb risk coefficients and compare P_risk and tier changes.

    Parameters
    ----------
    cfg:
        Loaded configuration.
    df:
        Preprocessed DataFrame with ``pm2_5``, ``pm10``, ``temperature``, ``humidity``.
    coefficients:
        Coefficients to perturb. Defaults to ``SENSITIVITY_COEFFICIENTS``.
    perturbation_pct:
        Symmetric perturbation fraction (e.g., 0.10 for ±10%). Defaults to
        ``cfg["risk"]["sensitivity_perturbation_pct"]``.

    Returns
    -------
    dict
        Sensitivity report with per-coefficient results.
    """
    coefficients = coefficients or SENSITIVITY_COEFFICIENTS
    pct = perturbation_pct if perturbation_pct is not None else cfg["risk"].get("sensitivity_perturbation_pct", 0.10)

    baseline_df = add_p_risk_columns(df, cfg)
    baseline_df = add_risk_tier_columns(baseline_df, cfg)

    results = {
        "baseline": {
            "mean_p_risk_adult": round(float(baseline_df["p_risk_adult"].mean()), 6),
            "mean_p_risk_pediatric": round(float(baseline_df["p_risk_pediatric"].mean()), 6),
            "adult_tiers": _tier_distribution(baseline_df["p_risk_adult"], cfg, "adult"),
            "pediatric_tiers": _tier_distribution(baseline_df["p_risk_pediatric"], cfg, "pediatric"),
        },
        "perturbation_pct": round(pct * 100, 2),
        "coefficients": {},
    }

    for coef in coefficients:
        per_coef = {}
        for sign, label in [(+pct, "plus"), (-pct, "minus")]:
            perturbed_cfg = _perturb_config(cfg, coef, sign)
            perturbed_df = add_p_risk_columns(df, perturbed_cfg)
            perturbed_df = add_risk_tier_columns(perturbed_df, perturbed_cfg)

            delta_adult = (perturbed_df["p_risk_adult"] - baseline_df["p_risk_adult"]) * 100.0
            delta_ped = (perturbed_df["p_risk_pediatric"] - baseline_df["p_risk_pediatric"]) * 100.0

            per_coef[label] = {
                "coefficient_value": round(perturbed_cfg["risk"][coef], 6),
                "mean_p_risk_adult": round(float(perturbed_df["p_risk_adult"].mean()), 6),
                "mean_p_risk_pediatric": round(float(perturbed_df["p_risk_pediatric"].mean()), 6),
                "delta_p_risk_adult_mean_pct": round(float(delta_adult.mean()), 4),
                "delta_p_risk_pediatric_mean_pct": round(float(delta_ped.mean()), 4),
                "adult_tiers": _tier_distribution(perturbed_df["p_risk_adult"], perturbed_cfg, "adult"),
                "pediatric_tiers": _tier_distribution(perturbed_df["p_risk_pediatric"], perturbed_cfg, "pediatric"),
            }
        results["coefficients"][coef] = per_coef

    return results


def plot_risk_sensitivity(
    report: Dict[str, Any],
    out_path: Path,
) -> Path:
    """Generate a tornado-style bar chart of coefficient impacts on adult P_risk."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coefs = list(report["coefficients"].keys())
    deltas_plus = [report["coefficients"][c]["plus"]["delta_p_risk_adult_mean_pct"] for c in coefs]
    deltas_minus = [report["coefficients"][c]["minus"]["delta_p_risk_adult_mean_pct"] for c in coefs]

    y = np.arange(len(coefs))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(coefs) + 2)))
    ax.barh(y, deltas_plus, color="firebrick", label=f"+{report['perturbation_pct']}%")
    ax.barh(y, deltas_minus, color="steelblue", label=f"-{report['perturbation_pct']}%")
    ax.set_yticks(y)
    ax.set_yticklabels(coefs)
    ax.set_xlabel("Change in mean adult P_risk (percentage points)")
    ax.set_title("Fig 4.12b — Risk Engine Sensitivity")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def run_risk_sensitivity(
    cfg: Dict[str, Any],
    out_dir: Optional[Path] = None,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run the full sensitivity analysis and export artifacts."""
    if out_dir is None:
        out_dir = Path(cfg["paths"]["results_dir"])
        if not out_dir.is_absolute():
            out_dir = get_project_root() / out_dir
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    viz_dir = out_dir / "visualizations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        preprocessed_path = get_project_root() / "data" / "processed" / "preprocessed_5min.csv"
        df = pd.read_csv(preprocessed_path, parse_dates=["timestamp"], index_col="timestamp")

    report = analyze_risk_sensitivity(cfg, df)
    with open(tables_dir / "table_4_12b_risk_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    plot_risk_sensitivity(report, viz_dir / "fig_4_12b_risk_sensitivity.png")

    return report


__all__ = [
    "SENSITIVITY_COEFFICIENTS",
    "analyze_risk_sensitivity",
    "plot_risk_sensitivity",
    "run_risk_sensitivity",
]
