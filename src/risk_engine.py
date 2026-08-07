"""ALLERT epidemiological risk scoring engine.

Converts PM concentrations into an age-stratified allergic rhinitis (AR)
risk probability using:
    - Baseline PM risk: Risk_PM = PM2.5*β25 + PM10*β10
    - Temperature and humidity modifiers
    - Odds-ratio → probability conversion with age-specific baseline prevalence
    - 72-hour continuous rolling distributed lag
    - Four-tier risk stratification

All coefficients and thresholds are read from ``config.yaml``.

Exported artifacts:
    - ``results/tables/table_4_11_risk_distribution.json``
    - ``results/tables/table_4_12_modifier_analysis.json``
    - ``results/visualizations/fig_4_7_risk_vs_pm25.png``
    - ``results/visualizations/fig_4_11_risk_distribution.png``

Thesis traceability:
    - Chapter 3.2, "Environmental Risk Estimation Engine"
    - Chapter 4, Table 4.11, Table 4.12, Fig. 4.7, Fig. 4.11
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config_loader import get_project_root


def _validate_lag_weights(cfg: Dict[str, Any]) -> None:
    weights = cfg["risk"]["distributed_lag_weights"]
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError(f"Distributed lag weights must sum to 1.0, got {sum(weights)}")


def _records_per_hour(series: pd.Series) -> int:
    median_delta = series.index.to_series().diff().median()
    if pd.isna(median_delta) or median_delta == pd.Timedelta(0):
        return 12
    return max(1, int(round(pd.Timedelta(hours=1) / median_delta)))


# ---------------------------------------------------------------------------
# Core risk calculations
# ---------------------------------------------------------------------------
def compute_pm_risk(pm25: float, pm10: float, cfg: Dict[str, Any]) -> float:
    """Compute log-odds PM risk contribution."""
    return pm25 * cfg["risk"]["pm2_5_beta"] + pm10 * cfg["risk"]["pm10_beta"]


def temperature_modifier(temp_c: float, cfg: Dict[str, Any]) -> float:
    """Linear temperature modifier."""
    return (temp_c - cfg["risk"]["temp_baseline_c"]) * cfg["risk"]["temp_beta"]


def humidity_modifier(rh: float, cfg: Dict[str, Any]) -> float:
    """Linear humidity modifier."""
    return (rh - cfg["risk"]["humidity_baseline_pct"]) * cfg["risk"]["humidity_beta"]


def combined_odds_ratio(risk_pm: float, m_temp: float, m_rh: float) -> float:
    """OR_total = exp(Risk_PM + M_temp + M_humidity)."""
    return float(np.exp(risk_pm + m_temp + m_rh))


def p_risk_from_or(or_total: float, age_group: str, cfg: Dict[str, Any]) -> float:
    """Convert combined OR to absolute AR probability in [0, 1]."""
    p_baseline = cfg["risk"][f"baseline_prevalence_{age_group}"]
    numerator = p_baseline * or_total
    denominator = 1.0 + p_baseline * (or_total - 1.0)
    return float(numerator / denominator)


def compute_p_risk(
    pm25: float, pm10: float, temp_c: float, rh: float, age_group: str, cfg: Dict[str, Any]
) -> float:
    """End-to-end single-timestep P_risk."""
    rp = compute_pm_risk(pm25, pm10, cfg)
    mt = temperature_modifier(temp_c, cfg)
    mr = humidity_modifier(rh, cfg)
    or_total = combined_odds_ratio(rp, mt, mr)
    return p_risk_from_or(or_total, age_group, cfg)


# ---------------------------------------------------------------------------
# Distributed lag
# ---------------------------------------------------------------------------
def rolling_72h_continuous_risk(p_risk_series: pd.Series, cfg: Dict[str, Any]) -> pd.Series:
    """Compute continuous rolling 72-hour weighted average of P_risk.

    The weight vector is built from ``distributed_lag_weights`` and
    ``distributed_lag_hours`` in config. The final weight sum is asserted
    to be 1.0.
    """
    _validate_lag_weights(cfg)

    weights = cfg["risk"]["distributed_lag_weights"]
    hours = cfg["risk"]["distributed_lag_hours"]
    rph = _records_per_hour(p_risk_series)

    # Build a weight vector covering 0..72h inclusive
    total_steps = hours[-1] * rph + 1
    w = np.zeros(total_steps, dtype=float)

    for weight, h in zip(weights, hours):
        if h == 0:
            start = 0
            end = 1
        else:
            start = (hours[hours.index(h) - 1] * rph + 1) if hours.index(h) > 0 else 0
            end = h * rph + 1
        block_size = end - start
        if block_size > 0:
            w[start:end] = weight / block_size

    if abs(w.sum() - 1.0) > 1e-12:
        w = w / w.sum()

    arr = p_risk_series.to_numpy(dtype=float)
    pad = np.zeros(len(w) - 1, dtype=float)
    arr_padded = np.concatenate([pad, arr])
    result = np.convolve(arr_padded, w[::-1], mode="valid")
    return pd.Series(result, index=p_risk_series.index)


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------
def classify_risk_tier(p_risk: float, age_group: str, cfg: Dict[str, Any]) -> Tuple[str, float]:
    """Return (tier_name, lower_threshold) for a P_risk value in [0, 1]."""
    p = float(p_risk)
    tiers = cfg["risk"][f"{age_group}_tiers"]
    # tiers dict values are fractions, e.g. 0.15 = 15%
    sorted_tiers = sorted(tiers.items(), key=lambda kv: kv[1])
    for i, (name, lo) in enumerate(sorted_tiers):
        hi = sorted_tiers[i + 1][1] if i + 1 < len(sorted_tiers) else 10.0
        if lo <= p < hi:
            return name, float(lo)
    return sorted_tiers[-1][0], float(sorted_tiers[-1][1])


# ---------------------------------------------------------------------------
# DataFrame-level helpers
# ---------------------------------------------------------------------------
def add_p_risk_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add adult and pediatric P_risk columns to a DataFrame."""
    df = df.copy()
    for age in ["adult", "pediatric"]:
        df[f"p_risk_{age}"] = df.apply(
            lambda r: compute_p_risk(
                r["pm2_5"], r["pm10"], r["temperature"], r["humidity"], age, cfg
            ),
            axis=1,
        )
    return df


def add_risk_tier_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add adult and pediatric risk-tier columns."""
    df = df.copy()
    for age in ["adult", "pediatric"]:
        df[f"risk_tier_{age}"] = df[f"p_risk_{age}"].apply(
            lambda v: classify_risk_tier(v, age, cfg)[0]
        )
    return df


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------
def _tier_counts(series: pd.Series, tiers: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    sorted_tiers = sorted(tiers.items(), key=lambda kv: kv[1])
    counts = {}
    total = len(series)
    for i, (name, lo) in enumerate(sorted_tiers):
        hi = sorted_tiers[i + 1][1] if i + 1 < len(sorted_tiers) else 10.0
        mask = (series >= lo) & (series < hi)
        count = int(mask.sum())
        counts[name] = {"count": count, "pct": round(100.0 * count / total, 2) if total else 0}
    return counts


def export_risk_distribution(
    df: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path
) -> Dict[str, Any]:
    """Export Table 4.11 — risk probability distribution by tier."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "total_calculations": int(len(df)),
        "adult": _tier_counts(df["p_risk_adult"], cfg["risk"]["adult_tiers"]),
        "pediatric": _tier_counts(df["p_risk_pediatric"], cfg["risk"]["pediatric_tiers"]),
    }
    with open(out_dir / "table_4_11_risk_distribution.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def export_modifier_analysis(
    df: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path
) -> Dict[str, Any]:
    """Export Table 4.12 — meteorological modifier contribution."""
    out_dir.mkdir(parents=True, exist_ok=True)

    m_temp = df["temperature"].apply(lambda t: temperature_modifier(t, cfg))
    m_rh = df["humidity"].apply(lambda rh: humidity_modifier(rh, cfg))

    pr_full = df.apply(
        lambda r: compute_p_risk(r["pm2_5"], r["pm10"], r["temperature"], r["humidity"], "adult", cfg),
        axis=1,
    )
    pr_pm_only = df.apply(
        lambda r: compute_p_risk(
            r["pm2_5"], r["pm10"], cfg["risk"]["temp_baseline_c"], cfg["risk"]["humidity_baseline_pct"], "adult", cfg
        ),
        axis=1,
    )
    delta = (pr_full - pr_pm_only) * 100.0

    result = {
        "temperature": {
            "observed_range_C": {
                "min": float(df["temperature"].min()),
                "max": float(df["temperature"].max()),
                "mean": float(df["temperature"].mean()),
            },
            "M_temp_range": {
                "min": float(m_temp.min()),
                "max": float(m_temp.max()),
                "mean": float(m_temp.mean()),
            },
            "delta_P_risk_from_temp_alone_pct": {
                "min": float(delta.min()),
                "max": float(delta.max()),
                "mean": float(delta.mean()),
            },
        },
        "humidity": {
            "observed_range_RH": {
                "min": float(df["humidity"].min()),
                "max": float(df["humidity"].max()),
                "mean": float(df["humidity"].mean()),
            },
            "M_humidity_range": {
                "min": float(m_rh.min()),
                "max": float(m_rh.max()),
                "mean": float(m_rh.mean()),
            },
            "delta_P_risk_from_humidity_alone_pct": {
                "min": float(delta.min()),
                "max": float(delta.max()),
                "mean": float(delta.mean()),
            },
        },
        "combined_modifier_impact": {
            "cases_where_temp_increases_risk": int((m_temp > 0).sum()),
            "cases_where_humidity_decreases_risk": int((m_rh < 0).sum()),
            "mean_combined_modifier_delta_pct": round(float(delta.mean()), 4),
            "delta_range_pct": {
                "min": round(float(delta.min()), 4),
                "max": round(float(delta.max()), 4),
            },
        },
    }
    with open(out_dir / "table_4_12_modifier_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def plot_risk_vs_pm25(df: pd.DataFrame, out_dir: Path) -> Path:
    """Generate Fig 4.7 — risk probability vs PM2.5."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.scatter(df["pm2_5"], df["p_risk_adult"] * 100.0, s=8, alpha=0.5, label="Adult")
    plt.scatter(df["pm2_5"], df["p_risk_pediatric"] * 100.0, s=8, alpha=0.5, label="Pediatric")
    plt.xlabel("PM2.5 (µg/m³)")
    plt.ylabel("AR Risk Probability (%)")
    plt.title("Fig 4.7 — Risk Probability vs PM2.5")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "fig_4_7_risk_vs_pm25.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_risk_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Generate Fig 4.7b — mean adult risk by PM2.5 and temperature bins."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    # Create a grid of PM2.5 vs temperature
    pm_bins = np.linspace(df["pm2_5"].min(), df["pm2_5"].max(), 15)
    temp_bins = np.linspace(df["temperature"].min(), df["temperature"].max(), 15)
    pm_labels = (pm_bins[:-1] + pm_bins[1:]) / 2
    temp_labels = (temp_bins[:-1] + temp_bins[1:]) / 2

    df["pm_bin"] = pd.cut(df["pm2_5"], bins=pm_bins, labels=pm_labels)
    df["temp_bin"] = pd.cut(df["temperature"], bins=temp_bins, labels=temp_labels)
    heat = df.groupby(["pm_bin", "temp_bin"], observed=False)["p_risk_adult"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 7))
    cax = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(len(temp_labels)))
    ax.set_xticklabels([f"{x:.1f}" for x in temp_labels], rotation=45)
    ax.set_yticks(range(len(pm_labels)))
    ax.set_yticklabels([f"{x:.1f}" for x in pm_labels])
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Fig 4.7b — Adult Risk Heatmap (PM2.5 vs Temperature)")
    fig.colorbar(cax, ax=ax, label="P_risk")
    plt.tight_layout()
    path = out_dir / "fig_4_7b_risk_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_risk_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """Generate Fig 4.11 — risk distribution histograms by age group."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, age, color in zip(axes, ["adult", "pediatric"], ["steelblue", "darkorange"]):
        data = df[f"p_risk_{age}"] * 100.0
        ax.hist(data, bins=30, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(data.median(), color="red", linestyle="--", label=f"median={data.median():.1f}%")
        ax.set_xlabel("Risk Probability (%)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{age.capitalize()} Risk Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 4.11 — Risk Probability Distribution by Age Group")
    plt.tight_layout()
    path = out_dir / "fig_4_11_risk_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
def run_risk_pipeline(cfg: Dict[str, Any], out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Compute risk tables and figures from the preprocessed raw data."""
    if out_dir is None:
        out_dir = Path(cfg["paths"]["results_dir"])
        if not out_dir.is_absolute():
            out_dir = get_project_root() / out_dir
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    viz_dir = out_dir / "visualizations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_path = get_project_root() / "data" / "processed" / "preprocessed_5min.csv"
    df = pd.read_csv(preprocessed_path, parse_dates=["timestamp"], index_col="timestamp")

    df = add_p_risk_columns(df, cfg)
    df = add_risk_tier_columns(df, cfg)

    # Add a rolling 72h continuous risk column for demonstration
    df["p_risk_adult_72h"] = rolling_72h_continuous_risk(df["p_risk_adult"], cfg)

    distribution = export_risk_distribution(df, cfg, tables_dir)
    modifiers = export_modifier_analysis(df, cfg, tables_dir)
    plot_risk_vs_pm25(df, viz_dir)
    plot_risk_heatmap(df, viz_dir)
    plot_risk_distribution(df, viz_dir)

    return {
        "risk_distribution": distribution,
        "modifier_analysis": modifiers,
        "n_records": int(len(df)),
    }


__all__ = [
    "compute_pm_risk",
    "temperature_modifier",
    "humidity_modifier",
    "combined_odds_ratio",
    "p_risk_from_or",
    "compute_p_risk",
    "rolling_72h_continuous_risk",
    "classify_risk_tier",
    "add_p_risk_columns",
    "add_risk_tier_columns",
    "run_risk_pipeline",
]
