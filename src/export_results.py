"""ALLERT Results Exporter

Generates thesis-ready JSON artifacts and visualization figures for
Chapter 4 tables that are not produced directly by the training or
anomaly modules.

Produces:
  - table_4_11_risk_distribution.json
  - table_4_12_modifier_analysis.json
  - table_4_13_latency.json   (stub / template)
  - Fig 4.11: Risk probability distribution histogram (adult vs pediatric)

Thesis Traceability:
- Chapter 4, Table 4.11 (Risk Probability Distribution)
- Chapter 4, Table 4.12 (Meteorological Modifier Contribution)
- Chapter 4, Table 4.13 (End-to-End Latency)
- Chapter 4, Fig. 4.11 (Risk Distribution Histogram)
"""
from pathlib import Path
from typing import Dict, Optional
import json
import numpy as np
import pandas as pd

from src import risk_engine as re


def export_risk_distribution(
    df: pd.DataFrame,
    age_col: str = "age_group",
    out_dir: Optional[Path] = None,
) -> Dict:
    """Build Table 4.11 — Risk Probability Distribution by Alert Tier.

    df must contain columns:
      - 'p_risk'      : float (0-1)
      - 'age_group'   : str ('adult' or 'pediatric')
    """
    def _tier_counts(series: pd.Series) -> Dict:
        low = int(((series < 0.15)).sum())
        mod = int(((series >= 0.15) & (series < 0.25)).sum())
        high = int(((series >= 0.25) & (series <= 0.40)).sum())
        vhigh = int(((series > 0.40)).sum())
        total = low + mod + high + vhigh
        return {
            "Low": {"count": low, "pct": round(100 * low / total, 2) if total else 0},
            "Moderate": {"count": mod, "pct": round(100 * mod / total, 2) if total else 0},
            "High": {"count": high, "pct": round(100 * high / total, 2) if total else 0},
            "Very_High": {"count": vhigh, "pct": round(100 * vhigh / total, 2) if total else 0},
        }

    adult = df[df[age_col] == "adult"]["p_risk"]
    ped = df[df[age_col] == "pediatric"]["p_risk"]

    result = {
        "total_calculations": int(len(df)),
        "adult": _tier_counts(adult),
        "pediatric": _tier_counts(ped),
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "table_4_11_risk_distribution.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("Exported Table 4.11 risk distribution to", out_dir / "table_4_11_risk_distribution.json")
    return result


def export_modifier_analysis(
    df: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> Dict:
    """Build Table 4.12 — Meteorological Modifier Contribution.

    df must contain columns:
      - 'temperature' : float (°C)
      - 'humidity'    : float (%RH)
      - 'pm2_5'       : float
      - 'pm10'        : float
    """
    mt = df["temperature"].apply(re.temperature_modifier)
    mh = df["humidity"].apply(re.humidity_modifier)

    # compute P_risk with and without modifiers to isolate their impact
    pr_full = df.apply(
        lambda r: re.compute_p_risk(r["pm2_5"], r["pm10"], r["temperature"], r["humidity"], r.get("age_group", "adult")), axis=1
    )
    pr_pm_only = df.apply(
        lambda r: re.compute_p_risk(r["pm2_5"], r["pm10"], 25.0, 60.0, r.get("age_group", "adult")), axis=1
    )
    delta = (pr_full - pr_pm_only) * 100.0  # percentage points

    result = {
        "temperature": {
            "observed_range_C": {
                "min": float(df["temperature"].min()),
                "max": float(df["temperature"].max()),
                "mean": float(df["temperature"].mean()),
            },
            "M_temp_range": {
                "min": float(mt.min()),
                "max": float(mt.max()),
                "mean": float(mt.mean()),
            },
        },
        "humidity": {
            "observed_range_RH": {
                "min": float(df["humidity"].min()),
                "max": float(df["humidity"].max()),
                "mean": float(df["humidity"].mean()),
            },
            "M_humidity_range": {
                "min": float(mh.min()),
                "max": float(mh.max()),
                "mean": float(mh.mean()),
            },
        },
        "combined_modifier_impact": {
            "cases_where_temp_increases_risk": int((mt > 0).sum()),
            "cases_where_humidity_decreases_risk": int((mh < 0).sum()),
            "mean_combined_modifier_delta_pct": round(float(delta.mean()), 4),
            "delta_range_pct": {
                "min": round(float(delta.min()), 4),
                "max": round(float(delta.max()), 4),
            },
        },
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "table_4_12_modifier_analysis.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("Exported Table 4.12 modifier analysis to", out_dir / "table_4_12_modifier_analysis.json")
    return result


def plot_risk_distribution(
    df: pd.DataFrame,
    age_col: str = "age_group",
    out_dir: Optional[Path] = None,
) -> Path:
    """Generate Fig 4.11 — Risk Probability Distribution Histogram (age-stratified, presentation-quality).

    Creates an overlaid histogram + KDE plot with tier threshold annotations
    and shaded danger zones for direct visual comparison between adult and pediatric.

    Saves PNG to out_dir and returns the file path.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(12, 7))

    adult = df[df[age_col] == "adult"]["p_risk"] * 100.0
    ped = df[df[age_col] == "pediatric"]["p_risk"] * 100.0

    # --- Overlaid histograms (semi-transparent for comparison) ---
    bins = np.linspace(0, 100, 35)
    ax.hist(adult, bins=bins, color="steelblue", alpha=0.5, label=f"Adult (n={len(adult)})", edgecolor="white", linewidth=0.5)
    ax.hist(ped, bins=bins, color="darkorange", alpha=0.5, label=f"Pediatric (n={len(ped)})", edgecolor="white", linewidth=0.5)

    # --- KDE curves (smooth density estimate) ---
    for data, color, label in [(adult, "steelblue", "Adult Trend"), (ped, "darkorange", "Pediatric Trend")]:
        if len(data) > 1:
            kde = gaussian_kde(data)
            x_range = np.linspace(0, 100, 500)
            kde_vals = kde(x_range)
            # Scale KDE to match histogram peak for visual alignment
            hist_counts, _ = np.histogram(data, bins=bins)
            scale_factor = hist_counts.max() / kde_vals.max()
            ax.plot(x_range, kde_vals * scale_factor, color=color, linewidth=3, linestyle="-", label=label)

    # --- Tier threshold vertical lines with annotations ---
    tier_lines = [
        (15, "Low → Moderate", "green", "Most of the time: safe conditions"),
        (25, "Moderate → High", "orange", "Caution: sensitive groups affected"),
        (40, "High → Very High", "red", "Danger: emergency-level risk"),
    ]
    for x, label, color, desc in tier_lines:
        ax.axvline(x, color=color, linestyle="--", linewidth=2, alpha=0.8)
        ax.text(x + 1, ax.get_ylim()[1] * 0.92, f"{label}\n({desc})", color=color, fontsize=10, rotation=0, va="top")

    # --- Shaded danger zone ---
    ax.axvspan(40, 100, color="red", alpha=0.05, label="Very High Risk Zone")

    # --- Median annotations ---
    for data, color, name in [(adult, "steelblue", "Adult"), (ped, "darkorange", "Pediatric")]:
        med = data.median()
        ax.axvline(med, color=color, linestyle=":", linewidth=2, alpha=0.9)
        ax.text(med, ax.get_ylim()[1] * 0.55, f"{name} median\n{med:.1f}%", color=color, fontsize=11, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=color))

    # --- Key statistics box ---
    low_adult = (adult < 15).mean() * 100
    low_ped = (ped < 15).mean() * 100
    vhigh_adult = (adult > 40).mean() * 100
    vhigh_ped = (ped > 40).mean() * 100
    stats_text = (
        f"Key Stats:\n"
        f"  Low-risk frequency: Adult {low_adult:.1f}% | Ped {low_ped:.1f}%\n"
        f"  Very High frequency: Adult {vhigh_adult:.1f}% | Ped {vhigh_ped:.1f}%\n"
        f"  → Children encounter Very High risk {vhigh_ped/vhigh_adult:.1f}× more often"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9, edgecolor="gray"))

    ax.set_xlabel("Risk Probability (%)", fontsize=13)
    ax.set_ylabel("Frequency / Density", fontsize=13)
    ax.set_title("Fig 4.11 — Risk Probability Distribution by Age Group", fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_xlim(0, 100)
    plt.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "fig4_11_risk_distribution.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print("Exported Fig 4.11 risk distribution to", path)
        return path
    plt.close()
    return None


def export_latency_template(out_dir: Optional[Path] = None) -> Dict:
    """Stub template for Table 4.13 — End-to-End Latency Measurements.

    Populate with real timestamps after deployment instrumentation.
    """
    template = {
        "note": "Populate with measured latency values from deployment instrumentation.",
        "components": {
            "sensor_to_cloud_ingestion": {"mean_ms": None, "p95_ms": None, "max_ms": None, "target_ms": 5000},
            "preprocessing": {"mean_ms": None, "p95_ms": None, "max_ms": None, "target_ms": 3000},
            "bilstm_inference": {"mean_ms": None, "p95_ms": None, "max_ms": None, "target_ms": 10000},
            "risk_engine_calculation": {"mean_ms": None, "p95_ms": None, "max_ms": None, "target_ms": 2000},
            "alert_dispatch": {"mean_ms": None, "p95_ms": None, "max_ms": None, "target_ms": 10000},
        },
        "total_e2e": {"mean_ms": None, "p95_ms": None, "max_ms": None, "target_ms": 30000},
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "table_4_13_latency.json", "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)
        print("Exported Table 4.13 latency template to", out_dir / "table_4_13_latency.json")
    return template


if __name__ == "__main__":
    # smoke test with dummy data
    rng = np.random.default_rng(99)
    n = 500
    df = pd.DataFrame({
        "pm2_5": rng.uniform(10, 80, n),
        "pm10": rng.uniform(20, 120, n),
        "temperature": rng.uniform(24, 34, n),
        "humidity": rng.uniform(60, 90, n),
        "age_group": rng.choice(["adult", "pediatric"], n),
    })
    df["p_risk"] = df.apply(
        lambda r: re.compute_p_risk(r["pm2_5"], r["pm10"], r["temperature"], r["humidity"], r.get("age_group", "adult")),
        axis=1,
    )
    out = Path("results")
    export_risk_distribution(df, out_dir=out)
    export_modifier_analysis(df, out_dir=out)
    export_latency_template(out_dir=out)
    plot_risk_distribution(df, out_dir=out)
