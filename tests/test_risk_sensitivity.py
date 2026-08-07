"""Tests for risk sensitivity analysis."""
import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.risk_sensitivity import analyze_risk_sensitivity, SENSITIVITY_COEFFICIENTS


def _make_df(n=100):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "pm2_5": np.linspace(10, 80, n),
            "pm10": np.linspace(20, 120, n),
            "temperature": np.linspace(25, 35, n),
            "humidity": np.linspace(60, 90, n),
        },
        index=idx,
    )


def test_sensitivity_output_shape():
    cfg = load_config()
    df = _make_df()
    report = analyze_risk_sensitivity(cfg, df)
    assert "baseline" in report
    assert "coefficients" in report
    for coef in SENSITIVITY_COEFFICIENTS:
        assert coef in report["coefficients"]
        assert "plus" in report["coefficients"][coef]
        assert "minus" in report["coefficients"][coef]


def test_sensitivity_monotonic_pm_beta():
    """Increasing PM2.5 beta should increase mean adult risk."""
    cfg = load_config()
    df = _make_df()
    report = analyze_risk_sensitivity(cfg, df)
    plus = report["coefficients"]["pm2_5_beta"]["plus"]
    minus = report["coefficients"]["pm2_5_beta"]["minus"]
    assert plus["mean_p_risk_adult"] >= minus["mean_p_risk_adult"]


def test_sensitivity_tiers_counts_sum():
    cfg = load_config()
    df = _make_df()
    report = analyze_risk_sensitivity(cfg, df)
    total = len(df)
    for coef in SENSITIVITY_COEFFICIENTS:
        for direction in ["plus", "minus"]:
            tiers = report["coefficients"][coef][direction]["adult_tiers"]
            tier_total = sum(t["count"] for t in tiers.values())
            assert tier_total == total
