"""ALLERT Risk Engine — Unit Tests."""
import pytest
import numpy as np
import pandas as pd

from src.config_loader import load_config
from src import risk_engine as re


@pytest.fixture
def cfg():
    return load_config()


def test_worked_example(cfg):
    """Reproduce the exact worked example in RISK_ENGINE.md."""
    pm25, pm10, temp, rh = 35.0, 60.0, 30.0, 70.0

    risk_pm = re.compute_pm_risk(pm25, pm10, cfg)
    expected_risk_pm = (pm25 * cfg["risk"]["pm2_5_beta"]) + (pm10 * cfg["risk"]["pm10_beta"])
    assert pytest.approx(risk_pm, rel=1e-4) == expected_risk_pm

    m_temp = re.temperature_modifier(temp, cfg)
    assert pytest.approx(m_temp, rel=1e-4) == (temp - cfg["risk"]["temp_baseline_c"]) * cfg["risk"]["temp_beta"]

    m_hum = re.humidity_modifier(rh, cfg)
    assert pytest.approx(m_hum, rel=1e-4) == (rh - cfg["risk"]["humidity_baseline_pct"]) * cfg["risk"]["humidity_beta"]

    or_total = re.combined_odds_ratio(risk_pm, m_temp, m_hum)
    assert pytest.approx(or_total, rel=1e-3) == 1.716

    p_risk = re.compute_p_risk(pm25, pm10, temp, rh, "adult", cfg)
    assert pytest.approx(p_risk * 100.0, abs=0.5) == 30.0

    tier, _ = re.classify_risk_tier(p_risk, "adult", cfg)
    assert tier == "high"


def test_pediatric_threshold_shift(cfg):
    """Pediatric baseline should yield higher P_risk than adult."""
    pm25, pm10, temp, rh = 35.0, 60.0, 30.0, 70.0
    p_adult = re.compute_p_risk(pm25, pm10, temp, rh, "adult", cfg)
    p_ped = re.compute_p_risk(pm25, pm10, temp, rh, "pediatric", cfg)
    assert p_ped > p_adult


def test_distributed_lag_weights_sum_to_one(cfg):
    weights = cfg["risk"]["distributed_lag_weights"]
    assert pytest.approx(sum(weights), abs=1e-12) == 1.0


def test_distributed_lag_normalization(cfg):
    """With constant input, rolling_72h_continuous_risk must return the same constant after warm-up."""
    constant = 0.25
    idx = pd.date_range("2024-01-01", periods=1000, freq="5min")
    s = pd.Series(constant, index=idx)
    result = re.rolling_72h_continuous_risk(s, cfg)
    assert pytest.approx(result.iloc[-1], abs=1e-6) == constant
