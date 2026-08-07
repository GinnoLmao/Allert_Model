"""Performance regression tests for pinned full-run metrics.

This test fails only if the current run's metrics drift outside the allowed
tolerance of the pinned values. If no pinned metrics are present, the test
skips with a clear reason.
"""
import json
from pathlib import Path

import pytest


PINNED_PATH = Path(__file__).with_name("fixtures") / "pinned_metrics.json"
BILSTM_PATH = Path(__file__).resolve().parents[1] / "results" / "tables" / "table_4_8_bilstm.json"


def _load_pinned():
    if not PINNED_PATH.exists():
        pytest.skip(f"No pinned metrics found at {PINNED_PATH}")
    with open(PINNED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_current():
    if not BILSTM_PATH.exists():
        pytest.skip(f"No current metrics found at {BILSTM_PATH}; run evaluation first")
    with open(BILSTM_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric(current_metrics, key, metric_name):
    # Keys are like "PM2_5_6h" — split pollutant/horizon on the last underscore
    horizon = key.split("_")[-1]
    pollutant = "_".join(key.split("_")[:-1])
    return current_metrics[pollutant][horizon][metric_name]


@pytest.mark.skipif(not BILSTM_PATH.exists(), reason="No current metrics artifact")
def test_mape_regression():
    """Fail if pinned MAPE drifts by more than the allowed tolerance."""
    pinned = _load_pinned()
    current = _load_current()
    current_metrics = current["metrics"]
    for key, expected in pinned.items():
        if key.startswith("_"):
            continue
        expected_mape = expected["MAPE"] if isinstance(expected, dict) else expected
        actual_mape = _metric(current_metrics, key, "MAPE")
        assert abs(actual_mape - expected_mape) < 5.0, (
            f"{key} MAPE drifted: expected {expected_mape}, got {actual_mape}"
        )


@pytest.mark.skipif(not BILSTM_PATH.exists(), reason="No current metrics artifact")
def test_mae_monotonic_with_horizon():
    """Sanity check: MAE should not systematically decrease with longer horizon."""
    current = _load_current()
    metrics = current["metrics"]
    for pollutant in ["PM2_5", "PM10"]:
        mae_6 = metrics[pollutant]["6h"]["MAE"]
        mae_24 = metrics[pollutant]["24h"]["MAE"]
        # Allow small fluctuation; 24h MAE should not be dramatically lower
        assert mae_24 >= 0.5 * mae_6, (
            f"{pollutant} 24h MAE ({mae_24}) suspiciously lower than 6h MAE ({mae_6})"
        )


def test_pinned_fixture_exists():
    """Ensure the example fixture is present and well-formed."""
    pinned = _load_pinned()
    assert "_note" in pinned
    assert "PM2_5_6h" in pinned
