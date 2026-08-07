# 03 — Configuration: Single Source of Truth

Every parameter, threshold, coefficient, and path in the ALLERT system must live in one file: `config/config.yaml`. This eliminates magic numbers, prevents inconsistencies, and makes the entire system reproducible.

## 3.1 Design Rules

1. **No hardcoded values in source code.** Every number must be read from `config.yaml`.
2. **Validate on load.** The config loader checks that splits sum to 1.0, weights sum to 1.0, and required keys exist.
3. **One file per environment.** Use `config.yaml` for the canonical setup. Override paths via environment variables only if necessary.
4. **Comment every section.** Future developers must understand why each value exists.

## 3.2 `config/config.yaml` Template

```yaml
# =============================================================================
# ALLERT System Configuration
# =============================================================================
project:
  name: "ALLERT"
  version: "2.0.0"
  random_seed: 42

paths:
  data_raw: "data/raw/air_quality_data.csv"
  data_processed: "data/processed/features_5min_scaled.csv"
  model_dir: "model"
  results_dir: "results"
  visualizations_dir: "results/visualizations"
  tables_dir: "results/tables"

sampling:
  raw_interval_seconds: 10
  transmission_interval_minutes: 5
  records_per_day: 288        # 24 * 60 / 5

preprocessing:
  gap_fill_threshold_hours: 2
  rolling_median_window: 3    # 5-min readings
  outlier_zscore_threshold: 5.0
  availability_target_pct: 95

features:
  target_variables:
    - pm2_5
    - pm10
  raw_inputs:
    - pm2_5
    - pm10
    - temperature
    - humidity
  lag_hours: [1, 6, 12, 24, 48, 72]
  rolling_windows_hours: [6, 12, 24, 168]   # 168 = 7 days
  velocity_hours: [1, 6]
  cyclical_features:
    - hour
    - dayofweek
    - month
  cumulative_exposure_hours: 72
  scaler_type: "minmax"       # or "standard"; fit on training only

model:
  sequence_length: 2016       # 7 days * 24 hours * 12 steps/hour
  lstm_units_layer1: 128
  lstm_units_layer2: 64
  dropout_rate: 0.25
  recurrent_dropout: 0.25
  l2_lambda: 1.0e-4
  learning_rate: 0.001
  batch_size: 32
  max_epochs: 200
  gradient_clip_norm: 1.0
  use_attention: true
  attention_units: 64
  forecast_horizons_hours: [6, 12, 24]
  loss_weights: [0.5, 0.3, 0.2]

training:
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  early_stopping_patience: 10
  early_stopping_monitor: "val_loss"
  restore_best_weights: true
  lr_reduction_factor: 0.5
  lr_reduction_patience: 5
  min_lr: 1.0e-6

anomaly:
  z_thresholds: [2.0, 3.0, 4.0]
  velocity_threshold_ugm3_per_hour: 15.0
  cusum_k_multiplier: 0.5     # k = multiplier * sigma_residual
  cusum_h_multiplier: 4.0     # h = multiplier * sigma_residual
  residual_rolling_window_hours: 24

risk:
  pm2_5_beta: 0.0104
  pm10_beta: 0.0026
  temp_baseline_c: 25.0
  temp_beta: 0.012
  humidity_baseline_pct: 60.0
  humidity_beta: -0.004
  baseline_prevalence_adult: 0.20
  baseline_prevalence_pediatric: 0.22
  distributed_lag_weights: [0.60, 0.25, 0.10, 0.05]
  distributed_lag_hours: [0, 24, 48, 72]
  adult_tiers: {low: 0.0, moderate: 0.15, high: 0.25, very_high: 0.40}
  pediatric_tiers: {low: 0.0, moderate: 0.13, high: 0.23, very_high: 0.38}

hpo:
  enabled: true
  n_trials: 50
  cv_folds: 5
  search_space:
    learning_rate: [1.0e-4, 1.0e-2]
    lstm_units_layer1: [64, 96, 128, 192, 256]
    lstm_units_layer2: [32, 48, 64, 96, 128]
    dropout_rate: [0.1, 0.5]
    batch_size: [16, 32, 64]
```

## 3.3 Config Loader (`src/config_loader.py`)

```python
"""Load and validate config.yaml."""
from pathlib import Path
import yaml
from typing import Dict, Any

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Dict[str, Any]) -> None:
    splits = (
        cfg["training"]["train_split"]
        + cfg["training"]["val_split"]
        + cfg["training"]["test_split"]
    )
    assert abs(splits - 1.0) < 1e-6, "Train/val/test splits must sum to 1.0"

    lw = cfg["model"]["loss_weights"]
    assert abs(sum(lw) - 1.0) < 1e-6, "Loss weights must sum to 1.0"

    dlw = cfg["risk"]["distributed_lag_weights"]
    assert abs(sum(dlw) - 1.0) < 1e-6, "Distributed lag weights must sum to 1.0"

    assert len(cfg["model"]["forecast_horizons_hours"]) == len(
        lw
    ), "Loss weights must match forecast horizons"

    assert cfg["preprocessing"]["gap_fill_threshold_hours"] >= 0
    assert cfg["model"]["sequence_length"] > 0
```

## 3.4 Accessing Config in Code

Every module should load config once at startup:

```python
from src.config_loader import load_config

cfg = load_config()

# Example usages
sequence_length = cfg["model"]["sequence_length"]
batch_size = cfg["model"]["batch_size"]
velocity_threshold = cfg["anomaly"]["velocity_threshold_ugm3_per_hour"]
```

## 3.5 Critical Validation Checks

| Check | Why It Matters |
|-------|----------------|
| Splits sum to 1.0 | Prevents silently losing data |
| Loss weights sum to 1.0 | Ensures composite loss is well-defined |
| Distributed lag weights sum to 1.0 | Ensures rolling average is normalized |
| Horizon count matches loss-weight count | Prevents shape mismatches in target tensor |
| Sequence length equals 7 days at 5-min resolution | `2016 = 7 × 24 × 12` |

## 3.6 Differences from Original Config

The original `config.yaml` was good but mixed sections inconsistently. This version:
- Groups related parameters under clear headers
- Adds explicit validation rules
- Separates `model` from `training` hyperparameters
- Adds `hpo` section for Optuna
- Removes obsolete keys and adds comments

---

*Next: read [`04_DATA_PIPELINE.md`](04_DATA_PIPELINE.md) to understand preprocessing.*
