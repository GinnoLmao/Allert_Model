# ALLERT Rebuild Master Guide
## A Cleaner, Correct, and Comprehensive Blueprint for the Bi-LSTM Air Quality Risk System

> **Purpose:** This guide distills every requirement, design decision, audit finding, and implementation detail from the original ALLERT documentation into a single, modular, easy-to-edit blueprint. It is designed to be handed to any developer or AI assistant to rebuild the system from scratch with fewer bugs, cleaner code, and stronger reproducibility.

> **Status:** Rebuild-ready | Based on original ALLERT thesis system (Iloilo City, Philippines) | Incorporates all critical fixes from the implementation audit.

---

## Table of Contents

1. [What We Are Building](#1-what-we-are-building)
2. [Corrected Design Principles](#2-corrected-design-principles)
3. [Recommended Folder Structure](#3-recommended-folder-structure)
4. [Configuration: Single Source of Truth](#4-configuration-single-source-of-truth)
5. [Data Pipeline](#5-data-pipeline)
6. [Feature Engineering](#6-feature-engineering)
7. [Bi-LSTM Model Architecture](#7-bi-lstm-model-architecture)
8. [Training Strategy (Corrected)](#8-training-strategy-corrected)
9. [Anomaly Detection](#9-anomaly-detection)
10. [Risk Engine](#10-risk-engine)
11. [Evaluation & Baselines](#11-evaluation--baselines)
12. [Results & Thesis Artifacts](#12-results--thesis-artifacts)
13. [Deployment Architecture](#13-deployment-architecture)
14. [Implementation Checklist](#14-implementation-checklist)
15. [Known Pitfalls & Corrections](#15-known-pitfalls--corrections)
16. [Quick Reference: Constants & Thresholds](#16-quick-reference-constants--thresholds)

---

## 1. What We Are Building

### 1.1 System Identity

| Field | Detail |
|-------|--------|
| **Name** | ALLERT — Airborne particulate forecasting and Environmental Risk aLeRT system |
| **Domain** | Environmental health informatics + IoT + deep learning |
| **Target Condition** | Allergic Rhinitis (AR) |
| **Deployment** | Iloilo City, Western Visayas, Philippines |
| **Core Innovation** | Multi-horizon PM2.5/PM10 forecasting + real-time anomaly detection + age-stratified clinical risk scoring |
| **Validation Framework** | ISO/IEC 25010 Software Quality Model |

### 1.2 Problem Statement

Current air quality tools are reactive and generic. ALLERT solves three gaps:

1. **Multi-horizon forecasting:** Predicts PM2.5 and PM10 at 6h, 12h, and 24h instead of only next-hour.
2. **Risk-based alerting:** Translates concentrations into clinically interpretable AR probability, not just threshold exceedance.
3. **Integrated anomaly detection:** Detects sudden pollution spikes that scheduled forecasts might miss.

### 1.3 Research Questions

| RQ | Question | Success Criterion |
|----|----------|-----------------|
| **RQ1** | Can a Bi-LSTM forecast PM2.5/PM10 at 6h/12h/24h with clinically useful accuracy? | MAPE < 5% (6h), < 7% (12h), < 10% (24h); Bi-LSTM outperforms ARIMA. |
| **RQ2** | Can dual-mode anomaly detection distinguish emission events from sensor noise? | Sensitivity > 95%, False Positive Rate < 10%. |
| **RQ3** | Does the risk engine correlate with actual health symptoms? | Spearman rho > 0.30, p < 0.05 vs. VAS/TNSS scores. |

---

## 2. Corrected Design Principles

These principles address the most important findings from the original implementation audit:

| # | Principle | Why It Matters |
|---|-----------|----------------|
| 1 | **Never concatenate train+validation before final training** | The original `train_final.py` concatenated train and validation, disabling early stopping. Validation must remain a held-out signal. |
| 2 | **Fit scalers ONLY on training data** | Prevents data leakage. Validation and test must be transformed using training statistics. |
| 3 | **Preserve chronological order everywhere** | No random shuffling. Time-series splits must be temporal. |
| 4 | **Single source of truth for config** | Every hyperparameter, threshold, and path lives in `config.yaml`. No magic numbers in code. |
| 5 | **Modular, testable components** | Each pipeline stage (preprocessing, features, model, evaluation) is an independent module with unit tests. |
| 6 | **Version every artifact** | Models, scalers, and result JSONs are versioned and traceable. |
| 7 | **Validate every assumption** | Every formula, threshold, and coefficient is documented and unit-tested where possible. |

---

## 3. Recommended Folder Structure

```
allert_rebuild/
├── config/
│   └── config.yaml                 # Single source of truth
├── data/
│   ├── raw/
│   │   └── air_quality_data.csv    # Raw sensor readings
│   ├── processed/
│   │   └── features_5min_scaled.csv
│   └── splits/
│       └── train_val_test_indices.json
├── src/
│   ├── __init__.py
│   ├── config_loader.py            # Load and validate config.yaml
│   ├── preprocessing.py            # Data cleaning, gap fill, scaler
│   ├── feature_engineering.py      # Lag, rolling, cyclical, velocity, exposure
│   ├── sequence_builder.py         # Sliding-window sequence generation
│   ├── pytorch_model.py            # PyTorch Bi-LSTM architecture and loss
│   ├── pytorch_training.py         # Training loop with checkpoint/resume
│   ├── training.py                 # Augmentation, walk-forward CV, curriculum helpers
│   ├── evaluation.py               # Metrics, ARIMA baseline, train/val/test split
│   ├── anomaly_detection.py        # Z-score, velocity, CUSUM
│   ├── risk_engine.py              # Epidemiological risk scoring
│   ├── validity.py                 # Spearman correlation with symptom data
│   ├── export_results.py           # JSON/PNG exporters
│   └── orchestrator.py             # End-to-end CLI pipeline
├── notebooks/
│   └── ALLERT_Master_Notebook.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_bilstm_model.py
│   ├── test_risk_engine.py
│   └── test_anomaly_detection.py
├── model/                          # Saved model artifacts
│   ├── best_model.pt
│   ├── scaler.joblib
│   ├── feature_cols.json
│   └── training_results.json
├── results/                        # Outputs for thesis
│   ├── tables/                     # JSON for Chapter 4 tables
│   └── visualizations/             # PNG figures
├── requirements.txt
└── README.md
```

---

## 4. Configuration: Single Source of Truth

### 4.1 `config.yaml` Template

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

### 4.2 Config Loader (`src/config_loader.py`)

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
    splits = cfg["training"]["train_split"] + cfg["training"]["val_split"] + cfg["training"]["test_split"]
    assert abs(splits - 1.0) < 1e-6, "Train/val/test splits must sum to 1.0"

    lw = cfg["model"]["loss_weights"]
    assert abs(sum(lw) - 1.0) < 1e-6, "Loss weights must sum to 1.0"

    dlw = cfg["risk"]["distributed_lag_weights"]
    assert abs(sum(dlw) - 1.0) < 1e-6, "Distributed lag weights must sum to 1.0"

    assert len(cfg["model"]["forecast_horizons_hours"]) == len(lw), "Loss weights must match horizons"
```

---

## 5. Data Pipeline

### 5.1 Input Data

| Variable | Unit | Source | Valid Range |
|----------|------|--------|-------------|
| PM2.5 | µg/m³ | Sensirion SPS30 | 0 – 1,000 |
| PM10 | µg/m³ | Sensirion SPS30 | 0 – 1,000 |
| Temperature | °C | Sensirion SHT40 | -10 – 60 |
| Humidity | %RH | Sensirion SHT40 | 0 – 100 |

### 5.2 Pipeline Steps

```
Raw CSV (5-min block averages)
    ↓
1. Parse timestamps, sort chronologically, deduplicate
    ↓
2. Range validation → flag OUT_OF_RANGE (do not drop)
    ↓
3. Z-score outlier detection vs. rolling 24h baseline → flag candidates
    ↓
4. Gap classification & fill
       < 2h → linear interpolation (mark INTERPOLATED)
       ≥ 2h → mark GAP (exclude from sequences)
    ↓
5. Rolling median filter (window=3) for noise reduction
    ↓
6. Chronological sort, validate temporal integrity
    ↓
Processed DataFrame ready for feature engineering
```

### 5.3 Required Output: Table 4.7 (`dataset_statistics.json`)

```json
{
  "collection_duration_days": 90.0,
  "collection_start": "2026-01-01",
  "collection_end": "2026-03-31",
  "total_raw_5min_records": 25920,
  "expected_5min_records": 25920,
  "availability_rate_pct": 100.0,
  "gap_summary": {
    "total_gaps_detected": 5,
    "gaps_under_2h_interpolated": 5,
    "gaps_2h_or_more_flagged": 0
  },
  "quality_flags": {"OK": 25800, "SENSOR_ERROR": 0, "OUT_OF_RANGE": 0},
  "imputation_flags": {"ORIGINAL": 25700, "INTERPOLATED": 120, "GAP": 0},
  "per_variable_availability": {
    "pm2_5": {"valid_records": 25800, "availability_pct": 99.54},
    "pm10": {"valid_records": 25800, "availability_pct": 99.54},
    "temperature": {"valid_records": 25920, "availability_pct": 100.0},
    "humidity": {"valid_records": 25920, "availability_pct": 100.0}
  }
}
```

### 5.4 Key Implementation Notes

- **Do not drop flagged records** until sequence generation. Keep provenance flags.
- **Fit scaler only on training partition.** Save scaler for inference.
- **Reject sequences containing GAP-flagged rows** to avoid training on fabricated windows.

---

## 6. Feature Engineering

### 6.1 Final Tensor Shape

```
Input tensor: (N, 2016, F)
  N = number of sequences
  2016 = 7 days × 24 hours × 12 five-minute steps/hour
  F = number of engineered features per timestep
```

### 6.2 Feature Groups

| Group | Features | Count | Justification |
|-------|----------|-------|---------------|
| Raw inputs | pm2_5, pm10, temperature, humidity | 4 | Baseline sensor measurements |
| Lag features (PM2.5) | lag_1h, 6h, 12h, 24h, 48h, 72h | 6 | Autoregressive structure |
| Lag features (PM10) | lag_1h, 6h, 12h, 24h, 48h, 72h | 6 | Autoregressive structure |
| Rolling mean | 6h, 12h, 24h, 168h per PM | 8 | Recent baseline & trend |
| Rolling std | 6h, 12h, 24h, 168h per PM | 8 | Volatility indicator |
| Rolling min/max | 6h, 12h, 24h per PM | 12 | Range dynamics |
| Velocity | 1h, 6h per PM | 4 | Rate-of-change for spikes |
| Cyclical time | hour_sin/cos, dow_sin/cos, month_sin/cos | 6 | Time topology |
| Cumulative exposure | 72h weighted PM2.5 & PM10 | 2 | Distributed lag dose |
| **Total** | | **~56** | Keep compact given 3-month dataset |

### 6.3 Important Formulas

#### Cyclical Encodings
```python
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
dow_sin  = sin(2π * dayofweek / 7)
dow_cos  = cos(2π * dayofweek / 7)
month_sin = sin(2π * month / 12)
month_cos = cos(2π * month / 12)
```

#### Rolling Statistics
```python
SMA_t(w) = (1/w) * Σ x[t-i]  for i in 0..w-1
SD_t(w)  = sqrt((1/w) * Σ (x[t-i] - SMA_t(w))**2)
```

#### Velocity
```python
velocity_t(k) = (x_t - x_{t-k}) / k   # in per-5-minute units
# Convert to µg/m³/hour by multiplying by 12
velocity_hourly = velocity_t(12) * 12
```

#### 72h Cumulative Exposure
```python
exposure_72h = 0.60*PM_t + 0.25*PM_{t-24h} + 0.10*PM_{t-48h} + 0.05*PM_{t-72h}
```

### 6.4 Scaling

- Use **MinMaxScaler** or **StandardScaler**.
- **Fit ONLY on training data.**
- Apply identical transformation to validation and test.
- Save scaler alongside model weights.

### 6.5 Sequence Generation

```python
def build_sequences(df, feature_cols, target_cols, sequence_length=2016, stride=1):
    """
    Returns:
        X: (N, 2016, F)
        y: (N, 6)  -> [pm25_6h, pm10_6h, pm25_12h, pm10_12h, pm25_24h, pm10_24h]
    """
    # Skip windows containing GAP-flagged rows
```

### 6.6 Target Vector Order (Important — Keep Consistent)

```python
TARGET_COLS = [
    "pm2_5_6h", "pm10_6h",
    "pm2_5_12h", "pm10_12h",
    "pm2_5_24h", "pm10_24h",
]
```

---

## 7. Bi-LSTM Model Architecture

### 7.1 Layer Stack

```
Input (batch, 2016, F)
    ↓
Bidirectional LSTM — 128 units, return_sequences=True
    ↓
Recurrent Dropout — 0.25
    ↓
Bidirectional LSTM — 64 units, return_sequences=True
    ↓
Recurrent Dropout — 0.25
    ↓
Temporal Pattern Attention (optional)
    ↓
Global Average Pooling across time dimension
    ↓
Three Parallel Dense Heads:
      ├── Head 6h  → (pm2_5_6h,  pm10_6h)
      ├── Head 12h → (pm2_5_12h, pm10_12h)
      └── Head 24h → (pm2_5_24h, pm10_24h)
```

### 7.2 Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sequence length | 2016 | 7 days of 5-min data |
| LSTM Layer 1 | 128 units (64 forward + 64 backward) | `return_sequences=True` |
| LSTM Layer 2 | 64 units (32 forward + 32 backward) | `return_sequences=True` |
| Recurrent dropout | 0.25 | Applied after each LSTM layer |
| L2 regularization | 1e-4 | On LSTM and Dense kernels |
| Optimizer | Adam | `nn.utils.clip_grad_norm_(..., max_norm=1.0)` |
| Output heads | 3 | One per forecast horizon |
| Units per head | 2 | PM2.5 and PM10 |

### 7.3 Composite Loss Function

```python
# Thesis-aligned temporal weighting
loss = 0.5 * MAE_6h + 0.3 * MAE_12h + 0.2 * MAE_24h
```

**Implementation (PyTorch):**

```python
import torch


def composite_loss(y_pred, y_true, weights=(0.5, 0.3, 0.2)):
    """Weighted MAE across 6h/12h/24h horizons."""
    weights_t = torch.tensor(weights, dtype=y_pred.dtype, device=y_pred.device)
    y_true_h = y_true.reshape(-1, 3, 2)
    y_pred_h = y_pred.reshape(-1, 3, 2)
    mae_per_horizon = torch.mean(torch.abs(y_true_h - y_pred_h), dim=-1)
    weighted = mae_per_horizon * weights_t
    return torch.mean(weighted, dim=-1).mean()
```

### 7.4 Model Builder Template

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPatternAttention(nn.Module):
    def __init__(self, hidden_dim: int, attention_units: int = 64):
        super().__init__()
        self.score_dense = nn.Linear(hidden_dim, attention_units)
        self.weight_dense = nn.Linear(attention_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.tanh(self.score_dense(x))
        scores = self.weight_dense(scores).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return x * weights


class ALLERTBiLSTM(nn.Module):
    def __init__(self, input_size: int, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        self.use_attention = mcfg.get("use_attention", True)
        units_1 = mcfg["lstm_units_layer1"]
        units_2 = mcfg["lstm_units_layer2"]
        dropout = mcfg.get("dropout_rate", 0.25)

        self.lstm1 = nn.LSTM(input_size, units_1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(units_1 * 2, units_2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.attention = (
            TemporalPatternAttention(units_2 * 2, mcfg.get("attention_units", 64))
            if self.use_attention else None
        )

        pooled_dim = units_2 * 2
        self.head_6h = nn.Linear(pooled_dim, 2)
        self.head_12h = nn.Linear(pooled_dim, 2)
        self.head_24h = nn.Linear(pooled_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        if self.attention is not None:
            x = self.attention(x)
        x = x.mean(dim=1)
        out6 = self.head_6h(x)
        out12 = self.head_12h(x)
        out24 = self.head_24h(x)
        return torch.cat([out6, out12, out24], dim=1)


def build_bilstm(input_shape, cfg):
    seq_len, n_features = input_shape
    return ALLERTBiLSTM(n_features, cfg)
```

### 7.5 Temporal Pattern Attention (Lightweight)

```python
class TemporalPatternAttention(nn.Module):
    def __init__(self, hidden_dim: int, attention_units: int = 64):
        super().__init__()
        self.score_dense = nn.Linear(hidden_dim, attention_units)
        self.weight_dense = nn.Linear(attention_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.tanh(self.score_dense(x))
        scores = self.weight_dense(scores).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return x * weights
```

---

## 8. Training Strategy (Corrected)

### 8.1 Data Splitting

```
Chronological Time →
|────────────── Training (~70%) ──────────────|── Validation (15%) ──|── Test (15%) ──|
Day 1                                         Day 63                Day 76           Day 90
```

**Rules:**
- No random shuffling.
- Fit scaler on training only.
- Early stopping monitors validation loss.
- Final metrics computed on test set only.

### 8.2 Corrected Training Function

> **CRITICAL:** Do NOT concatenate train+validation before training. The original implementation had this bug, which disabled early stopping.

Implemented in `src/pytorch_training.py`:

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


def train_model(cfg, X, y, train_mask, val_mask):
    set_random_seeds(cfg)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    model = build_bilstm((seq_len, n_features), cfg).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=cfg["model"]["learning_rate"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["training"]["lr_reduction_factor"],
        patience=cfg["training"]["lr_reduction_patience"],
        min_lr=cfg["training"]["min_lr"],
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=cfg["model"]["batch_size"], shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
        batch_size=cfg["model"]["batch_size"], shuffle=False,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = cfg["training"]["early_stopping_patience"]
    history = []

    for epoch in range(cfg["model"]["max_epochs"]):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = composite_loss(pred, yb, weights=tuple(cfg["model"]["loss_weights"]))
            loss = loss + model.l2_regularization()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["model"]["gradient_clip_norm"])
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = composite_loss(pred, yb, weights=tuple(cfg["model"]["loss_weights"]))
                val_losses.append(loss.item())

        train_loss = float(torch.mean(torch.tensor(train_losses)))
        val_loss = float(torch.mean(torch.tensor(val_losses)))
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "model/best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return history
```

`src/pytorch_training.py` also writes periodic checkpoints to `model/checkpoints/` and resumes from the latest checkpoint automatically.

### 8.3 Hyperparameter Optimization (Optuna)

```python
def objective(trial, cfg):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "lstm_units_layer1": trial.suggest_categorical("lstm_units_layer1", [64, 96, 128, 192, 256]),
        "lstm_units_layer2": trial.suggest_categorical("lstm_units_layer2", [32, 48, 64, 96, 128]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    # Use 5-fold temporal cross-validation
    return mean_val_loss
```

### 8.4 Walk-Forward CV Design

```
Fold 1: Train [Days 1–40]   → Validate [Days 41–50]
Fold 2: Train [Days 1–50]   → Validate [Days 51–60]
Fold 3: Train [Days 1–60]   → Validate [Days 61–70]
Fold 4: Train [Days 1–70]   → Validate [Days 71–78]
Fold 5: Train [Days 1–78]   → Validate [Days 79–86]
```

---

## 9. Anomaly Detection

### 9.1 Dual-Mode Architecture

```
Real-time Sensor Stream
    ├─→ Branch A: Anomaly Detection (Z-score, Velocity, CUSUM)
    └─→ Branch B: Scheduled Bi-LSTM Forecasting
           ↓
      Merged Alert Queue
```

### 9.2 Method 1: Z-Score Residual Analysis

```python
# Residual at time t
e_t = abs(x_actual_t - x_forecast_t)

# Rolling statistics over 24h window (288 five-min steps)
mu_e = rolling_mean(e_t, 288)
sigma_e = rolling_std(e_t, 288)

# Z-score
Z_t = (e_t - mu_e) / sigma_e
```

| Z Range | Tier | Action |
|---------|------|--------|
| Z < 2.0 | Normal | Monitor |
| 2.0 ≤ Z < 3.0 | Spike Watch | Log |
| 3.0 ≤ Z < 4.0 | Spike Warning | Immediate alert |
| Z ≥ 4.0 | Spike Emergency | Immediate alert |

### 9.3 Method 2: Velocity Override

```python
velocity_hourly = (PM_t - PM_{t-1}) * 12   # 5-min step → hourly rate
if velocity_hourly > 15.0:
    trigger_velocity_alert()
```

### 9.4 Method 3: CUSUM Drift Detection

```python
k = 0.5 * sigma_e
h = 4.0 * sigma_e
S_i = max(0, S_{i-1} + e_i - k)
if S_i > h:
    trigger_cusum_alert()
    S_i = 0   # reset after alert
```

### 9.5 Required Output: Table 4.9

```json
{
  "z_score_threshold": 3.0,
  "velocity_threshold_ugm3_per_hour": 15.0,
  "cusum_k": 0.25,
  "cusum_h": 2.0,
  "contingency": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
  "derived_metrics": {
    "sensitivity_pct": 0.0,
    "specificity_pct": 0.0,
    "false_positive_rate_pct": 0.0,
    "precision_pct": 0.0,
    "f1_score": 0.0
  }
}
```

---

## 10. Risk Engine

### 10.1 Pipeline

```
LSTM Forecast (PM2.5, PM10, Temp, RH)
    ↓
1. PM Risk Score: Risk_PM = PM2.5*0.0104 + PM10*0.0026
    ↓
2. Temp Modifier: M_temp = (T - 25) * 0.012
    ↓
3. Humidity Modifier: M_humidity = (RH - 60) * (-0.004)
    ↓
4. Combined OR: OR_total = exp(Risk_PM + M_temp + M_humidity)
    ↓
5. Absolute Risk: P_risk = (P_baseline * OR_total) / (1 + P_baseline*(OR_total - 1))
    ↓
6. 72h Rolling Weighted Average: Risk_continuous = Σ w(τ)*P_risk(t-τ)
    ↓
7. Age-Stratified Tier Classification
```

### 10.2 Distributed Lag Weights (5-minute native)

| Lag Range | Hours | Weight Allocation | Per-step Weight |
|-----------|-------|-------------------|-----------------|
| 0 – 288 steps | 0–24h | 60% | 0.60 / 289 ≈ 0.002076 |
| 289 – 576 steps | 24–48h | 25% | 0.25 / 288 ≈ 0.000868 |
| 577 – 864 steps | 48–72h | 10% | 0.10 / 288 ≈ 0.000347 |
| τ = 864 | 72h boundary | 5% | 0.05 |

**Assert:** `abs(sum(weights) - 1.0) < 1e-12`

### 10.3 Risk Tier Thresholds

| Tier | Adult | Pediatric |
|------|-------|-----------|
| Low | < 15% | < 13% |
| Moderate | 15–25% | 13–23% |
| High | 25–40% | 23–38% |
| Very High | > 40% | > 38% |

### 10.4 Required Outputs

- `table_4_11_risk_distribution.json`
- `table_4_12_modifier_analysis.json`
- `fig_4_7_risk_vs_pm25.png`
- `fig_4_11_risk_distribution.png`

---

## 11. Evaluation & Baselines

### 11.1 Forecasting Metrics

| Metric | Formula | Target (PM2.5/PM10) |
|--------|---------|---------------------|
| MAE | mean(\|y - ŷ\|) | Primary interpretable metric |
| RMSE | sqrt(mean((y - ŷ)²)) | Penalizes large errors |
| MAPE | 100 * mean(\|y - ŷ\| / y) | **6h < 5%, 12h < 7%, 24h < 10%** |
| R² | 1 - SS_res/SS_tot | Positive and > ARIMA |

### 11.2 ARIMA Baseline

- Model: ARIMA(2,1,2)
- Fit on training data.
- Predict same horizons on identical test set.
- Export `arima_results.json`.

### 11.3 Clinical Validity

- Spearman's rho between predicted risk tier and VAS/TNSS scores.
- Target: rho > 0.30, p < 0.05.
- Export `table_4_15_construct_validity.json`.

---

## 12. Results & Thesis Artifacts

### 12.1 Required Tables

| Table | Content | Output File |
|-------|---------|-------------|
| 4.6 | Sensor calibration (R², MAE, bias) | External hardware logs |
| 4.7 | Data quality & preprocessing | `tables/dataset_statistics.json` |
| 4.8 | Bi-LSTM forecasting performance | `tables/table_4_8_bilstm.json` |
| 4.8 | ARIMA baseline | `tables/arima_results.json` |
| 4.9 | Anomaly detection contingency | `tables/table_4_9_anomaly.json` |
| 4.10 | CUSUM detection lag | `tables/table_4_10_cusum.json` |
| 4.11 | Risk tier distribution | `tables/table_4_11_risk_distribution.json` |
| 4.12 | Modifier magnitude analysis | `tables/table_4_12_modifier_analysis.json` |
| 4.13 | End-to-end latency | `tables/table_4_13_latency.json` |
| 4.14 | SUS/TAM scores | External survey data |
| 4.15 | Construct validity | `tables/table_4_15_construct_validity.json` |

### 12.2 Required Figures

| Figure | Description | Output File |
|--------|-------------|-------------|
| 4.5a | Training convergence (loss curves) | `visualizations/fig_4_5a_training_convergence.png` |
| 4.5b | Bi-LSTM vs ARIMA overlay | `visualizations/fig_4_5_bilstm_vs_arima.png` |
| 4.6 | Residual distribution + ACF | `visualizations/fig_4_6_residual_distribution.png` |
| 4.7 | Risk vs PM2.5 | `visualizations/fig_4_7_risk_vs_pm25.png` |
| 4.11 | Risk distribution histograms | `visualizations/fig_4_11_risk_distribution.png` |
| 4.14 | Validity scatter | `visualizations/fig_4_14_construct_validity.png` |

---

## 13. Deployment Architecture

### 13.1 Six-Layer Stack

```
┌─────────────────────────────────────┐
│ Layer 6: Client Apps                │
│ Web App | User Portal | Admin Panel │
├─────────────────────────────────────┤
│ Layer 5: Backend API                │
│ Auth | Forecast API | Symptoms API  │
├─────────────────────────────────────┤
│ Layer 4: Alert Engine               │
│ Risk Classifier | Alert Dispatcher  │
├─────────────────────────────────────┤
│ Layer 3: ML Engine                  │
│ LSTM | Anomaly Detector | Risk Eng  │
├─────────────────────────────────────┤
│ Layer 2: Data Processing            │
│ QA | Gap Fill | Feature Engineering │
├─────────────────────────────────────┤
│ Layer 1: Data Ingestion             │
│ Ingestion API | Buffer | Data Logger│
├─────────────────────────────────────┤
│ IoT Edge: ESP32-S3 + SPS30 + SHT40  │
└─────────────────────────────────────┘
```

### 13.2 Operational Targets

| Target | Value |
|--------|-------|
| Sensor uptime | ≈ 99% |
| End-to-end latency | < 30 seconds |
| Data availability | ≥ 95% hourly completeness |
| API response (cached) | < 500 ms |
| Retraining cycle | Every 30 days |

---

## 14. Implementation Checklist

### Phase 1: Foundation
- [ ] Create folder structure
- [ ] Write `config.yaml` with all parameters
- [ ] Implement `config_loader.py` with validation
- [ ] Implement preprocessing pipeline (clean, gap fill, flags)
- [ ] Implement feature engineering (lags, rolling, cyclical, velocity, exposure)
- [ ] Implement sequence builder with GAP exclusion
- [ ] Write unit tests for preprocessing and feature engineering

### Phase 2: Model
- [ ] Implement `bilstm_model.py` with corrected architecture
- [ ] Implement composite loss function
- [ ] Implement training function with held-out validation
- [ ] Implement temporal train/val/test split
- [ ] Implement walk-forward CV for Optuna
- [ ] Write unit tests for model building

### Phase 3: Evaluation
- [ ] Implement regression metrics
- [ ] Implement ARIMA(2,1,2) baseline
- [ ] Implement anomaly detection (Z-score, velocity, CUSUM)
- [ ] Implement risk engine with 72h distributed lag
- [ ] Implement Spearman validity check
- [ ] Implement JSON/PNG exporters

### Phase 4: Integration
- [ ] Build orchestrator CLI
- [ ] Build master notebook
- [ ] Run end-to-end on presentation subset
- [ ] Run full training on 3-month dataset
- [ ] Verify all Chapter 4 artifacts are produced

---

## 15. Known Pitfalls & Corrections

| # | Pitfall (from original audit) | Correction |
|---|-------------------------------|------------|
| 1 | Concatenating train+validation disables early stopping | Keep validation held-out; pass `(X_val, y_val)` to `model.fit()` |
| 2 | Fitting scaler on full dataset | Fit scaler on training split only |
| 3 | Random shuffling of sequences | Use chronological splits exclusively |
| 4 | Magic numbers in code | Centralize all values in `config.yaml` |
| 5 | Missing validation monitoring | Use `EarlyStopping` + `ReduceLROnPlateau` on `val_loss` |
| 6 | Unclear target column order | Standardize as `[pm25_6h, pm10_6h, pm25_12h, pm10_12h, pm25_24h, pm10_24h]` |
| 7 | CUSUM parameters undocumented | Set `k = 0.5 * sigma`, `h = 4.0 * sigma`; reset after alert |
| 8 | Distributed lag weights not normalized | Assert sum = 1.0; test with constant input |
| 9 | Missing figure automation | Add `plot_risk_distribution()` and `plot_validity_scatter()` |
| 10 | No unified orchestrator | Create `orchestrator.py` CLI and master notebook |

---

## 16. Quick Reference: Constants & Thresholds

| Constant | Value | Location |
|----------|-------|----------|
| Sequence length | 2016 steps (7 days × 5 min) | `model.sequence_length` |
| LSTM units | 128 → 64 | `model.*` |
| Dropout | 0.25 | `model.dropout_rate` |
| L2 lambda | 1e-4 | `model.l2_lambda` |
| Loss weights | 0.5 / 0.3 / 0.2 | `model.loss_weights` |
| Learning rate | 0.001 | `model.learning_rate` |
| Batch size | 32 | `model.batch_size` |
| Max epochs | 200 | `model.max_epochs` |
| Train/Val/Test | 70% / 15% / 15% | `training.*_split` |
| Early stopping patience | 10 | `training.early_stopping_patience` |
| Z-score thresholds | 2.0 / 3.0 / 4.0 | `anomaly.z_thresholds` |
| Velocity threshold | 15 µg/m³/hour | `anomaly.velocity_threshold_ugm3_per_hour` |
| PM2.5 beta | 0.0104 | `risk.pm2_5_beta` |
| PM10 beta | 0.0026 | `risk.pm10_beta` |
| Temp baseline | 25°C | `risk.temp_baseline_c` |
| Humidity baseline | 60% | `risk.humidity_baseline_pct` |
| Adult baseline prevalence | 0.20 | `risk.baseline_prevalence_adult` |
| Pediatric baseline prevalence | 0.22 | `risk.baseline_prevalence_pediatric` |
| Distributed lag weights | 0.60 / 0.25 / 0.10 / 0.05 | `risk.distributed_lag_weights` |

---

## Closing Note

This guide is intentionally **self-contained, modular, and traceable**. Each section maps to the original thesis structure while correcting the critical bugs and inconsistencies found in the first implementation. Use it as the single reference document for rebuilding ALLERT from scratch — cleaner, more efficient, easier to edit, and correct.
