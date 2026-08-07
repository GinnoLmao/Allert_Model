# 02 — Recommended Folder Structure

A clean folder structure makes the system modular, testable, and easy for any AI assistant or developer to navigate. Every file has a single, obvious home.

## 2.1 Top-Level Layout

```
allert_rebuild/
├── config/
│   └── config.yaml                 # Single source of truth
├── data/
│   ├── raw/
│   │   └── air_quality_data.csv    # Raw 5-min block averages
│   ├── processed/
│   │   └── features_5min_scaled.csv
│   └── splits/
│       └── train_val_test_indices.json
├── src/
│   ├── __init__.py
│   ├── config_loader.py            # Load and validate config.yaml
│   ├── preprocessing.py            # Cleaning, gap fill, scaler
│   ├── feature_engineering.py      # Lags, rolling, cyclical, velocity, exposure
│   ├── sequence_builder.py         # Sliding-window sequences
│   ├── pytorch_model.py            # PyTorch Bi-LSTM architecture and loss
│   ├── pytorch_training.py         # Training loop with checkpoint/resume
│   ├── training.py                 # Augmentation, CV, curriculum helpers
│   ├── evaluation.py               # Metrics, splits, ARIMA baseline
│   ├── anomaly_detection.py        # Z-score, velocity, CUSUM
│   ├── risk_engine.py              # Epidemiological risk scoring
│   ├── validity.py                 # Spearman correlation
│   ├── export_results.py           # JSON and PNG exporters
│   └── orchestrator.py             # End-to-end CLI pipeline
├── notebooks/
│   └── ALLERT_Master_Notebook.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_bilstm_model.py
│   ├── test_risk_engine.py
│   └── test_anomaly_detection.py
├── model/                          # Saved artifacts
│   ├── best_model.pt
│   ├── scaler.joblib
│   ├── feature_cols.json
│   └── training_results.json
├── results/                        # Thesis outputs
│   ├── tables/
│   └── visualizations/
├── docs/                           # This documentation suite
├── requirements.txt
└── README.md
```

## 2.2 Directory Responsibilities

### `config/`
Holds `config.yaml` only. No other file should define hyperparameters or thresholds.

### `data/`
- `raw/` — Immutable raw sensor CSV.
- `processed/` — Feature-engineered, scaled data.
- `splits/` — Chronological train/val/test index boundaries.

### `src/`
One responsibility per module:

| Module | Responsibility |
|--------|----------------|
| `config_loader.py` | Load YAML, validate sums and constraints |
| `preprocessing.py` | Clean raw data, gap fill, flag records, scale |
| `feature_engineering.py` | Build all engineered features |
| `sequence_builder.py` | Build `(N, 2016, F)` tensors and `(N, 6)` targets |
| `pytorch_model.py` | Build the PyTorch Bi-LSTM |
| `pytorch_training.py` | Train with held-out validation and checkpoint/resume |
| `training.py` | Augmentation, walk-forward CV, curriculum helpers |
| `evaluation.py` | Metrics, temporal splits, ARIMA baseline |
| `anomaly_detection.py` | Z-score, velocity override, CUSUM |
| `risk_engine.py` | Convert forecasts to AR risk |
| `validity.py` | Correlation with symptom scores |
| `export_results.py` | Save tables and figures |
| `orchestrator.py` | Run the full pipeline from CLI |

### `tests/`
Unit tests for every deterministic module. Priority order:

1. `test_risk_engine.py` — verify worked example and weight normalization
2. `test_preprocessing.py` — verify gap filling, flags, scaler leakage prevention
3. `test_feature_engineering.py` — verify lag offsets and cyclical encodings
4. `test_anomaly_detection.py` — verify Z-score, velocity, CUSUM logic
5. `test_bilstm_model.py` — verify model output shapes
### `model/`

Versioned artifacts:
- Trained model weights (`best_model.pt`)
- Fitted scaler (`scaler.joblib`)
- Feature column list (`feature_cols.json`)
- Training metadata (`training_results.json`)

### `results/`
Everything needed for thesis Chapter 4:
- `tables/` — JSON files for Tables 4.7–4.15
- `visualizations/` — PNG figures for Chapter 4

## 2.3 Files That Should NOT Exist

Avoid these anti-patterns:

| Bad Practice | Why |
|--------------|-----|
| Hardcoded paths in notebooks/scripts | Use `config.yaml` |
| Duplicate config in multiple files | Single source of truth |
| Mixing raw and processed data | Data provenance becomes unclear |
| Giant monolithic scripts | Hard to test and debug |
| Results scattered outside `results/` | Hard to find thesis artifacts |

---

*Next: read [`03_CONFIGURATION.md`](03_CONFIGURATION.md) to design the central config file.*
