# 12 — Implementation Checklist

This checklist breaks the rebuild into concrete phases. Mark items complete as you implement them.

## Phase 1: Foundation

- [ ] Create folder structure from `02_FOLDER_STRUCTURE.md`
- [ ] Write `config/config.yaml` with all parameters
- [ ] Implement `src/config_loader.py` with validation
- [ ] Copy raw data to `data/raw/air_quality_data.csv`
- [ ] Implement `src/preprocessing.py`
  - [ ] Timestamp parsing and chronological sort
  - [ ] Range validation with flags
  - [ ] Z-score outlier detection
  - [ ] Gap classification and interpolation
  - [ ] Rolling median filter
  - [ ] Dataset statistics export (Table 4.7)
- [ ] Implement `src/feature_engineering.py`
  - [ ] Cyclical encodings
  - [ ] Lag features
  - [ ] Rolling statistics
  - [ ] Velocity indicators
  - [ ] 72h cumulative exposure
- [ ] Implement `src/sequence_builder.py`
  - [ ] Sliding windows (length=2016, stride=1)
  - [ ] GAP exclusion
  - [ ] Target vector generation
- [ ] Write unit tests for preprocessing, feature engineering, and sequence builder

## Phase 2: Model

- [ ] Implement `src/bilstm_model.py`
  - [ ] `build_bilstm()` with two Bi-LSTM layers
  - [ ] Recurrent dropout and L2 regularization
  - [ ] Optional temporal pattern attention
  - [ ] Three parallel output heads
  - [ ] Composite loss function
- [ ] Implement `src/evaluation.py`
  - [ ] Chronological train/val/test split
  - [ ] Walk-forward CV indices
  - [ ] MAE, RMSE, MAPE, R² functions
  - [ ] ARIMA(2,1,2) baseline
- [ ] Implement corrected training function with held-out validation
  - [ ] EarlyStopping on `val_loss`
  - [ ] ReduceLROnPlateau
  - [ ] Model checkpointing
- [ ] Implement Optuna HPO (optional)
- [ ] Write unit tests for model builder

## Phase 3: Post-Processing

- [ ] Implement `src/anomaly_detection.py`
  - [ ] Z-score residual monitor
  - [ ] Velocity override
  - [ ] CUSUM drift detector
  - [ ] Contingency table export (Table 4.9)
  - [ ] CUSUM lag analysis export (Table 4.10)
- [ ] Implement `src/risk_engine.py`
  - [ ] PM risk score
  - [ ] Temperature and humidity modifiers
  - [ ] OR → probability conversion
  - [ ] 72h distributed lag
  - [ ] Age-stratified tier classification
  - [ ] Risk distribution export (Table 4.11)
  - [ ] Modifier analysis export (Table 4.12)
- [ ] Implement `src/validity.py`
  - [ ] Spearman correlation
  - [ ] Construct validity export (Table 4.15)
- [ ] Write unit tests for anomaly detection and risk engine

## Phase 4: Evaluation & Export

- [ ] Implement `src/export_results.py`
  - [ ] JSON exporters for all tables
  - [ ] Matplotlib figure generators
- [ ] Generate all required visualizations
- [ ] Run full evaluation on test set
- [ ] Compare Bi-LSTM vs ARIMA
- [ ] Verify all Chapter 4 artifacts exist

## Phase 5: Integration

- [ ] Implement `src/orchestrator.py` CLI
- [ ] Build `notebooks/ALLERT_Master_Notebook.ipynb`
- [ ] Run end-to-end on a small presentation subset
- [ ] Run full training on 3-month dataset
- [ ] Verify reproducibility with fixed random seed
- [ ] Run full test suite

## Phase 6: Documentation & Hardening

- [ ] Update README.md with run instructions
- [ ] Add inline traceability comments to all `src/` files
- [ ] Document any deviations from thesis methodology
- [ ] Add logging at each pipeline stage
- [ ] Add latency instrumentation (Table 4.13)

## Known Pitfalls & Corrections

| # | Pitfall | Correction |
|---|---------|------------|
| 1 | Concatenating train+validation before training | Keep validation held-out; pass `(X_val, y_val)` to `model.fit()` |
| 2 | Fitting scaler on full dataset | Fit scaler on training split only |
| 3 | Random shuffling of sequences | Use chronological splits exclusively |
| 4 | Magic numbers in code | Centralize all values in `config.yaml` |
| 5 | Missing validation monitoring | Use EarlyStopping + ReduceLROnPlateau on `val_loss` |
| 6 | Unclear target column order | Standardize as `[pm25_6h, pm10_6h, pm25_12h, pm10_12h, pm25_24h, pm10_24h]` |
| 7 | CUSUM parameters undocumented | Set `k = 0.5 * sigma`, `h = 4.0 * sigma`; reset after alert |
| 8 | Distributed lag weights not normalized | Assert sum = 1.0; test with constant input |
| 9 | Missing figure automation | Add risk distribution and validity scatter plots |
| 10 | No unified orchestrator | Create `orchestrator.py` CLI and master notebook |
| 11 | Dropping flagged rows silently | Preserve provenance flags and exclude GAP windows at sequence build |
| 12 | No random seed fixation | Set Python, NumPy, TensorFlow seeds for reproducibility |

## Definition of Done

The rebuild is complete when:

1. All unit tests pass.
2. Full pipeline runs end-to-end without errors.
3. All Chapter 4 tables and figures are generated.
4. Bi-LSTM outperforms ARIMA at 12h and 24h horizons.
5. Risk engine produces epidemiologically sensible tier distributions.
6. Anomaly detection meets sensitivity > 95% and FPR < 10% on labeled data.
7. Training uses held-out validation with early stopping and LR reduction.

---

*Next: read [`13_FILE_MANIFEST.md`](13_FILE_MANIFEST.md) to know what existing files to bring into this rebuild.*
