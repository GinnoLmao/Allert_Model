# 10 — Evaluation & Baselines

This document defines how to evaluate the forecasting model, anomaly detection, system performance, usability, and clinical validity.

## 10.1 Phase 1: Technical Validation

### 10.1.1 Forecasting Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| MAE | mean(\|y - ŷ\|) | Interpretable error in µg/m³ |
| RMSE | sqrt(mean((y - ŷ)²)) | Penalizes large errors |
| MAPE | 100 × mean(\|y - ŷ\| / y) | Scale-independent percentage |
| R² | 1 - SS_res / SS_tot | Explained variance |

**Critical note:** Precision, recall, and F-measure are **classification metrics**. They apply **only** to anomaly detection, **not** to PM forecasting.

**Metric-unit correctness:** `src/evaluation.py` loads the fitted `TargetScaler` and inverts predictions and targets before computing metrics. The default `config.yaml` keeps targets in original µg/m³ units (`features.scale_targets: false`), so the inversion is an identity transform. The exported `table_4_8_bilstm.json` and `arima_results.json` include a `"units"` field set to `"original µg/m³"`. To verify this, run `tests/test_target_scaling.py` and inspect the `"units"` key in the generated JSON.

### 10.1.2 Accuracy Targets

| Horizon | MAPE Target |
|---------|-------------|
| 6-hour | < 5% |
| 12-hour | < 7% |
| 24-hour | < 10% |

These apply independently to PM2.5 and PM10.

### 10.1.3 Baselines

#### ARIMA(2,1,2)

- Fit on training data
- Predict identical horizons on test set
- Export `results/tables/arima_results.json`

#### Persistence (Naive)

- Uses the median of the last hour of training data as the forecast for all test steps.
- Provides a simple lower-bound benchmark.
- Export `results/tables/persistence_results.json`

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_series, order=(2, 1, 2))
model_fit = model.fit()
forecasts = model_fit.forecast(steps=horizon_steps)
```

### 10.1.4 Forecasting Evaluation JSON

`results/tables/table_4_8_bilstm.json`:

```json
{
  "model": "Bi-LSTM (TensorFlow)",
  "units": "original µg/m³",
  "metrics": {
    "PM2_5": {
      "6h": {"MAE": 2.1, "RMSE": 3.0, "MAPE": 4.5, "R2": 0.85},
      "12h": {"MAE": 3.2, "RMSE": 4.5, "MAPE": 6.8, "R2": 0.72},
      "24h": {"MAE": 4.5, "RMSE": 6.1, "MAPE": 9.2, "R2": 0.58}
    },
    "PM10": {
      "6h": {"MAE": 2.5, "RMSE": 3.6, "MAPE": 4.8, "R2": 0.82},
      "12h": {"MAE": 3.8, "RMSE": 5.2, "MAPE": 7.1, "R2": 0.68},
      "24h": {"MAE": 5.1, "RMSE": 7.0, "MAPE": 9.8, "R2": 0.52}
    }
  }
}
```

## 10.2 Phase 2: Anomaly Detection Evaluation

| Metric | Formula | Target |
|--------|---------|--------|
| Sensitivity | TP / (TP + FN) | > 95% |
| Specificity | TN / (TN + FP) | > 90% |
| False Positive Rate | FP / (FP + TN) | < 10% |
| Precision | TP / (TP + FP) | > 85% |
| F1-Score | 2PR / (P + R) | > 90% |

See `08_ANOMALY_DETECTION.md` for output schemas.

## 10.3 Phase 3: System Performance Efficiency

Measure end-to-end latency across N = 100 sampled cycles:

| Stage | Target |
|-------|--------|
| Sensor → Cloud | < 5 s |
| Preprocessing | < 3 s |
| Bi-LSTM inference | < 10 s |
| Risk engine | < 2 s |
| Alert dispatch | < 10 s |
| **Total E2E** | **< 30 s** |

Export: `results/tables/table_4_13_latency.json`

Use the orchestrator `--with-timing` flag to populate this table automatically:

```powershell
.venv\Scripts\python src\orchestrator.py --mode evaluate --subset-days 20 --stride 2 --batch-size 64 --with-timing
```

The latency report is produced by `src/utils/timing.py` and visualized in `results/visualizations/fig_4_13_latency_breakdown.png`.

## 10.4 Phase 4: Usability Evaluation

### 10.4.1 System Usability Scale (SUS)

```
SUS Score = (Sum of item scores - 10) × 2.5
```

| Range | Interpretation |
|-------|----------------|
| ≥ 90 | Excellent |
| ≥ 80 | Strong |
| ≥ 68 | Acceptable |
| < 68 | Below acceptable |

### 10.4.2 Technology Acceptance Model (TAM)

| Construct | Scale | Target |
|-----------|-------|--------|
| Perceived Ease of Use (PEOU) | 7-point Likert | > 5.0 |
| Perceived Usefulness (PU) | 7-point Likert | > 5.0 |
| Behavioral Intention (BI) | 7-point Likert | > 5.0 |

## 10.5 Phase 5: Clinical Validity

### 10.5.1 Symptom Tracking Protocol

| Instrument | Description | Range |
|------------|-------------|-------|
| VAS | Overall AR discomfort | 0–100 |
| TNSS | Nasal symptom composite | 0–12 |

Minimum 30 days of paired observations per participant.

### 10.5.2 Correlation Analysis

Use **Spearman's rank-order correlation** because risk tiers are ordinal.

```
H0: No monotonic relationship between predicted risk tier and VAS score
H1: Higher risk tiers correlate with higher VAS scores
```

**Target:** rho > 0.30, p < 0.05.

Export: `results/tables/table_4_15_construct_validity.json`

```json
{
  "spearman_rho": 0.42,
  "pearson_r": 0.38,
  "p_value": 0.001,
  "n_symptom_records": 200
}
```

## 10.6 Required Visualizations

| Figure | Description | Output File |
|--------|-------------|-------------|
| 4.4 | HPO optimization history | `fig_4_4_hpo_optimization_history.png` |
| 4.4b | HPO parameter importance | `fig_4_4b_hpo_param_importance.png` |
| 4.5a | Training convergence | `fig_4_5a_training_convergence.png` |
| 4.5b | Bi-LSTM vs ARIMA overlay | `fig_4_5b_bilstm_vs_arima.png` |
| 4.6 | Residual distribution + ACF | `fig_4_6_residual_distribution.png` |
| 4.6b | CUSUM decision function | `fig_4_6b_cusum_decision.png` |
| 4.6c | Confusion matrices | `fig_4_6c_confusion_matrix_z.png`, `fig_4_6c_confusion_matrix_cusum.png` |
| 4.7 | Risk vs PM2.5 | `fig_4_7_risk_vs_pm25.png` |
| 4.7b | Risk heatmap | `fig_4_7b_risk_heatmap.png` |
| 4.8b | Permutation importance | `fig_4_8b_permutation_importance.png` |
| 4.8d | Attention weights | `fig_4_8d_attention_weights.png` |
| 4.8e | Error by regime | `fig_4_8e_error_by_regime.png` |
| 4.11 | Risk distribution histograms | `fig_4_11_risk_distribution.png` |
| 4.12b | Risk sensitivity tornado | `fig_4_12b_risk_sensitivity.png` |
| 4.13 | Latency breakdown | `fig_4_13_latency_breakdown.png` |
| 4.14 | Validity scatter | `fig_4_14_construct_validity.png` |

## 10.7 Evaluation Summary Table

| Component | Method | Primary Metric | Target |
|-----------|--------|----------------|--------|
| Forecasting 6h | Temporal CV | MAPE | < 5% |
| Forecasting 12h | Temporal CV | MAPE | < 7% |
| Forecasting 24h | Temporal CV | MAPE | < 10% |
| ARIMA comparison | Hold-out test | MAE, R² | Bi-LSTM better at 12h/24h |
| Persistence comparison | Hold-out test | MAE, R² | Bi-LSTM better |
| Anomaly detection | Contingency | Sensitivity | > 95% |
| Anomaly detection | Contingency | FPR | < 10% |
| Latency | Timestamp logging | E2E seconds | < 30 s |
| Usability | SUS | Score | ≥ 68 |
| TAM | Survey | Mean PU/PEOU/BI | ≥ 3.5/5 or 5.0/7 |
| Clinical validity | Spearman | rho, p | > 0.30, < 0.05 |

---

*Next: read [`11_DEPLOYMENT.md`](11_DEPLOYMENT.md) for production architecture.*
