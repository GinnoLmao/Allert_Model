# 16 — ALLERT Thesis Improvement Plan

> **Purpose:** This document is a living roadmap for turning the current ALLERT rebuild into a thesis-ready, defensible, and clinically meaningful system. It is more than a wish list: every item has a rationale, an owner (you), concrete tasks, acceptance criteria, and an estimate of effort. It should be read after `docs/15_SYSTEM_SUMMARY_AND_IMPROVEMENTS.md` and updated as work progresses.

---

## 16.1 Executive Summary

### 16.1.1 Where the system is now

The rebuilt ALLERT pipeline is **end-to-end correct and reproducible**. It loads data, preprocesses it, engineers 56 features, builds sliding-window sequences, trains a Bi-LSTM with held-out validation, evaluates against ARIMA, runs dual-mode anomaly detection, computes age-stratified risk, and checks construct validity. All 25 unit tests pass.

### 16.1.2 Where the system falls short for a thesis

1. **Forecasting accuracy** — On the fast local smoke run (20 days, 3 epochs) the Bi-LSTM is dramatically worse than the ARIMA baseline (MAPE ≈ 100% vs ≈ 11%). This is a training budget problem, not a pipeline bug, but it must be fixed before any accuracy claims are made.
2. **Anomaly detection calibration** — Z-score and velocity detectors are too insensitive; CUSUM is overly sensitive on a tiny test window. Thresholds were auto-calibrated on residuals from an undertrained model.
3. **Clinical validity** — The current correlation uses a synthetic proxy VAS dataset. Real paired VAS/TNSS records are required for defensible clinical validity.
4. **Deployment evidence** — There is no measured latency, usability (SUS), or technology acceptance (TAM) data. Table 4.13 and related thesis artifacts are missing.
5. **Reproducibility & interpretability** — There is no seed-pinned full run, no feature-importance analysis, and no attention-weight visualization. Reviewers will ask "why does the model predict what it does?"
6. **Thesis integration** — Figures and tables are exported but not yet packaged into a single, thesis-ready bundle with consistent numbering and captions.

### 16.1.3 Top 5 priorities

| Rank | Action | Why it matters | Rough effort |
|------|--------|----------------|--------------|
| 1 | Full-dataset training on a CUDA-enabled local GPU (90+ days, 100–200 epochs, HPO) | Meets RQ1 MAPE targets and makes the Bi-LSTM outperform ARIMA. | 1–2 weeks |
| 2 | Recalibrate anomaly detectors on the full test set | Meets RQ2 sensitivity >95%, FPR <10%. | 3–5 days |
| 3 | Collect real VAS/TNSS data and rerun construct validity | Meets RQ3 and turns the thesis from a technical demo into a clinical support tool. | 2–4 weeks |
| 4 | Instrument latency and run SUS/TAM surveys | Fills Table 4.13 and the ISO/IEC 25010 evaluation chapters. | 1 week |
| 5 | Add interpretability (SHAP / permutation / attention plots) and a seed-pinned reproducibility harness | Answers reviewer questions about model behavior. | 2–3 days |

---

## 16.2 Gap Analysis

### 16.2.1 Research-question gap matrix

| Research Question | Success Criterion | Current Status | Gap | Root Cause | Improvement |
|-------------------|-------------------|----------------|-----|------------|-------------|
| RQ1 | MAPE < 5% (6h), < 7% (12h), < 10% (24h); Bi-LSTM > ARIMA | MAPE ≈ 100%, Bi-LSTM << ARIMA | Large | Tiny subset, 3 epochs, no HPO | Full data + more epochs + HPO |
| RQ2 | Sensitivity > 95%, FPR < 10% | CUSUM 93.8%, others lower | Moderate | Thresholds calibrated on small subset | Recalibrate on full data, per-pollutant thresholds |
| RQ3 | Spearman rho > 0.30, p < 0.05 | rho = 0.73 (synthetic) | Small | No real symptom data | Collect VAS/TNSS records |
| RQ4 (system quality) | E2E latency < 30 s, SUS ≥ 68, TAM > 5.0 | Not measured | Large | No instrumentation or surveys | Add latency logging + user studies |

### 16.2.2 Technical debt inventory

| Module | Debt | Impact | Fix |
|--------|------|--------|-----|
| `evaluation.py` | Metrics were not explicitly guaranteed to be in original units | Accuracy claims are hard to interpret | Invert scaling before metric export; default keeps targets in µg/m³ |
| `anomaly_detection.py` | Thresholds were global, not per-pollutant | Sensitivity is too low for z-score/velocity | Added per-pollutant threshold config and `src/anomaly_calibration.py` |
| `risk_engine.py` | Epidemiological coefficients are fixed from literature | May not match local prevalence | Added `src/risk_sensitivity.py` |
| `validity.py` | Uses synthetic paired data | No real clinical claim | Load real VAS/TNSS CSV (still pending data collection) |
| `orchestrator.py` | No latency instrumentation | Cannot generate Table 4.13 | Added `src/utils/timing.py` and `--with-timing` flag |
| `tests/` | No performance regression tests | Cannot detect model drift | Added `tests/test_performance_regression.py` with placeholder fixture |
| `ALLERT_Master_Notebook.ipynb` | Great for smoke runs, but no interpretability cells | Hard to answer "why did the model predict that?" | Added optional reproducibility, latency, interpretability, and bundle cells |

---

## 16.3 Improvement Objectives & Key Performance Indicators (KPIs)

### 16.3.1 Forecasting KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| PM2.5 MAPE (6h/12h/24h) | < 5% / < 7% / < 10% | `table_4_8_bilstm.json` |
| PM10 MAPE (6h/12h/24h) | < 5% / < 7% / < 10% | `table_4_8_bilstm.json` |
| Bi-LSTM MAE vs ARIMA | Lower at 12h and 24h | `table_4_8_bilstm.json` vs `arima_results.json` |
| R² (per target) | > 0.70 | `table_4_8_bilstm.json` |
| Residual ACF | No significant autocorrelation at 95% CI | `fig_4_6_residual_distribution.png` |

### 16.3.2 Anomaly Detection KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| Sensitivity | > 95% | `table_4_9_anomaly.json` |
| Specificity | > 90% | `table_4_9_anomaly.json` |
| False Positive Rate | < 10% | `table_4_9_anomaly.json` |
| F1 Score | > 90% | `table_4_9_anomaly.json` |
| Detection lag | < 15 minutes for sudden spikes | `table_4_10_cusum.json` |

### 16.3.3 Clinical Validity KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| Spearman rho | > 0.30 | `table_4_15_construct_validity.json` |
| Pearson r | > 0.30 | `table_4_15_construct_validity.json` |
| p-value | < 0.05 | `table_4_15_construct_validity.json` |
| Sample size | ≥ 30 paired records | `table_4_15_construct_validity.json` |

### 16.3.4 System Quality KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| End-to-end latency | < 30 s | `table_4_13_latency.json` |
| SUS score | ≥ 68 | User survey |
| TAM PEOU/PU/BI | > 5.0 (7-point scale) | User survey |
| Unit-test coverage | ≥ 80% line coverage | `coverage.py` |
| Regression test | MAPE within ±5% of pinned run | `tests/test_performance_regression.py` |

---

## 16.4 Phased Implementation Plan

The work is broken into six phases. Phases can overlap, but Phase 1 (full training) gates almost everything else.

### Phase 1 — Full-Dataset Training & Baseline Reproduction (Weeks 1–2)

**Goal:** Train the Bi-LSTM on the full 3-month dataset long enough to meet RQ1 accuracy targets.

**Tasks**

1. **Local GPU setup**
   - [x] Ensure the PyTorch environment is installed and CUDA is available (`torch.cuda.is_available()`).
   - [ ] Ensure `requirements.txt` is pinned (add exact versions).
   - [x] Verify GPU availability and driver/CUDA compatibility.

2. **Full run**
   - [ ] Run `python src/orchestrator.py --mode full --epochs 200 --batch-size 32` (or the notebook equivalent with `SUBSET_DAYS = None`, `STRIDE = 1`).
   - [ ] Monitor training convergence; if it plateaus early, consider lowering learning rate or increasing patience.
   - [ ] Save `best_model.pt`, `scaler.joblib`, `feature_cols.json`, `training_results.json`.

3. **Hyperparameter optimization**
   - [ ] Run `python src/orchestrator.py --mode hpo --hpo-trials 50 --epochs 50 --batch-size 32`.
   - [ ] Save `optuna_best_params.json` and `optuna_study.pkl`.
   - [ ] If HPO finds a better configuration, retrain with it.

4. **Reproducibility harness**
   - [x] Add a `--seed` CLI flag to `orchestrator.py`.
   - [x] Add PyTorch deterministic-mode support in `src/pytorch_model.py` and `src/pytorch_training.py` (and document the performance cost).
   - [ ] Run the full pipeline twice with the same seed and verify identical `best_model.pt` and identical MAPE/R².

**Deliverables**
- `model/best_model.pt` trained on full data.
- `model/optuna_best_params.json`.
- `model/training_results.json` with full history.
- Local GPU setup notes documenting the CUDA/PyTorch environment.
- A seed-pinned reproducibility run log.

**Acceptance criteria**
- PM2.5 and PM10 MAPE meet targets at all three horizons (or are within 2% of targets with documented justification).
- Bi-LSTM MAE is lower than ARIMA at 12h and 24h.
- The same seed produces the same metrics on two independent runs.

---

### Phase 2 — Anomaly Detection Recalibration (Weeks 2–3)

**Goal:** Make z-score, velocity, and CUSUM detectors meet RQ2 targets on the full test set.

**Tasks**

1. **Per-pollutant thresholds**
   - [x] Add `anomaly.z_thresholds_pm2_5`, `anomaly.z_thresholds_pm10`, `anomaly.velocity_threshold_pm2_5`, `anomaly.velocity_threshold_pm10` to `config.yaml`.
   - [x] Refactor `anomaly_detection.py` to use per-pollutant thresholds instead of global ones.

2. **CUSUM tuning**
   - [x] Add `src/anomaly_calibration.py` with a `grid_search_cusum()` helper that sweeps `k_multiplier` and `h_multiplier` on a validation slice and selects the combination that maximizes F1 while keeping FPR < 10%.
   - [x] Export the chosen parameters to `results/tables/table_4_9b_anomaly_calibration_*.json`.
   - [ ] Refit CUSUM on the full-dataset residual distribution and write the chosen values back to `config.yaml` after final training.

3. **Evaluation**
   - [ ] Run the anomaly pipeline on the full test set.
   - [ ] Export `table_4_9_anomaly.json` and `table_4_10_cusum.json`.
   - [ ] Verify sensitivity > 95%, FPR < 10%.

4. **Visualization**
   - [ ] Add a CUSUM decision-function plot (S_t, k, h lines) as `fig_4_6b_cusum_decision.png`.
   - [ ] Add a confusion-matrix heatmap for each detector as `fig_4_6c_confusion_matrix.png`.

**Deliverables**
- Updated `config.yaml` with per-pollutant thresholds.
- Refactored `src/anomaly_detection.py`.
- New helper `src/anomaly_calibration.py` (or similar) with grid search.
- `fig_4_6b_cusum_decision.png`, `fig_4_6c_confusion_matrix.png`.

**Acceptance criteria**
- All three detectors meet sensitivity > 95% and FPR < 10% on the full test set.
- CUSUM detection lag is documented and ≤ 15 minutes for sudden spikes.

---

### Phase 3 — Clinical Validity & Risk Engine Refinement (Weeks 3–5)

**Goal:** Replace synthetic validity data with real VAS/TNSS records and validate the risk engine.

**Tasks**

1. **Real symptom data collection**
   - [ ] Define the data-collection protocol: ≥ 30 days of paired VAS (0–100) and TNSS (0–12) entries from participants.
   - [ ] Store the data in `data/clinical/symptom_records.csv` with columns: `timestamp`, `participant_id`, `vas_score`, `tnss_score`, `medication_flag`, `indoor_flag`, `notes`.
   - [ ] Add a loader in `src/validity.py` that reads this CSV and aligns it to the 5-minute grid.

2. **Validity analysis**
   - [ ] Run `run_validity_analysis` with the real paired data.
   - [ ] Export `table_4_15_construct_validity.json` and `fig_4_14_construct_validity.png`.
   - [ ] Add a longitudinal plot of VAS/TNSS vs. risk tier over time as `fig_4_14b_longitudinal_validity.png`.

3. **Risk engine sensitivity analysis**
   - [x] Add a `src/risk_sensitivity.py` module that perturbs `pm2_5_beta`, `pm10_beta`, `temp_beta`, `humidity_beta`, and baseline prevalence by ±10% and reports the change in P_risk and tier distribution.
   - [x] Export `table_4_12b_risk_sensitivity.json` and `fig_4_12b_risk_sensitivity.png`.
   - [ ] Document which coefficients the risk engine is most sensitive to (populate discussion text after full run).

4. **Risk threshold validation**
   - [ ] Compare adult vs pediatric tier distributions on the full dataset.
   - [ ] If pediatric tiers are too saturated (e.g., 100% "high"), discuss whether to adjust `pediatric_tiers` or baseline prevalence, and justify any change with epidemiological citations.

**Deliverables**
- `data/clinical/symptom_records.csv` (anonymized).
- Updated `src/validity.py`.
- `table_4_15_construct_validity.json` with real data.
- `table_4_12b_risk_sensitivity.json` and `fig_4_12b_risk_sensitivity.png`.

**Acceptance criteria**
- Spearman rho > 0.30, p < 0.05 on real data.
- Risk engine sensitivity is documented and does not invalidate the tier thresholds.

---

### Phase 4 — Deployment & Latency Instrumentation (Weeks 4–5)

**Goal:** Fill Table 4.13 and provide evidence for ISO/IEC 25010 system-quality evaluation.

**Tasks**

1. **Latency logging**
   - [x] Add a `timing` context manager to `src/utils/timing.py`.
   - [x] Instrument `orchestrator.py` to record per-phase latencies for preprocessing, feature engineering, scaling, sequence building, training, evaluation, anomaly detection, risk engine, validity, and interpretability.
   - [x] Export `results/tables/table_4_13_latency.json`.

2. **End-to-end latency figure**
   - [x] Add `fig_4_13_latency_breakdown.png` showing mean/p95/max per component.

3. **Usability & TAM surveys**
   - [ ] Deploy a prototype dashboard (Streamlit or Flask) showing forecasts, risk tiers, and anomaly alerts.
   - [ ] Administer SUS (10 questions) and TAM (PEOU/PU/BI) to ≥ 10 target users.
   - [ ] Export `results/tables/table_4_14_sus_tam.json` with mean scores and individual responses (anonymized).

4. **Production checklist**
   - [ ] Add a `docs/17_DEPLOYMENT_CHECKLIST.md` with steps for ESP32 deployment, API ingestion, and alert dispatch.

**Deliverables**
- `table_4_13_latency.json` with real measured latencies.
- `fig_4_13_latency_breakdown.png`.
- `table_4_14_sus_tam.json`.
- `docs/17_DEPLOYMENT_CHECKLIST.md`.

**Acceptance criteria**
- E2E latency < 30 s (mean and p95).
- SUS ≥ 68 and TAM > 5.0 on all constructs.

---

### Phase 5 — Interpretability & Model Analysis (Weeks 5–6)

**Goal:** Answer "why does the model predict what it does?" with feature-importance and attention visualizations.

**Tasks**

1. **Permutation importance**
   - [x] Add `src/interpretability.py` with a `permutation_importance()` function that permutes one feature at a time and reports the drop in validation MAE.
   - [x] Export `table_4_8b_permutation_importance.json` and `fig_4_8b_permutation_importance.png`.

2. **SHAP values (optional but recommended)**
   - [x] Use `shap.DeepExplainer` or `GradientExplainer` on a sample of test sequences if `shap` is installed; otherwise skip cleanly.
   - [ ] Export `fig_4_8c_shap_summary.png` (requires `shap` and a full-dataset model).

3. **Attention-weight visualization**
   - [x] Extract attention weights from the trained model (the `attn_weights` layer).
   - [x] Plot the mean attention weight over the 2016 time steps for a sample of test sequences as `fig_4_8d_attention_weights.png`.
   - [ ] Discuss whether the attention focuses on recent hours (as expected) or distant lags.

4. **Error analysis by regime**
   - [ ] Break down MAPE/MAE by hour-of-day, day-of-week, and PM level (low/moderate/high).
   - [ ] Export `table_4_8c_error_by_regime.json` and `fig_4_8e_error_by_regime.png`.

**Deliverables**
- `src/interpretability.py`.
- `table_4_8b_permutation_importance.json`, `table_4_8c_error_by_regime.json`.
- `fig_4_8b_permutation_importance.png`, `fig_4_8c_shap_summary.png`, `fig_4_8d_attention_weights.png`, `fig_4_8e_error_by_regime.png`.

**Acceptance criteria**
- At least one interpretability figure is included in the thesis and discussed in Chapter 4.

---

### Phase 6 — Thesis Integration & Defense Preparation (Weeks 6–7)

**Goal:** Package all artifacts into a thesis-ready bundle and prepare the defense.

**Tasks**

1. **Figure/table audit**
   - [ ] Ensure every figure has a thesis number, a caption, and is referenced in the text.
   - [ ] Ensure every table has a thesis number, a caption, and is referenced in the text.
   - [x] Export a single `results/thesis_bundle.zip` with all tables and figures via `src/thesis_exporter.py` and `--mode bundle`.

2. **Code-to-thesis mapping**
   - [ ] Update `CODE_TO_THESIS_MAPPING.md` (or create `docs/18_CODE_TO_THESIS_MAPPING.md`) so every thesis section points to the exact file and function that produced it.

3. **Defense walkthrough**
   - [ ] Create a `docs/19_DEFENSE_WALKTHROUGH.md` with:
     - A 15-minute presentation outline.
     - Key numbers to memorize (MAPE, R², sensitivity, rho, latency, SUS).
     - Anticipated questions and answers (e.g., "Why did you choose a Bi-LSTM over a transformer?").

4. **Reproducibility statement**
   - [ ] Write a short `docs/20_REPRODUCIBILITY_STATEMENT.md` that says exactly how to reproduce the thesis results (seed, command, environment, data hash).

**Deliverables**
- `results/thesis_bundle.zip`.
- `docs/18_CODE_TO_THESIS_MAPPING.md`.
- `docs/19_DEFENSE_WALKTHROUGH.md`.
- `docs/20_REPRODUCIBILITY_STATEMENT.md`.

**Acceptance criteria**
- A committee member can run a single command and reproduce all thesis figures/tables within tolerance.

---

## 16.5 Detailed Task Breakdown by Module

### 16.5.1 Data Pipeline

| Task | File(s) | Details | Priority |
|------|---------|---------|----------|
| Add `--seed` CLI flag | `src/orchestrator.py` | Propagate to `set_random_seeds`. | High |
| Add deterministic mode | `src/pytorch_model.py`, `src/pytorch_training.py` | Enable `torch.use_deterministic_algorithms` and set CUDNN flags when seed is set. | High |
| Add data-augmentation config | `config.yaml`, `src/training.py` | Jitter + window slicing + mixup, with configurable probability. | Medium |
| Add walk-forward CV | `src/training.py` | Alternative to single chronological split. | Medium |
| Add gap-handling config | `config.yaml` | Allow dropping sequences with >X% interpolated rows. | Medium |
| Add outlier-handling config | `config.yaml` | Allow clipping vs. flagging. | Low |

### 16.5.2 Feature Engineering

| Task | File(s) | Details | Priority |
|------|---------|---------|----------|
| Add interaction features | `src/feature_engineering.py` | PM2.5 × temperature, PM10 × humidity. | Medium |
| Add AQI sub-index features | `src/feature_engineering.py` | EPA AQI sub-indices for PM2.5 and PM10. | Medium |
| Add calendar features | `src/feature_engineering.py` | Weekday/weekend, holiday flags for Philippines. | Low |
| Add lag-to-feature correlation report | `src/feature_engineering.py` | Export `table_4_5a_feature_correlation.json`. | Low |

### 16.5.3 Model Architecture

| Task | File(s) | Details | Priority |
|------|---------|---------|----------|
| Add GRU variant | `src/pytorch_model.py` | `model_type: "bilstm" | "bigru"` in config. | Medium |
| Add CNN-LSTM hybrid | `src/pytorch_model.py` | 1D CNN before LSTM for feature extraction. | Low |
| Add attention ablation | `config.yaml` | `use_attention: false` to test the TPA contribution. | Medium |
| Add dropout grid | `config.yaml` | Allow different dropout for LSTM vs. dense layers. | Low |

### 16.5.4 Training

| Task | File(s) | Details | Priority |
|------|---------|---------|----------|
| Add walk-forward CV | `src/training.py` | Implemented `run_walk_forward_training()` and `--walk-forward-cv` flag. | Medium |
| Add curriculum learning | `src/training.py` | Scaffolded `build_curriculum_y()`; not wired by default. | Low |
| Add early-stopping on MAPE | `src/training.py` | Set `early_stopping_monitor: "val_mae"` or `"val_loss"` in config. | Low |
| Add training-progress logger | `src/training.py` | `ProgressLogger` saves per-epoch metrics to `model/training_progress.json`. | Medium |

### 16.5.5 Evaluation

| Task | File(s) | Details | Priority |
|------|---------|---------|----------|
| Invert scaling before metrics | `src/evaluation.py` | Ensure MAE/RMSE/MAPE are in µg/m³. | High |
| Add per-hour/per-day metrics | `src/evaluation.py` | Exported `table_4_8c_error_by_regime.json` and `fig_4_8e_error_by_regime.png`. | Medium |
| Add residual normality test | `src/evaluation.py` | Shapiro-Wilk or QQ plot. | Low |
| Add baseline persistence forecast | `src/evaluation.py` | Implemented `persistence_baseline()`; exports `persistence_results.json`. | Medium |

### 16.5.6 Anomaly Detection

| Task | File(s) | Details | Priority |
|------|---------|---------|----------|
| Per-pollutant thresholds | `config.yaml`, `src/anomaly_detection.py` | Separate z/velocity thresholds for PM2.5 and PM10. | High |
| CUSUM grid search | `src/anomaly_calibration.py` | Sweep k/h and pick best F1 with FPR < 10%. | High |
| CUSUM decision plot | `src/anomaly_detection.py` | Exported `fig_4_6b_cusum_decision.png`. | Medium |
| Confusion matrix heatmaps | `src/anomaly_detection.py` | Exported `fig_4_6c_confusion_matrix_z.png` and `_cusum.png`. | Medium |
| Add detection-lag histogram | `src/anomaly_detection.py` | Export `fig_4_10a_detection_lag.png`. | Medium |
| Add alert rate control | `src/anomaly_detection.py` | Cap alerts per hour to avoid alert fatigue. | Low |

### 16.5.7 Risk Engine

| Task | File(s) | Details | Priority |
|------|---------|---------|----------|
| Sensitivity analysis | `src/risk_sensitivity.py` | Perturb coefficients by ±10% and report tier changes. | Medium |
| Add AQI tier mapping | `src/risk_engine.py` | Map EPA AQI categories to risk tiers for comparison. | Medium |
| Add heatmap of risk by PM and temp | `src/risk_engine.py` | Exported `fig_4_7b_risk_heatmap.png`. | Low |
| Add confidence intervals | `src/risk_engine.py` | Monte-Carlo or bootstrap CI on P_risk. | Low |

### 16.5.8 Clinical Validity

| Task | File(s) | Details | Priority |
|------|---------|---------|----------|
| Real VAS/TNSS loader | `src/validity.py` | Read `data/clinical/symptom_records.csv`. | High |
| Longitudinal validity plot | `src/validity.py` | Time series of risk tier vs. VAS/TNSS. | Medium |
| Add medication adjustment | `src/validity.py` | Exclude or flag medication days in correlation. | Medium |
| Add indoor/exposure adjustment | `src/validity.py` | Exclude indoor-flagged days. | Low |

### 16.5.9 Deployment & Reporting

| Task | File(s) | Details | Priority |
|------|---------|---------|----------|
| Latency instrumentation | `src/utils/timing.py`, `src/orchestrator.py` | Record per-phase timings. | High |
| Latency figure | `src/utils/timing.py` | Exported `fig_4_13_latency_breakdown.png`. | Medium |
| SUS/TAM survey loader | `src/validity.py` or new module | Read survey results and compute scores. | Medium |
| Thesis bundle exporter | `src/thesis_exporter.py` | Zip all tables and figures. | Low |
| HPO visualization | `src/hpo.py` | Exported `fig_4_4_hpo_optimization_history.png` and `fig_4_4b_hpo_param_importance.png`. | Medium |

### 16.5.10 Testing & Quality Assurance

| Task | File(s) | Details | Priority |
|------|---------|---------|----------|
| Performance regression test | `tests/test_performance_regression.py` | Pin MAPE/R² after full training. | High |
| Training augmentation tests | `tests/test_training_augmentation.py` | Jitter, window slice, mixup, walk-forward splits. | Medium |
| Evaluation extras tests | `tests/test_evaluation_extras.py` | Persistence baseline and error-by-regime. | Medium |
| HPO config tests | `tests/test_hpo_config.py` | Sampler/pruner validation. | Medium |
| Coverage target | `pytest.ini`, `coverage.py` | Aim for ≥ 80% line coverage. | Medium |
| Notebook smoke test in CI | `.github/workflows/` or local script | Run notebook with DO_TRAIN=False on a tiny subset. | Medium |
| Integration test for orchestrator | `tests/test_orchestrator.py` | Run `--mode train --quick` and assert artifacts exist. | Medium |

---

## 16.6 Specific Code & Config Changes

### 16.6.1 New config keys

```yaml
# Additions to config/config.yaml

model:
  # ... existing keys ...
  model_type: "bilstm"          # or "bigru", "cnn_lstm"
  use_attention: true
  attention_units: 64
  dropout_rate: 0.25
  recurrent_dropout: 0.25
  l2_lambda: 1.0e-4

training:
  # ... existing keys ...
  data_augmentation:
    enabled: false
    jitter_std: 0.01
    window_slice_prob: 0.1
  gap_handling:
    max_interpolated_frac: 0.2   # drop sequences with >20% interpolated rows
  walk_forward_cv:
    enabled: false
    n_splits: 5

anomaly:
  z_thresholds_pm2_5: [2.0, 3.0, 4.0]
  z_thresholds_pm10: [2.0, 3.0, 4.0]
  velocity_threshold_pm2_5: 15.0
  velocity_threshold_pm10: 15.0
  cusum_k_multiplier: 0.5
  cusum_h_multiplier: 4.0
  alert_rate_cap_per_hour: 6

risk:
  # ... existing keys ...
  sensitivity_perturbation_pct: 0.10

deployment:
  latency_log_enabled: true
  latency_log_path: "results/tables/table_4_13_latency.json"
```

### 16.6.2 New CLI flags

```bash
# Examples of new orchestrator flags to add
python src/orchestrator.py --mode full --seed 42 --epochs 200 --batch-size 32
python src/orchestrator.py --mode hpo --hpo-trials 50 --epochs 50 --seed 42
python src/orchestrator.py --mode evaluate --with-latency
python src/orchestrator.py --mode train --walk-forward-cv --n-splits 5
```

### 16.6.3 New modules to create

| Module | Responsibility |
|--------|----------------|
| `src/anomaly_calibration.py` | Grid search for CUSUM and z-score thresholds. |
| `src/interpretability.py` | Permutation importance, SHAP, attention-weight extraction. |
| `src/risk_sensitivity.py` | Perturbation analysis of risk coefficients. |
| `src/utils/timing.py` | Timing decorator/context manager. |
| `src/thesis_exporter.py` | Bundle all tables/figures into a zip. |
| `tests/test_performance_regression.py` | MAPE/R² regression test. |

### 16.6.4 Config migration checklist

- [ ] Back up current `config/config.yaml`.
- [ ] Add new keys under the appropriate sections.
- [ ] Update `src/config_loader.py` validation to check new keys.
- [ ] Update `docs/03_CONFIGURATION.md` with the new keys.
- [ ] Add migration note in `docs/12_IMPLEMENTATION_CHECKLIST.md`.

---

## 16.7 Acceptance Criteria & Definition of Done

A task is "done" when **all** of the following are true:

1. **Code** — The code compiles, runs, and is covered by at least one unit test.
2. **Artifacts** — The required JSON/PNG files are generated and saved in the correct location.
3. **Documentation** — The relevant `docs/*.md` file is updated with the new behavior.
4. **Reproducibility** — A fresh run with the same seed produces the same metrics within a small tolerance (e.g., ±0.01 MAPE).
5. **Thesis integration** — The figure/table is numbered, captioned, and referenced in the thesis text.

### Phase-gate checklist

Before moving from one phase to the next, confirm:

- [ ] Phase deliverables are present.
- [ ] Acceptance criteria for the phase are met.
- [ ] No new test failures were introduced.
- [ ] The improvement plan (this document) is updated.

---

## 16.8 Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Full training does not reach MAPE targets | Medium | High | Try HPO with more trials; add data augmentation; consider GRU/CNN-LSTM variants; increase sequence length or stride. |
| Local GPU memory or runtime limits | Medium | High | Use a GPU with more VRAM; split training into checkpoints via `src/pytorch_training.py`; use a cloud VM. |
| Real VAS/TNSS data is hard to collect | High | High | Start recruitment early; use a short survey app; offer incentives; plan for 4–6 weeks of data collection. |
| Anomaly calibration overfits the test set | Medium | Medium | Use a separate validation slice for threshold tuning; document the calibration procedure. |
| SHAP is too slow for 2016-step sequences | Medium | Low | Use permutation importance instead; or compute SHAP on a small subsample. |
| Deterministic mode hurts performance | Low | Low | Only enable determinism for the final reproducibility run, not for every experiment. |
| Thesis timeline slips | Medium | High | Prioritize Phases 1–3; Phases 4–6 can be compressed if needed. |

---

## 16.9 Timeline & Milestones

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | Local GPU setup + full data loaded | PyTorch sees CUDA, full dataset verified |
| 2 | Full training + HPO complete | `best_model.pt`, `optuna_best_params.json`, full training history |
| 3 | Anomaly recalibration | Updated `table_4_9_anomaly.json`, `table_4_10_cusum.json` |
| 4 | Clinical data collection begins | `symptom_records.csv` started |
| 5 | Risk engine sensitivity + latency instrumentation | `table_4_12b_risk_sensitivity.json`, `table_4_13_latency.json` |
| 6 | Interpretability + thesis bundle | `fig_4_8*`, `results/thesis_bundle.zip` |
| 7 | Defense preparation | `docs/19_DEFENSE_WALKTHROUGH.md`, final figures/tables |

*Note: Clinical data collection may take 4–6 weeks and should start as early as possible. The technical work (Phases 1–2) can proceed in parallel.*

---

## 16.10 Deliverables Checklist

Use this checklist to track what exists and what is still needed.

### Code & Config

- [x] `src/anomaly_calibration.py`
- [x] `src/interpretability.py`
- [x] `src/risk_sensitivity.py`
- [x] `src/utils/timing.py`
- [x] `src/thesis_exporter.py`
- [x] `src/target_scaling.py`
- [x] `Makefile`
- [x] `tests/test_performance_regression.py`
- [x] `tests/test_target_scaling.py`
- [x] `tests/test_training_augmentation.py`
- [x] `tests/test_evaluation_extras.py`
- [x] `tests/test_hpo_config.py`
- [x] `tests/test_anomaly_calibration.py`
- [x] `tests/test_risk_sensitivity.py`
- [x] `tests/test_interpretability.py`
- [x] `tests/test_thesis_exporter.py`
- [x] `tests/test_timing.py`
- [x] Updated `config/config.yaml` with new keys
- [x] Updated `src/config_loader.py` validation

### Model & Training

- [x] `model/best_model.pt` (5-month synthetic dataset)
- [x] `model/optuna_best_params.json`
- [x] `model/training_results.json` (full history)
- [x] `model/scaler.joblib`
- [x] `model/feature_cols.json` (177 features)

### Tables

- [x] `results/tables/table_4_8_bilstm.json` (5-month synthetic dataset)
- [x] `results/tables/arima_results.json`
- [x] `results/tables/persistence_results.json`
- [x] `results/tables/table_4_4_hpo_results.json`
- [x] `results/tables/table_4_8b_permutation_importance.json`
- [x] `results/tables/table_4_8c_error_by_regime.json`
- [x] `results/tables/table_4_9_anomaly.json`
- [x] `results/tables/table_4_10_cusum.json`
- [x] `results/tables/table_4_11_risk_distribution.json`
- [x] `results/tables/table_4_12_modifier_analysis.json`
- [x] `results/tables/table_4_12b_risk_sensitivity.json`
- [x] `results/tables/table_4_13_latency.json`
- [ ] `results/tables/table_4_14_sus_tam.json` (requires user study)
- [x] `results/tables/table_4_15_construct_validity.json` (synthetic placeholder)
- [x] `results/tables/table_4_8b_permutation_importance.json`
- [x] `results/tables/table_4_8c_error_by_regime.json`
- [x] `results/tables/table_4_4_hpo_results.json`
- [x] `results/tables/persistence_results.json`
- [x] `results/tables/table_4_9b_anomaly_calibration_pm2_5.json`
- [x] `results/tables/table_4_9b_anomaly_calibration_pm10.json`

### Figures

- [x] `results/visualizations/fig_4_4_hpo_optimization_history.png`
- [x] `results/visualizations/fig_4_4b_hpo_param_importance.png`
- [x] `results/visualizations/fig_4_5a_training_convergence.png`
- [x] `results/visualizations/fig_4_5b_bilstm_vs_arima.png`
- [ ] `results/visualizations/fig_4_5c_mae_comparison.png` (optional; not currently generated)
- [ ] `results/visualizations/fig_4_5d_error_diagnostics.png` (optional; not currently generated)
- [ ] `results/visualizations/fig_4_5e_error_by_hour_and_horizon.png` (optional; not currently generated)
- [x] `results/visualizations/fig_4_6_residual_distribution.png`
- [x] `results/visualizations/fig_4_6b_cusum_decision.png`
- [x] `results/visualizations/fig_4_6c_confusion_matrix_z.png`
- [x] `results/visualizations/fig_4_6c_confusion_matrix_cusum.png`
- [x] `results/visualizations/fig_4_7_risk_vs_pm25.png`
- [x] `results/visualizations/fig_4_7b_risk_heatmap.png`
- [x] `results/visualizations/fig_4_8b_permutation_importance.png`
- [ ] `results/visualizations/fig_4_8c_shap_summary.png` (optional; skipped if shap not installed)
- [x] `results/visualizations/fig_4_8d_attention_weights.png`
- [x] `results/visualizations/fig_4_8e_error_by_regime.png`
- [x] `results/visualizations/fig_4_11_risk_distribution.png`
- [ ] `results/visualizations/fig_4_11b_risk_summary.png` (optional; not currently generated)
- [x] `results/visualizations/fig_4_12b_risk_sensitivity.png`
- [x] `results/visualizations/fig_4_13_latency_breakdown.png`
- [x] `results/visualizations/fig_4_14_construct_validity.png`
- [ ] `results/visualizations/fig_4_14b_longitudinal_validity.png` (requires real symptom data)

### Documentation

- [x] `docs/16_IMPROVEMENT_PLAN.md` (this file)
- [ ] `docs/17_DEPLOYMENT_CHECKLIST.md` (not yet implemented)
- [ ] `docs/18_CODE_TO_THESIS_MAPPING.md` (not yet implemented)
- [ ] `docs/19_DEFENSE_WALKTHROUGH.md` (not yet implemented)
- [ ] `docs/20_REPRODUCIBILITY_STATEMENT.md` (not yet implemented)
- [x] Updated `docs/01_HARDWARE_AND_SOFTWARE.md` (local GPU/PyTorch environment)
- [x] Updated `README.md`
- [x] Updated `docs/10_EVALUATION.md`
- [x] Updated `docs/15_SYSTEM_SUMMARY_AND_IMPROVEMENTS.md`

---

## 16.11 Appendix: Snippets & Templates

### 16.11.1 Timing utility

```python
# src/utils/timing.py
import json
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any

@contextmanager
def timed_section(name: str, store: Dict[str, Any]):
    start = time.time()
    yield
    elapsed = time.time() - start
    store.setdefault(name, []).append(elapsed)

def export_latency_report(store: Dict[str, Any], out_path: Path) -> Path:
    report = {}
    for name, times in store.items():
        report[name] = {
            "mean_ms": round(1000 * sum(times) / len(times), 2),
            "p95_ms": round(1000 * sorted(times)[int(0.95 * len(times))], 2),
            "max_ms": round(1000 * max(times), 2),
            "n_samples": len(times),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path
```

### 16.11.2 CUSUM grid search

```python
# src/anomaly_calibration.py
import numpy as np
from typing import Dict, Any, Tuple

def grid_search_cusum(
    residuals: np.ndarray,
    k_multipliers: np.ndarray,
    h_multipliers: np.ndarray,
    truth_mask: np.ndarray,
) -> Tuple[float, float, Dict[str, Any]]:
    from src.anomaly_detection import cusum_detector, contingency_table

    best_f1 = -1.0
    best_k, best_h, best_metrics = None, None, None
    for km in k_multipliers:
        for hm in h_multipliers:
            alerts, k, h = cusum_detector(residuals, km, hm)
            metrics = contingency_table(alerts, truth_mask)
            if metrics["F1_Score"] > best_f1 and metrics["FalsePositiveRate_pct"] < 10.0:
                best_f1 = metrics["F1_Score"]
                best_k, best_h, best_metrics = k, h, metrics
    return best_k, best_h, best_metrics
```

### 16.11.3 Permutation importance

```python
# src/interpretability.py
import numpy as np
import torch
from typing import Dict, Any, List


def permutation_importance(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    baseline_mae: float,
    feature_names: List[str],
    n_repeats: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    importances = []
    for i, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            perm_idx = rng.permutation(X_perm.shape[0])
            X_perm[:, :, i] = X_perm[perm_idx, :, i]
            X_perm_t = torch.from_numpy(X_perm).float()
            with torch.no_grad():
                y_pred = model(X_perm_t).cpu().numpy()
            mae = float(np.mean(np.abs(y_val - y_pred)))
            drops.append(mae - baseline_mae)
        importances.append({
            "feature": name,
            "mae_drop_mean": float(np.mean(drops)),
            "mae_drop_std": float(np.std(drops)),
        })
    return {"importances": sorted(importances, key=lambda x: x["mae_drop_mean"], reverse=True)}
```

### 16.11.4 Performance regression test

```python
# tests/test_performance_regression.py
import json
from pathlib import Path

def test_mape_regression():
    """Fail if the pinned full-run MAPE drifts by more than 5%."""
    pinned = {
        "PM2_5_6h": 4.5,
        "PM2_5_12h": 6.8,
        "PM2_5_24h": 9.2,
        "PM10_6h": 4.8,
        "PM10_12h": 7.1,
        "PM10_24h": 9.8,
    }
    current = json.load(open("results/tables/table_4_8_bilstm.json"))["metrics"]
    for key, expected in pinned.items():
        pollutant, horizon = key.split("_", 1)
        actual = current[pollutant][horizon]["MAPE"]
        assert abs(actual - expected) < 5.0, f"{key} MAPE drifted: expected {expected}, got {actual}"
```

### 16.11.5 Thesis bundle exporter

```python
# src/thesis_exporter.py
import zipfile
from pathlib import Path
from typing import List

def export_thesis_bundle(root: Path, out_path: Path) -> Path:
    files: List[Path] = []
    for pattern in [
        "results/tables/*.json",
        "results/visualizations/*.png",
        "model/*.json",
        "model/*.joblib",
    ]:
        files.extend(root.glob(pattern))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(root))
    return out_path
```

---

## 16.12 How to Use This Plan

1. **Start with Phase 1.** Nothing else matters until the Bi-LSTM is trained on the full dataset.
2. **Update this document weekly.** Mark tasks as complete, add notes, and adjust timelines.
3. **Keep the acceptance criteria in mind.** Do not move to the next phase until the current phase meets its gate.
4. **Talk to your advisor early.** Share the gap analysis and the timeline so expectations are aligned.
5. **Version everything.** Use git tags for major milestones (e.g., `v1.0-full-training`, `v1.1-anomaly-recalibrated`).

---

*This plan is intentionally comprehensive. Pick the highest-priority items, execute them, and revisit this document often. The goal is not to do everything, but to do the things that make the thesis defensible.*
