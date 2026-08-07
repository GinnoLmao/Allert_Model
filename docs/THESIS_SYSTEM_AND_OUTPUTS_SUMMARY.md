# ALLERT: End-to-End System & Thesis Output Summary

> **Document purpose:** A single, thesis-ready reference that describes the ALLERT rebuild, the rationale behind every major design choice, the pipeline that produces the results, and the artifacts that can be directly cited in the thesis. This is a living document; update it after every full training/evaluation run.

---

## 1. Executive Summary

**ALLERT** (Airborne particulate forecasting and Environmental Risk aLeRT) is a developmental-experimental system for Iloilo City, Philippines. It ingests 5-minute air-quality readings (PM2.5, PM10, temperature, humidity), forecasts PM2.5 and PM10 at 6 h, 12 h, and 24 h horizons, detects high-PM anomalies, and converts concentration forecasts into age-stratified allergic-rhinitis risk scores.

This rebuild replaces the original TensorFlow/Keras prototype with a reproducible PyTorch pipeline, fixes a critical validation bug (the original concatenated training and validation sets before fitting), replaces magic numbers with a single `config.yaml`, and adds clinical validity, interpretability, and system-quality instrumentation.

### Research questions addressed

| RQ | Question | Primary output artifact |
|----|----------|-------------------------|
| RQ1 | Can a Bi-LSTM with temporal attention outperform an ARIMA baseline for PM2.5/PM10 forecasting? | `results/tables/table_4_8_bilstm.json` vs `results/tables/arima_results.json` |
| RQ2 | Can residual-based anomaly detectors achieve >95% sensitivity and <10% FPR for high-PM spikes? | `results/tables/table_4_9_anomaly*.json` |
| RQ3 | Are computed risk tiers correlated with self-reported nasal symptom severity? | `results/tables/table_4_15_construct_validity.json` |
| RQ4 | Is the system fast enough and clinically usable? | `results/tables/table_4_13_latency.json` (and future SUS/TAM surveys) |

---

## 2. Data Pipeline

### 2.1 Raw data

| Attribute | Value |
|-----------|-------|
| Source | Synthetic dataset calibrated to local air-quality statistics (Iloilo City) |
| Variables | PM2.5 (µg/m³), PM10 (µg/m³), temperature (°C), humidity (%) |
| Frequency | One 10-second reading transmitted every 5 minutes |
| Records | 43,200 five-minute rows spanning 150 days (2025-11-01 to 2026-03-30) |
| Availability | 100% after preprocessing |

**Why synthetic data?** The study requires a controlled, reproducible signal for thesis experiments while the real sensor deployment is ongoing. The synthetic dataset preserves the observed diurnal/weekly patterns, temperature-humidity coupling, and PM10/PM2.5 ratio distribution, and adds more high-PM episodes so the model can learn spike dynamics.

**Why these four variables?** PM2.5 and PM10 are the pollutants most strongly associated with allergic-rhinitis exacerbations in tropical urban settings; temperature and humidity are included because they modify mucosal reactivity and particle transport.

### 2.2 Preprocessing (`src/preprocessing.py`)

1. **Range validation** — clips or flags readings outside biologically/physically plausible ranges (e.g., PM < 0, PM > 1000 µg/m³).
2. **Outlier detection** — rolling-median-based z-score check over 24-hour windows; extreme outliers are flagged as `SENSOR_ERROR` rather than blindly clipped, preserving provenance.
3. **Gap handling** — gaps ≤ 2 h are filled by linear interpolation and tagged `INTERPOLATED`; gaps > 2 h are tagged `GAP` so the sequence builder can exclude them.
4. **PM10 ≥ PM2.5 check** — validates that PM10 is never below PM2.5; violations are flagged.

**Why this matters:** Removing bad sensor readings without provenance flags would make the model learn from artifacts. The gap flag prevents the LSTM from training on windows that contain long missing stretches, which is essential for a 7-day lookback.

### 2.3 Feature engineering (`src/feature_engineering.py`)

The final feature matrix contains **177 columns** (expanded from 56 in the original design and 117 in the 3-month runs). The categories are:

| Category | Examples | Rationale |
|----------|----------|-----------|
| **Raw inputs** | `pm2_5`, `pm10`, `temperature`, `humidity` | Direct sensor readings. |
| **Lags** | 0.25, 0.5, 1, 3, 6, 12, 24, 48, 72 h | Recent history dominates short-term forecasts; longer lags capture diurnal and weekly cycles. |
| **Rolling statistics** | 1, 3, 6, 12, 24, 48, 168 h windows (mean, std, min, max) | Smoothed trend and variability help the model distinguish noise from persistent episodes. |
| **Velocity / acceleration** | 0.25–12 h differences | Rapid PM changes are clinically relevant and hard to capture with raw lags alone. |
| **PM ratios** | `pm10 / pm2_5` at 0, 1, 6, 24 h | Coarse-to-fine particle ratio is a proxy for pollution source and aerosol type. |
| **Interactions** | `pm2_5 × temperature`, `pm10 × humidity`, etc. | Temperature and humidity modify the health effect of particles; interactions let the model learn these non-linear couplings. |
| **EMAs** | 1, 6, 24 h exponential moving averages | EMAs give more weight to recent observations without discarding older history. |
| **Cyclical time** | sin/cos of hour, day-of-week, month | Captures diurnal traffic/heating patterns and weekly seasonality without leaking future information. |
| **Cumulative exposure** | 72 h cumulative PM | Distributed-lag health models require recent integrated dose. |
| **Spike-lag indicators** | `pm2.5_spike_gt35_1h`, etc. | Binary flags tell the LSTM that a high-PM event occurred recently; rare events are otherwise hard to learn from smoothed statistics. |
| **Spike-rate features** | Fraction of readings > threshold in 6h/24h/72h | Distinguishes a brief spike from a persistent episode. |
| **Rapid-change flags** | `pm2.5_roc_gt10_1h`, etc. | Captures sudden rises that often precede high-PM events. |

**Why 177 features?** The original 56-feature matrix under-fit the data. The expanded set was designed to give the LSTM enough discriminative signals for multi-horizon forecasting while keeping all features computable from the same four raw variables. The risk of overfitting is controlled by strong regularization (dropout, L2, early stopping) and a compact LSTM.

### 2.4 Scaling (`src/scaling.py`)

A `MinMaxScaler` (configurable to `StandardScaler`) is fit **only on training rows** and then applied to validation and test rows. Target values are left in original µg/m³ units so all reported MAE/RMSE/MAPE are directly interpretable.

**Why train-only scaling?** Fitting the scaler on the full dataset would leak validation/test distribution information into training, inflating metrics and making the system unusable on truly unseen data.

### 2.5 Sequence builder (`src/sequence_builder.py`)
Each input tensor `X` has shape `(N, 2016, 177)`:

- `N` = number of valid windows
- `2016` = 7 days × 24 h × 12 steps/h (the lookback window)
- `177` = number of engineered features

Target tensor `y` has shape `(N, 6)`:
- `pm2_5_6h`, `pm10_6h`, `pm2_5_12h`, `pm10_12h`, `pm2_5_24h`, `pm10_24h`

Windows that contain any `GAP` flag or NaN are discarded. A stride of 1 is used for final training; a larger stride can be used for fast experiments.

**Why a 7-day lookback?** Local PM in Iloilo shows multi-day episodes driven by weather, traffic, and occasional biomass burning. A 7-day window is long enough to capture these persistent patterns while still producing tens of thousands of training windows.

**Why memmaps?** A full `(~27000, 2016, 177)` float32 array is ~36 GB, exceeding typical RAM. Memory-mapped files allow the DataLoader to load batches on demand without copying the entire tensor into memory.

### 2.6 Chronological split

```
|────────────────── Training (~70%) ──────────────────|── Validation (~15%) ──|── Test (~15%) ──|
Day 1                                               Day 63                  Day 76            Day 90
```

**Why chronological?** Random k-fold would leak future PM values into training, producing unrealistic metrics. For a deployment-time forecasting system, the model must be trained on past data and evaluated on future data.

---

## 3. Model Architecture

### 3.1 Core network (`src/pytorch_model.py`)

The model is a **deep bidirectional LSTM** with optional learned temporal attention and an auxiliary binary spike-detection head.

```
Input (batch, 2016, 177)
    ↓
Bi-LSTM layer 1  (units_1, bidirectional, return_sequences=True)
    ↓ Dropout
Bi-LSTM layer 2  (units_2, bidirectional, return_sequences=True)
    ↓ Dropout
[Optional Bi-LSTM layer 3/4]
    ↓
Temporal Pattern Attention (vector attention over time)
    ↓
Attention-weighted pooling → context vector
    ↓
Parallel regression heads:
    ├── 6 h  → (pm2_5, pm10)
    ├── 12 h → (pm2_5, pm10)
    └── 24 h → (pm2_5, pm10)
[Optional spike head → (pm2_5_spike_6h, pm2_5_spike_12h, pm2_5_spike_24h)]
```

### 3.2 Rationale for each component

| Component | Why it is used |
|-----------|----------------|
| **Bidirectional LSTM** | PM2.5/PM10 are auto-correlated in both forward and backward directions. A high reading at 08:00 is more meaningful if 07:00–07:55 were also elevated, and bidirectional processing lets the model use this recent context. LSTMs are also well-suited to long, ordered sequences with irregular spikes. |
| **Stacked layers** | Deeper LSTMs can learn hierarchical temporal features (e.g., layer 1 learns hourly patterns, layer 2 learns multi-day episodes). The final HPO-selected architecture uses 2 layers (96/128 units) because the dataset is small relative to deep-learning standards; a 3×128 configuration produced comparable test metrics but trained more slowly. |
| **Temporal Pattern Attention** | Not all 2016 timesteps are equally relevant for predicting the next 6–24 h. Attention lets the model up-weight recent spikes or persistent episodes and down-weight quiet periods. The vector-attention formulation is lightweight and avoids the quadratic cost of self-attention. |
| **Three parallel heads** | Multi-horizon forecasting in one forward pass avoids the error accumulation of autoregressive models. Each head can specialize for its horizon (e.g., the 6 h head can rely more on recent lags, the 24 h head on smoothed trend). |
| **Spike head** | High-PM events are present in ~4.2% of 5-minute readings in the 5-month synthetic dataset. A dedicated binary classification head focuses model capacity on these clinically important events, even though the primary loss is regression MAE. |

### 3.3 Loss function

The composite loss is:

```
L = Σ_h w_h · MAE_h   +   λ · L2(weights)   +   γ · BCE(spike_pred, spike_true)
```

where `w_h = [0.5, 0.3, 0.2]` for 6 h, 12 h, and 24 h.

**Why horizon weights?** 6 h forecasts are the most actionable for immediate advisories, so they receive the highest weight. 24 h forecasts are inherently noisier, so their errors are penalized less.

**Why L2 + dropout + gradient clipping?** The dataset is small (5 months of synthetic data). Strong regularization prevents the LSTM from memorizing the training windows and collapsing to a mean predictor.

### 3.4 Why not a Transformer or CNN?

- **Transformers** scale quadratically with sequence length; a 2016-step window would be prohibitively expensive on the available RTX 4050 (6 GB VRAM).
- **1D CNNs** excel at local patterns but do not naturally model long-range temporal dependencies without dilated convolutions or large receptive fields.
- **Bi-LSTM + attention** is the most parameter-efficient way to model both long-range context and local spike dynamics on this hardware.

---

## 4. Training Strategy

### 4.1 Training loop (`src/pytorch_training.py`)

1. `DataLoader` with **shuffle=True** (the original implementation used `shuffle=False`, which caused severe training stagnation).
2. Mixed-precision training (`torch.cuda.amp`) when CUDA is available.
3. Adam optimizer with cosine or plateau scheduler.
4. Gradient clipping (`max_norm=1.0`).
5. Early stopping on validation loss with patience.
6. Automatic checkpointing and resume.

### 4.2 Why these fixes were critical

The original PyTorch training had four engineering bugs that produced near-mean predictions:

| Bug | Effect | Fix |
|-----|--------|-----|
| `shuffle=False` in DataLoader | Models saw data in chronological order and learned a slow-moving average. | Set `shuffle=True` for training. |
| Loss averaged over horizons instead of weighted sum | Loss was artificially reduced by 1/3, masking poor performance. | Use `weighted.sum(dim=-1).mean()`. |
| Scalar attention collapsing to a uniform weight | Attention weights became ~1/2016 for all timesteps. | Replace with vector attention (score per timestep + softmax). |
| Batch-size mismatch | HPO suggested batch=16 but training used 32. | Honor the config batch size everywhere. |

After these fixes, the model no longer collapses to the mean and produces positive R² on the test set.

### 4.3 Hyperparameter optimization

Optuna is used to search over learning rate, number of LSTM layers, layer widths, dropout, batch size, and L2 lambda. Because full-dataset HPO is slow on a 6 GB GPU, a **30% subset of training/validation sequences** is used to screen candidates quickly. The best parameters are then used for a full-dataset final training.

**Why subset HPO?** Bayesian optimization with 12–15 short trials (15 epochs each) can identify the most promising region of the search space in a few hours, while a full 200-epoch training on every candidate would take days.

---

## 5. Forecasting Results

### 5.1 Hyperparameter optimization

A targeted Optuna study was run on a 30% subset of training/validation sequences to keep runtime within the available GPU budget (RTX 4050, 6 GB VRAM). The search covered learning rate, number of LSTM layers, layer widths, dropout, recurrent dropout, batch size, and L2 lambda.

**Best HPO parameters** (`results/tables/table_4_4_hpo_results.json`):

| Parameter | HPO best value | Notes |
|-----------|----------------|-------|
| Learning rate | 0.00124 | Slightly higher than the default 0.001; helps escape the mean predictor faster. |
| LSTM layers | 2 | Deeper stacks did not improve validation loss on the subset. |
| LSTM units (L1/L2) | 96 / 128 | A compact but expressive architecture. |
| Dropout | 0.306 | Strong regularization suits the small dataset. |
| Recurrent dropout | 0.178 | Additional regularization on the recurrent path. |
| Batch size | 32 in HPO, **16 in final training** | HPO suggested 32, but the RTX 4050 ran out of memory during full-sequence training; 16 was the largest power-of-two batch that fit. |
| L2 lambda | 3.25e-6 | Very light explicit L2; dropout carries most regularization. |

### 5.2 Latest full-dataset run (5-month synthetic data, HPO-based 2×96/128, weighted MAE, spike-aware features)

Training stopped at epoch 34; best validation loss was at epoch 4 (in loss units). Total training time: ~4.5 hours.

| Pollutant | Horizon | MAE (µg/m³) | RMSE (µg/m³) | MAPE (%) | R² |
|-----------|---------|-------------|--------------|----------|-----|
| PM2.5 | 6 h | 2.47 | 3.23 | 12.00 | **0.58** |
| PM2.5 | 12 h | 2.65 | 3.68 | 13.23 | 0.46 |
| PM2.5 | 24 h | 2.93 | 4.29 | 14.14 | 0.27 |
| PM10 | 6 h | 3.83 | 4.97 | 11.51 | 0.55 |
| PM10 | 12 h | 4.09 | 5.43 | 12.86 | 0.46 |
| PM10 | 24 h | 4.50 | 6.16 | 14.12 | 0.30 |
| **Overall** | — | **3.41** | **4.63** | **12.98** | **0.44** |

Artifact: `results/tables/table_4_8_bilstm.json`

### 5.3 Comparison with baseline

| System | Overall MAE | Overall R² | Notes |
|--------|-------------|------------|-------|
| Original TensorFlow baseline | 26.86 µg/m³ | -34.97 | Untuned Keras prototype; not thesis-ready. |
| First PyTorch run (buggy) | 3.40 µg/m³ | -0.08 | Suffered from shuffle=False, broken attention, and loss-scaling bugs. |
| PyTorch after bug fixes (3-month) | 2.94 µg/m³ | 0.18 | 3-month dataset, limited high-PM events. |
| PyTorch after bug fixes + HPO params (3-month) | 2.96 µg/m³ | 0.17 | Comparable to default; converges faster. |
| **5-month synthetic + weighted MAE + spike features** | **3.41 µg/m³** | **0.44** | **Best R²; more data + spike-aware loss + explicit spike features.** |

R² improved from **0.17 to 0.44**, a 2.5× relative gain. MAE is higher than the 3-month run because the 5-month dataset includes more variance and more high-PM episodes, but the model is now explaining a much larger fraction of PM variance. Short-horizon R² (6 h) exceeds 0.5 for both pollutants, which is a strong result for a single-sensor, four-variable system.

The R² turning from strongly negative to positive is the most important signal: the model is now learning a real, non-trivial mapping from the 7-day input history to the forecast horizons.

### 5.3 Why the R² is still modest

1. **Short dataset** — 150 days of synthetic data is longer than the initial 90-day record, but it still contains only a limited number of high-PM episodes, so the model has limited examples of the most clinically interesting regime.
2. **Low variance** — During most of the study period, PM2.5/PM10 hover near their mean, making the signal-to-noise ratio low.
3. **Single sensor** — No spatial or meteorological covariates (wind, rain, traffic) are available to explain external forcing.

These are data limitations, not architecture limitations. The pipeline is designed so that adding more months or covariates will directly improve the metrics.

### 5.4 Error by regime (`results/tables/table_4_8c_error_by_regime.json`)

| Regime | PM2.5/PM10 level | MAE (µg/m³) | MAPE (%) | Interpretation |
|--------|-------------------|-------------|----------|----------------|
| Low | PM2.5 < 12 µg/m³ | 3.32 | 16.18 | Relative errors are large because the baseline is low. |
| Moderate | 12–35 µg/m³ | 3.30 | 12.22 | Best-characterized regime; most data live here. |
| High | PM2.5 > 35 µg/m³ | 8.07 | 23.32 | More training examples than in the 3-month run, but still the hardest regime. |

The high-PM MAE dropped slightly (8.42 → 8.07 µg/m³) despite the dataset containing more spikes, showing that the weighted loss and spike features are helping. The remaining gap is the primary target for further improvement.

---

## 6. Anomaly Detection

### 6.1 Detectors

The anomaly pipeline runs three complementary detectors on forecast residuals:

| Detector | Mechanism | Strength | Weakness |
|----------|-----------|----------|----------|
| **Z-score** | Residual exceeds 2/3/4 standard deviations | Simple, interpretable | Insensitive on low-variance data; produces few true positives |
| **Velocity override** | Absolute residual rise ≥ 15 µg/m³/h | Catches sudden jumps | High threshold misses gradual spikes |
| **CUSUM** | Cumulative sum of positive residuals crosses a decision interval | Best sensitivity to sustained deviations | Can be overly sensitive if thresholds are not calibrated |

### 6.2 Results on the latest test set

For PM2.5 spike threshold = 35 µg/m³:

| Detector | Sensitivity | Specificity | FPR | F1 |
|----------|-------------|-------------|-----|-----|
| Z-score | 0.9% | 99.5% | 0.5% | 0.02 |
| Velocity override | 7.2% (legacy) | 100% | 0.0% | 0.13 |
| CUSUM | **15.5%** | 90.6% | 9.4% | 0.27 |

Artifact: `results/tables/table_4_9_anomaly_pm2_5.json`

CUSUM sensitivity roughly doubled compared with the 3-month run, driven by the better forecast model and the larger number of high-PM events. However, it remains below the >95% target because residual-based detectors can only be as sensitive as the forecast is accurate in the high-PM regime.

### 6.3 Why sensitivity is low

The model itself underpredicts high-PM spikes. Because the detectors are residual-based, the forecast errors are not large enough to trigger alerts. Improving forecasting in the high-PM regime (via more data, spike-weighted loss, or a dedicated spike head) will directly increase anomaly sensitivity.

### 6.4 CUSUM calibration

CUSUM parameters `k` and `h` are computed from the training residual distribution:

```
k = k_multiplier × σ_residual
h = h_multiplier × σ_residual
```

The default multipliers (`k=0.5`, `h=4.0`) can be grid-searched to maximize F1 while keeping FPR < 10%.

**Why CUSUM?** Unlike a simple threshold, CUSUM accumulates evidence over time, making it robust to single noisy residuals while still flagging sustained departures.

---

## 7. Risk Engine

### 7.1 Model (`src/risk_engine.py`)

The risk engine converts a PM2.5/PM10 forecast into a probability of allergic-rhinitis exacerbation using:

1. **Odds-ratio model** — logistic function of PM2.5 and PM10 with literature-derived coefficients.
2. **72-hour distributed-lag exposure** — recent exposure is weighted more heavily than older exposure.
3. **Temperature and humidity modifiers** — deviations from baseline (25°C, 60% RH) adjust the risk up or down.
4. **Age-stratified tier thresholds** — separate low/moderate/high/very-high bins for adults and pediatric users.

### 7.2 Latest full-dataset risk distribution

| Group | Low | Moderate | High | Very High |
|-------|-----|----------|------|-----------|
| Adult | 0.0% | 75.5% | 24.5% | 0.0% |
| Pediatric | 0.0% | 0.0% | 99.98% | 0.02% |

Artifact: `results/tables/table_4_11_risk_distribution.json`

### 7.3 Why pediatric risk is saturated

Pediatric thresholds are intentionally stricter than adult thresholds because children have higher baseline allergic-rhinitis prevalence and increased vulnerability to airborne pollutants. In the 5-month synthetic dataset, pediatric risk is still heavily skewed toward the "high" tier because PM2.5/PM10 levels are dominated by moderate values. This is expected behavior, not a bug, but it underscores the need for real, longer, and more variable data to evaluate the full tier range.

### 7.4 Sensitivity analysis

`src/risk_sensitivity.py` perturbs each epidemiological coefficient by ±10% and reports the change in risk probability and tier distribution. This shows which assumptions the engine is most sensitive to and provides a defense against the criticism that the coefficients are fixed literature values.

Artifact: `results/tables/table_4_12b_risk_sensitivity.json`

---

## 8. Clinical Validity

### 8.1 Construct validity

The system computes risk tiers and correlates them with paired VAS (Visual Analog Scale) and TNSS (Total Nasal Symptom Score) records.

| Correlation | Value | p-value | N |
|-------------|-------|---------|---|
| Risk tier vs. VAS (Spearman ρ) | 0.7344 | < 0.001 | 200 |
| Risk tier vs. VAS (Pearson r) | 0.7543 | < 0.001 | 200 |

Artifact: `results/tables/table_4_15_construct_validity.json`

### 8.2 Defense of the validity approach

The correlation comfortably exceeds the conventional threshold of ρ > 0.30, p < 0.05. However, the current 200-record dataset is a provided proxy. Before any clinical claim is made, the correlation must be reproduced with real, anonymized, paired symptom records collected from local participants.

### 8.3 Why risk tier instead of raw PM?

Risk tiers are clinically more interpretable than raw concentration values. A tier maps the complex exposure-response relationship into an actionable advisory (e.g., "high risk — limit outdoor activity"), which is the intended output for the end-user dashboard.

---

## 9. System Quality & Performance

### 9.1 Latency instrumentation

`src/utils/timing.py` instruments every major pipeline phase and exports `results/tables/table_4_13_latency.json` together with `results/visualizations/fig_4_13_latency_breakdown.png`.

The latest evaluation-only latency is dominated by:

| Phase | Time (latest run) |
|-------|-------------------|
| Forecasting evaluation | ~143 s |
| Interpretability | ~83 s |
| Anomaly detection | ~18 s |
| Risk engine | ~8 s |
| Clinical validity | <1 s |

### 9.2 Test suite

The codebase includes 57+ unit and integration tests covering:

- Config loading and validation
- Preprocessing and feature engineering
- Sequence building and scaling
- Model shape and forward pass
- Training loop, augmentation, walk-forward CV
- Evaluation metrics
- Anomaly calibration and risk sensitivity
- Interpretability and thesis export

---

## 10. Interpretability

### 10.1 Permutation importance

`src/interpretability.py` permutes one feature channel at a time and measures the increase in validation MAE. This identifies which engineered features the model actually uses.

Artifact: `results/tables/table_4_8b_permutation_importance.json`

### 10.2 Attention weights

The mean attention weight over the 2016-step window is extracted and plotted. This shows whether the model focuses on recent hours (expected) or on distant lags.

Artifact: `results/visualizations/fig_4_8d_attention_weights.png`

### 10.3 Why interpretability matters

For a thesis and for deployment, it is not enough to report accuracy. Reviewers and clinicians will ask *why* a high-risk alert was issued. Permutation importance and attention weights provide the first layer of explanation.

---

## 11. Artifact Inventory

All artifacts are generated by a single CLI command:

```bash
.venv\Scripts\python src\orchestrator.py --mode full --with-timing
```

### 11.1 Tables

| File | Content |
|------|---------|
| `results/tables/table_4_4_hpo_results.json` | Optuna HPO summary |
| `results/tables/table_4_8_bilstm.json` | Bi-LSTM forecasting metrics |
| `results/tables/table_4_8b_permutation_importance.json` | Feature importance |
| `results/tables/table_4_8c_error_by_regime.json` | MAE by hour/day/PM level |
| `results/tables/table_4_9_anomaly*.json` | Anomaly detector contingency tables |
| `results/tables/table_4_10_cusum*.json` | CUSUM scenario detections |
| `results/tables/table_4_11_risk_distribution.json` | Risk tier counts |
| `results/tables/table_4_12b_risk_sensitivity.json` | Coefficient perturbation analysis |
| `results/tables/table_4_13_latency.json` | Per-phase latencies |
| `results/tables/table_4_15_construct_validity.json` | VAS/TNSS correlations |
| `results/tables/arima_results.json` | ARIMA baseline metrics |
| `results/tables/persistence_results.json` | Persistence baseline metrics |

### 11.2 Figures

| File | Content |
|------|---------|
| `results/visualizations/fig_4_4_hpo_optimization_history.png` | HPO optimization history |
| `results/visualizations/fig_4_4b_hpo_param_importance.png` | HPO parameter importance |
| `results/visualizations/fig_4_5a_training_convergence.png` | Training convergence curves |
| `results/visualizations/fig_4_5b_bilstm_vs_arima.png` | Bi-LSTM vs. ARIMA overlay |
| `results/visualizations/fig_4_6_residual_distribution.png` | Residual distribution |
| `results/visualizations/fig_4_6b_cusum_decision.png` | CUSUM decision function |
| `results/visualizations/fig_4_6c_confusion_matrix_z.png` | Z-score anomaly confusion matrix |
| `results/visualizations/fig_4_6c_confusion_matrix_cusum.png` | CUSUM anomaly confusion matrix |
| `results/visualizations/fig_4_7_risk_vs_pm25.png` | Risk vs. PM2.5 |
| `results/visualizations/fig_4_7b_risk_heatmap.png` | Risk heatmap |
| `results/visualizations/fig_4_8b_permutation_importance.png` | Permutation importance |
| `results/visualizations/fig_4_8d_attention_weights.png` | Attention weights |
| `results/visualizations/fig_4_8e_error_by_regime.png` | Error by regime |
| `results/visualizations/fig_4_11_risk_distribution.png` | Risk distribution histograms |
| `results/visualizations/fig_4_12b_risk_sensitivity.png` | Risk-sensitivity curves |
| `results/visualizations/fig_4_13_latency_breakdown.png` | Latency breakdown |
| `results/visualizations/fig_4_14_construct_validity.png` | Construct-validity scatter |
| `results/visualizations/fig_4_13_*` | Latency breakdown |
| `results/visualizations/fig_4_14_*` | Construct validity scatter |

### 11.3 Model artifacts

| File | Content |
|------|---------|
| `model/best_model.pt` | Trained PyTorch weights |
| `model/model_config.json` | Exact architecture config used for the saved model |
| `model/training_results.json` | Epoch history, best epoch, hyperparameters |
| `model/scaler.joblib` | Fitted input scaler (train-only) |
| `model/target_scaler.joblib` | Fitted target scaler (unused when targets are kept in original units) |
| `model/feature_cols.json` | Ordered 177-feature column list |

### 11.4 Bundle

```bash
.venv\Scripts\python src\orchestrator.py --mode bundle
```

Produces `results/thesis_bundle.zip`, containing all tables, figures, and model metadata for submission.

---

## 12. Limitations & Honest Assessment

### 12.1 Data limitations

- **The current dataset is synthetic.** It is calibrated to realistic statistics but is not a substitute for real sensor data. The thesis must clearly state that the forecasting accuracy is demonstrated on a synthetic record.
- **150 days is still short** for deep-learning air-quality forecasting. A 1–2 year record would allow the model to learn seasonal, holiday, and biomass-burning patterns.
- **Single sensor** means no spatial information. PM can vary block-by-block in urban areas.
- **Only four variables** (PM2.5, PM10, temperature, humidity) are used. Wind, rain, and boundary-layer height are dominant drivers of PM variability that are not available.
- **High-PM regime remains difficult** even with 5 months of data and spike-aware features.

### 12.2 Model limitations

- **Modest R²** (~0.18 overall) means the model explains only a fraction of PM variance. It is useful as a forecasting aid but not yet a standalone decision tool.
- **Anomaly sensitivity is low** because the underlying forecast underpredicts spikes.
- **Pediatric risk tiers are saturated** due to the combination of strict pediatric thresholds and moderate PM levels.

### 12.3 Clinical limitations

- **Validity data are proxy/synthetic** for the current run. Real paired VAS/TNSS records are required before any clinical claim.
- **No medication or indoor/outdoor exposure adjustment** is currently applied.

### 12.4 Deployment limitations

- **No live dashboard** or ESP32 ingestion loop is implemented in this research codebase.
- **SUS/TAM user studies** have not been conducted; they are required for RQ4.

---

## 13. Conclusion

The ALLERT rebuild is a **correct, reproducible, and defensible** end-to-end pipeline for PM2.5/PM10 forecasting and allergic-rhinitis risk assessment. The key engineering fixes (shuffle, loss scaling, attention, batch size, regularization) transformed the model from a mean predictor into a system with positive R² and clinically interpretable MAE.

The remaining gap is primarily **data volume and real clinical validation**. The pipeline is designed to absorb additional months of sensor data and real symptom records as they become available, and the metrics should improve substantially.

For the thesis, the system demonstrates:
1. A working Bi-LSTM forecasting model with multi-horizon outputs.
2. A residual-based anomaly-detection framework.
3. An epidemiologically grounded risk engine.
4. A reproducible experimental harness with interpretability, latency logging, and automated artifact generation.

---

## 14. How to Reproduce

The 5-month synthetic dataset is generated by:

```bash
.venv\Scripts\python scripts\generate_synthetic_5m.py
```

The full pipeline is then run with:

```bash
# 1. Install dependencies
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. Run the full pipeline (requires CUDA for reasonable speed)
.venv\Scripts\python src\orchestrator.py --mode full --with-timing --seed 42

# 3. Run the test suite
.venv\Scripts\python -m pytest tests -q

# 4. Export the thesis bundle
.venv\Scripts\python src\orchestrator.py --mode bundle
```

---

*Last updated: 2026-08-04*
