# 09 — Risk Engine

The risk engine translates LSTM-predicted PM concentrations into a clinically interpretable Allergic Rhinitis (AR) probability, using epidemiological odds ratios, meteorological modifiers, and a 72-hour distributed lag.

## 9.1 Pipeline Overview

```
LSTM Forecast Output (PM2.5, PM10, Temp, RH)
    ↓
1. Compute PM Risk Score (log-odds)
    ↓
2. Apply Temperature Modifier
    ↓
3. Apply Humidity Modifier
    ↓
4. Compute Combined Odds Ratio (OR_total)
    ↓
5. Convert OR to Risk Probability (P_risk)
    ↓
6. Apply 72-Hour Distributed Lag Weighting
    ↓
7. Apply Age Stratification
    ↓
8. Classify into Risk Tier
    ↓
Output: Risk Tier + P_risk (%) + Clinical Advisory
```

## 9.2 Step 1: Baseline PM Risk Score

```python
Risk_PM = (PM2.5 × 0.0104) + (PM10 × 0.0026)
```

| Pollutant | Coefficient | Basis |
|-----------|-------------|-------|
| PM2.5 | 0.0104 | ln(OR) per 1 µg/m³ increase |
| PM10 | 0.0026 | ln(OR) per 1 µg/m³ increase |

This produces a **log-odds score** representing pollutant-attributable increase in AR risk.

## 9.3 Step 2: Temperature Modifier

```python
M_temp = (T - 25) × 0.012
```

- Baseline: 25°C
- Positive above baseline: heat increases airway reactivity
- Negative below baseline: slight risk reduction

## 9.4 Step 3: Humidity Modifier

```python
M_humidity = (RH - 60) × (-0.004)
```

- Baseline: 60% RH
- Positive below baseline: dry air concentrates particles
- Negative above baseline: humidity is slightly protective

## 9.5 Step 4: Combined Odds Ratio

```python
OR_total = exp(Risk_PM + M_temp + M_humidity)
```

## 9.6 Step 5: Convert OR to Absolute Risk Probability

```python
P_risk = [(P_baseline × OR_total) / (1 + P_baseline × (OR_total - 1))] × 100
```

| Age Group | P_baseline | Rationale |
|-----------|------------|-----------|
| Adult (≥18) | 0.20 | Philippine adult AR prevalence (NNHS 2008) |
| Pediatric (<18) | 0.22 | Higher pediatric susceptibility |

## 9.7 Step 6: 72-Hour Continuous Rolling Distributed Lag

The continuous rolling risk at 5-minute granularity:

```python
Risk_continuous(t) = Σ w(τ) × P_risk(t - τ)   for τ = 0..864
```

| τ Range (steps) | Hours | Weight Allocation | Per-step Weight |
|-----------------|-------|-------------------|-----------------|
| 0 – 288 | 0–24h | 60% | 0.60 / 289 ≈ 0.002076 |
| 289 – 576 | 24–48h | 25% | 0.25 / 288 ≈ 0.000868 |
| 577 – 864 | 48–72h | 10% | 0.10 / 288 ≈ 0.000347 |
| τ = 864 | 72h boundary | 5% | 0.05 |

**Critical assertion:** `abs(sum(weights) - 1.0) < 1e-12`

**Validation:** With constant `P_risk(t) = C`, `Risk_continuous(t)` must equal `C`.

## 9.8 Step 7 & 8: Age-Stratified Tier Classification

### Adult Thresholds

| Tier | P_risk Range | Action |
|------|--------------|--------|
| Low | < 15% | Normal activity |
| Moderate | 15–25% | Monitoring; preventive action considered |
| High | 25–40% | Limit outdoor exposure |
| Very High | > 40% | Emergency alert; indoor shelter |

### Pediatric Thresholds

| Tier | P_risk Range | Action |
|------|--------------|--------|
| Low | < 13% | Normal activity |
| Moderate | 13–23% | Monitoring; preventive action |
| High | 23–38% | Limit outdoor exposure |
| Very High | > 38% | Emergency alert; indoor shelter |

Pediatric thresholds are shifted 2 percentage points lower to reflect higher physiological vulnerability.

## 9.9 Worked Example

Given:
- PM2.5 = 35 µg/m³
- PM10 = 60 µg/m³
- Temperature = 30°C
- Humidity = 70%
- Adult user

```
Risk_PM = (35 × 0.0104) + (60 × 0.0026) = 0.364 + 0.156 = 0.520
M_temp = (30 - 25) × 0.012 = 0.060
M_humidity = (70 - 60) × (-0.004) = -0.040
OR_total = exp(0.520 + 0.060 - 0.040) = exp(0.540) ≈ 1.716
P_risk = [(0.20 × 1.716) / (1 + 0.20 × (1.716 - 1))] × 100
       = [0.3432 / 1.1432] × 100
       ≈ 30.0%
```

**Tier:** HIGH (25–40% for adults) → Limit outdoor exposure.

## 9.10 Required Outputs

### Table 4.11: Risk Distribution

`results/tables/table_4_11_risk_distribution.json`:

```json
{
  "total_calculations": 10000,
  "adult": {
    "Low": {"count": 6520, "pct": 65.2},
    "Moderate": {"count": 2210, "pct": 22.1},
    "High": {"count": 980, "pct": 9.8},
    "Very_High": {"count": 290, "pct": 2.9}
  },
  "pediatric": {
    "Low": {"count": 5850, "pct": 58.5},
    "Moderate": {"count": 2530, "pct": 25.3},
    "High": {"count": 1120, "pct": 11.2},
    "Very_High": {"count": 500, "pct": 5.0}
  }
}
```

### Table 4.12: Modifier Analysis

`results/tables/table_4_12_modifier_analysis.json`:

```json
{
  "temperature": {
    "observed_range_C": {"min": 24.0, "max": 36.0, "mean": 29.5},
    "M_temp_range": {"min": -0.012, "max": 0.132, "mean": 0.054},
    "delta_P_risk_from_temp_alone": {"min_pct": -0.2, "max_pct": 2.5, "mean_pct": 1.1}
  },
  "humidity": {
    "observed_range_RH": {"min": 55.0, "max": 92.0, "mean": 74.0},
    "M_humidity_range": {"min": 0.02, "max": -0.128, "mean": -0.056},
    "delta_P_risk_from_humidity_alone": {"min_pct": -1.2, "max_pct": 0.4, "mean_pct": -0.5}
  }
}
```

## 9.11 Risks and Limitations

| Limitation | Implication |
|------------|-------------|
| Population-level probability, not individual prediction | Cannot predict a specific patient's symptoms |
| Coefficients from published literature | May not perfectly match Iloilo City pollution profile |
| Linear temperature/humidity modifiers | Real relationships may be non-linear at extremes |
| 72h lag weights are stylized | Local lag dynamics may differ |

## 9.12 Future Extensibility

- Bayesian updating of beta coefficients with local symptom data
- Non-linear meteorological modifiers
- Integration of pollen/mold spore counts
- Personalized risk calibration per user

---

*Next: read [`10_EVALUATION.md`](10_EVALUATION.md) for metrics and baselines.*
