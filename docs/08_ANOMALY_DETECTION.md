# 08 — Anomaly Detection

The anomaly detection pipeline runs in parallel with the scheduled Bi-LSTM forecaster to detect sudden or gradual pollution events that require immediate alerts.

## 8.1 Dual-Mode Architecture

```
Real-time Sensor Stream (every 5 minutes)
    ├─→ Branch A: Anomaly Detection
    │     ├── Z-score residual monitor
    │     ├── Velocity override
    │     └── CUSUM drift detector
    └─→ Branch B: Scheduled Bi-LSTM Forecasting
           ↓
      Merged Alert Queue
```

An anomaly triggers an immediate spike alert, bypassing the scheduled forecast delay.

## 8.2 Method 1: Z-Score Residual Analysis

### Step 1: Compute Residual

```python
e_t = abs(x_actual_t - x_forecast_t)
```

### Step 2: Rolling Residual Statistics

```python
window_steps = 24 * 12  # 24 hours at 5-min resolution
mu_e = rolling_mean(e_t, window_steps)
sigma_e = rolling_std(e_t, window_steps)
```

### Step 3: Compute Z-Score

```python
Z_t = (e_t - mu_e) / sigma_e
```

### Step 4: Classify Tier

| Z Range | Tier | Action |
|---------|------|--------|
| Z < 2.0 | Normal | Continue monitoring |
| 2.0 ≤ Z < 3.0 | Spike Watch | Log and monitor |
| 3.0 ≤ Z < 4.0 | Spike Warning | Immediate alert |
| Z ≥ 4.0 | Spike Emergency | Immediate alert |

## 8.3 Method 2: Velocity Override

A hard-coded emergency override catches violent sudden spikes before Z-score statistics stabilize.

```python
velocity_hourly = (PM_t - PM_{t-1}) * 12   # 5-min step → hourly rate

if velocity_hourly > 15.0:   # µg/m³/hour
    trigger_velocity_alert(minimum_tier="Spike Warning")
```

## 8.4 Method 3: CUSUM Drift Detection

CUSUM catches slow, sustained deteriorations that single-point Z-scores miss.

```python
sigma_e = rolling_std(residuals, window)
k = 0.5 * sigma_e
h = 4.0 * sigma_e

S_i = max(0, S_{i-1} + e_i - k)

if S_i > h:
    trigger_cusum_alert()
    S_i = 0   # reset after alert
```

| Parameter | Formula | Purpose |
|-----------|---------|---------|
| `k` | 0.5 × σ_e | Allowable slack |
| `h` | 4.0 × σ_e | Decision interval |

## 8.5 Integration with Alert Generation

```python
if Z_t >= 3.0 or velocity > 15.0 or S_i > h:
    tier = classify_anomaly_tier(Z_t, velocity, S_i)
    if anomaly_tier > current_scheduled_tier:
        escalate_to_anomaly_tier()
    generate_immediate_spike_alert()
else:
    continue_monitoring()
```

## 8.6 Evaluation Targets

| Metric | Formula | Target |
|--------|---------|--------|
| Sensitivity | TP / (TP + FN) | > 95% |
| False Positive Rate | FP / (FP + TN) | < 10% |
| Alert latency | sensor reading → notification | < 30 seconds |

## 8.7 Ground Truth Labeling

An anomaly event is labeled when:
- PM2.5 or PM10 exceeds WHO 24h guideline (15 µg/m³), **or**
- PM velocity exceeds 15 µg/m³/hour, **and**
- Verified against DENR-EMB reference data where available.

## 8.8 Required Output: Table 4.9

Export `results/tables/table_4_9_anomaly.json`:

```json
{
  "z_score_threshold": 3.0,
  "velocity_threshold_ugm3_per_hour": 15.0,
  "cusum_k": 0.25,
  "cusum_h": 2.0,
  "labeled_events_total": 100,
  "contingency": {"TP": 95, "FP": 8, "TN": 292, "FN": 5},
  "derived_metrics": {
    "sensitivity_pct": 95.0,
    "specificity_pct": 97.3,
    "false_positive_rate_pct": 2.7,
    "precision_pct": 92.2,
    "f1_score": 0.936
  }
}
```

## 8.9 Required Output: Table 4.10

Export `results/tables/table_4_10_cusum.json`:

```json
[
  {
    "scenario_id": 1,
    "anomaly_type": "gradual_drift",
    "onset_timestamp": "2026-02-01T08:00:00",
    "cusum_detection_timestamp": "2026-02-01T09:30:00",
    "z_score_detection_timestamp": "2026-02-01T10:00:00",
    "cusum_detection_delay_minutes": 90,
    "z_score_detection_delay_minutes": 120,
    "peak_z_score": 4.5,
    "peak_cusum_S": 5.2,
    "correctly_detected": true
  }
]
```

## 8.10 Operational Considerations

### Alert Throttling
- Dispatch alerts only on distinct tier escalations.
- Suppress repeated alerts within the same spike event.
- Respect quiet hours for Watch/Advisory tiers; Warning/Emergency override quiet hours.

### Spike Event Duration Tracking
- If Z-score remains above 2.0 for > 3 consecutive readings (15 minutes), report a sustained spike.
- Resolution is noted when Z-score returns below 2.0 for 3 consecutive readings.

## 8.11 Risks

| Risk | Mitigation |
|------|------------|
| Residual baseline unstable early in deployment | 24–48 hour monitoring-only mode before live alerts |
| Model degradation inflates residuals globally | Periodic retraining |
| Velocity-only false positives | Conservative 15 µg/m³/h threshold; tune locally |
| CUSUM k/h sensitivity | Empirically calibrate on historical spikes |

---

*Next: read [`09_RISK_ENGINE.md`](09_RISK_ENGINE.md) to convert forecasts into clinical risk.*
