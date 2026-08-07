# 04 — Data Pipeline

The data pipeline transforms raw IoT sensor readings into a clean, chronologically valid, analysis-ready dataset. It is the foundation of every downstream module.

## 4.1 Input Data

| Variable | Unit | Source | Valid Range |
|----------|------|--------|-------------|
| PM2.5 | µg/m³ | Sensirion SPS30 | 0 – 1,000 |
| PM10 | µg/m³ | Sensirion SPS30 | 0 – 1,000 |
| Temperature | °C | Sensirion SHT40 | -10 – 60 |
| Humidity | %RH | Sensirion SHT40 | 0 – 100 |

## 4.2 Pipeline Overview

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

## 4.3 Step-by-Step Logic

### Step 1: Load and Sort

```python
df = pd.read_csv(path, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df = df.drop_duplicates(subset=["timestamp"], keep="first")
```

Validation checks:
- No future timestamps
- Monotonically increasing timestamps after sort
- No duplicate timestamps

### Step 2: Range Validation

Readings outside physically possible ranges are **flagged, not dropped**:

```python
RANGE_LIMITS = {
    "pm2_5": (0, 1000),
    "pm10": (0, 1000),
    "temperature": (-10, 60),
    "humidity": (0, 100),
}

for col, (lo, hi) in RANGE_LIMITS.items():
    out_of_range = (df[col] < lo) | (df[col] > hi)
    df.loc[out_of_range, "quality_flag"] = "OUT_OF_RANGE"
```

### Step 3: Outlier Detection

Use a rolling Z-score against a 24-hour window. A conservative threshold of 5.0 avoids removing legitimate pollution spikes.

```python
window_steps = 24 * 12  # 24 hours at 5-min resolution
for col in ["pm2_5", "pm10"]:
    rolling_mean = df[col].rolling(window=window_steps, min_periods=1).mean()
    rolling_std = df[col].rolling(window=window_steps, min_periods=1).std().replace(0, np.nan)
    z_score = (df[col] - rolling_mean) / rolling_std
    df.loc[z_score.abs() > 5.0, "quality_flag"] = "SENSOR_ERROR"
```

### Step 4: Gap Classification and Filling

```python
GAL_FILL_THRESHOLD_HOURS = 2
threshold_steps = GAL_FILL_THRESHOLD_HOURS * 12

# Detect gaps
df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 60.0  # minutes
missing_steps = (df["time_diff"] - 5).fillna(0) / 5  # expected 5-min intervals

# Fill short gaps with linear interpolation
df[col] = df[col].interpolate(method="linear", limit=threshold_steps)
df.loc[df["imputation_flag"].isna() & df[col].isna(), "imputation_flag"] = "INTERPOLATED"

# Mark long gaps
df.loc[missing_steps > threshold_steps, "imputation_flag"] = "GAP"
```

| Gap Duration | Treatment | Flag |
|--------------|-----------|------|
| < 2 hours | Linear interpolation | `INTERPOLATED` |
| ≥ 2 hours | Do not interpolate | `GAP` |

### Step 5: Rolling Median Filter

```python
ROLLING_MEDIAN_WINDOW = 3
for col in ["pm2_5", "pm10", "temperature", "humidity"]:
    df[col] = df[col].rolling(window=ROLLING_MEDIAN_WINDOW, min_periods=1, center=True).median()
```

This removes transient high-frequency noise while preserving genuine pollution trends.

### Step 6: Chronological Validation

Final checks before feature engineering:
- Timestamps are strictly increasing
- No duplicate timestamps
- Every reading has a `quality_flag` and `imputation_flag`

## 4.4 Provenance Flags

| Flag | Meaning | Used in Training? |
|------|---------|-------------------|
| `OK` | Original valid reading | Yes |
| `INTERPOLATED` | Filled short gap | Yes |
| `GAP` | Long gap, not interpolated | No (exclude windows containing this) |
| `SENSOR_ERROR` | Statistical outlier / freeze | No |
| `OUT_OF_RANGE` | Physically impossible | No |

## 4.5 Required Output: Table 4.7

After preprocessing, export `results/tables/dataset_statistics.json`:

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

## 4.6 Scaling

After feature engineering, scale the feature matrix.

**Critical rules:**
- Fit scaler **only** on the training partition.
- Apply the fitted scaler to validation and test.
- Save the scaler to `model/scaler.joblib` for inference.

```python
from sklearn.preprocessing import MinMaxScaler
import joblib

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.joblib")
```

## 4.7 Key Implementation Notes

- **Never drop flagged rows during preprocessing.** Keep them with flags so sequence generation can exclude contaminated windows.
- **Fit scaler only on training data.** Data leakage invalidates evaluation metrics.
- **Validate that PM10 ≥ PM2.5** at ingestion. If not, flag as sensor error.
- **Use 5-minute native resolution** for the canonical modeling pipeline.

## 4.8 Known Risks

| Risk | Mitigation |
|------|------------|
| High interpolation rate inflates metrics | Cap interpolation at < 5% of records |
| Long gaps break sequence generation | Exclude any window containing a `GAP` row |
| Sensor drift over time | Periodic recalibration against reference instruments |
| Single sensor failure | Design multi-sensor fusion as future work |

---

*Next: read [`05_FEATURE_ENGINEERING.md`](05_FEATURE_ENGINEERING.md) to build the model inputs.*
