# 05 — Feature Engineering

Feature engineering transforms raw PM2.5, PM10, temperature, and humidity into a rich temporal representation that captures autoregressive structure, pollution trends, cyclical patterns, and clinical lag dynamics.

## 5.1 Final Tensor Shape

```
Input tensor: (N, 2016, F)
  N = number of sequences
  2016 = 7 days × 24 hours × 12 five-minute steps/hour
  F = number of engineered features per timestep
```

## 5.2 Feature Groups

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

## 5.3 Group 1: Raw Input Variables

These are the baseline measured variables after preprocessing:

| Feature | Source | Unit |
|---------|--------|------|
| `pm2_5` | SPS30 sensor | µg/m³ |
| `pm10` | SPS30 sensor | µg/m³ |
| `temperature` | SHT40 sensor | °C |
| `humidity` | SHT40 sensor | %RH |

## 5.4 Group 2: Autoregressive Lag Features

Lagged copies of PM2.5 and PM10 using exact 5-minute step offsets:

| Temporal Offset | Steps (5-min) | Feature Name |
|----------------:|--------------:|--------------|
| 1 hour | 12 | lag_1h |
| 6 hours | 72 | lag_6h |
| 12 hours | 144 | lag_12h |
| 24 hours | 288 | lag_24h |
| 48 hours | 576 | lag_48h |
| 72 hours | 864 | lag_72h |

**Clinical basis:** Distributed lag evidence from AR epidemiology:
- 60% of inflammatory effects manifest on the same day
- 25% at a one-day lag
- 10% at a two-day lag
- 5% at a three-day lag

```python
for pollutant in ["pm2_5", "pm10"]:
    for hours in [1, 6, 12, 24, 48, 72]:
        steps = hours * 12
        df[f"{pollutant}_lag_{hours}h"] = df[pollutant].shift(steps)
```

## 5.5 Group 3: Rolling Statistical Summaries

Computed over four windows: 6h, 12h, 24h, and 7 days (168h).

### Rolling Mean
```python
SMA_t(w) = (1/w) * Σ x[t-i] for i in 0..w-1
```

### Rolling Standard Deviation
```python
SD_t(w) = sqrt((1/w) * Σ (x[t-i] - SMA_t(w))**2)
```

### Rolling Min / Max
```python
MIN_t(w) = min(x[t-w+1], ..., x[t])
MAX_t(w) = max(x[t-w+1], ..., x[t])
```

**Windows in 5-min steps:**
- 6h = 72 steps
- 12h = 144 steps
- 24h = 288 steps
- 7d = 2016 steps

```python
for pollutant in ["pm2_5", "pm10"]:
    for hours in [6, 12, 24, 168]:
        steps = hours * 12
        df[f"{pollutant}_mean_{hours}h"] = df[pollutant].rolling(steps, min_periods=1).mean()
        df[f"{pollutant}_std_{hours}h"] = df[pollutant].rolling(steps, min_periods=1).std()
        if hours != 168:  # skip 7-day min/max to keep dimensionality manageable
            df[f"{pollutant}_min_{hours}h"] = df[pollutant].rolling(steps, min_periods=1).min()
            df[f"{pollutant}_max_{hours}h"] = df[pollutant].rolling(steps, min_periods=1).max()
```

## 5.6 Group 4: Pollutant Velocity (Rate-of-Change)

```python
velocity_t(k) = (x_t - x_{t-k}) / k
```

Convert to hourly rate by multiplying by 12.

| Feature | Step (5-min) | Hourly Conversion |
|---------|-------------:|-------------------|
| velocity_1h | 12 | `(x_t - x_{t-12}) / 12 * 12` = µg/m³/hour |
| velocity_6h | 72 | `(x_t - x_{t-72}) / 72 * 12` = µg/m³/hour |

```python
for pollutant in ["pm2_5", "pm10"]:
    for hours in [1, 6]:
        steps = hours * 12
        df[f"{pollutant}_vel_{hours}h"] = (df[pollutant] - df[pollutant].shift(steps)) / steps * 12
```

## 5.7 Group 5: Cyclical Time Encodings

Temporal variables are cyclical. Sine/cosine encoding preserves topological closeness (e.g., hour 23 is adjacent to hour 0).

```python
# Hour of day
hour = df["timestamp"].dt.hour
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)

# Day of week
dow = df["timestamp"].dt.dayofweek
dow_sin = sin(2π * dow / 7)
dow_cos = cos(2π * dow / 7)

# Month of year
month = df["timestamp"].dt.month
month_sin = sin(2π * month / 12)
month_cos = cos(2π * month / 12)
```

## 5.8 Group 6: 72-Hour Cumulative Exposure

Weighted cumulative exposure reflecting distributed lag structure:

```python
exposure_72h = (
    0.60 * PM_t +
    0.25 * PM_{t-24h} +
    0.10 * PM_{t-48h} +
    0.05 * PM_{t-72h}
)
```

```python
for pollutant in ["pm2_5", "pm10"]:
    df[f"{pollutant}_cum_72h"] = (
        0.60 * df[pollutant] +
        0.25 * df[pollutant].shift(24 * 12) +
        0.10 * df[pollutant].shift(48 * 12) +
        0.05 * df[pollutant].shift(72 * 12)
    )
```

## 5.9 Feature Engineering Workflow

```
5-minute block-averaged data
    ↓
1. Compute lag features at exact 5-minute offsets
    ↓
2. Compute rolling statistics (mean, std, min, max)
    ↓
3. Compute velocity indicators
    ↓
4. Compute cyclical time encodings
    ↓
5. Compute 72h cumulative exposure weights
    ↓
6. Fit scaler on training data only
    ↓
7. Transform all splits
    ↓
8. Build sliding window sequences (length=2016, stride=1)
    ↓
Output: Tensor (N, 2016, F)
```

## 5.10 Scaling

- Use **MinMaxScaler** or **StandardScaler**.
- Fit **only on training data**.
- Save scaler to `model/scaler.joblib`.
- Apply identical transformation to validation and test.

## 5.11 Sequence Generation

```python
def build_sequences(df, feature_cols, target_cols, sequence_length=2016, stride=1):
    """
    Returns:
        X: (N, 2016, F)
        y: (N, 6) -> [pm25_6h, pm10_6h, pm25_12h, pm10_12h, pm25_24h, pm10_24h]
    """
    # Skip windows containing GAP-flagged rows
```

### Target Vector Order (Keep Consistent Everywhere)

```python
TARGET_COLS = [
    "pm2_5_6h", "pm10_6h",
    "pm2_5_12h", "pm10_12h",
    "pm2_5_24h", "pm10_24h",
]
```

## 5.12 Assumptions and Risks

| Assumption | Risk |
|------------|------|
| 56 features is appropriate for ~90 days of data | Overfitting if validation loss stagnates early |
| Lag features up to 72h are clinically meaningful | First 3 days of data cannot produce valid windows |
| 7-day rolling windows capture long-term baseline | Reduces effective training data by 7 days |
| Cyclical encodings beat raw hour/day/month | True for neural networks; verified by EDA |

## 5.13 Future Extensibility

- Add wind speed, pressure, rainfall as meteorological covariates
- Add pollen or bioaerosol features if sensors are deployed
- Use mutual information or SHAP to prune low-importance features
- Try CNN-LSTM hybrid for local pattern extraction

---

*Next: read [`06_MODEL_ARCHITECTURE.md`](06_MODEL_ARCHITECTURE.md) to design the Bi-LSTM.*
