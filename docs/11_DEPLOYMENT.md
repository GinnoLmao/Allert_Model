# 11 — Deployment Architecture

This document describes the six-layer cloud-integrated architecture for production deployment.

## 11.1 Six-Layer Stack

```
┌─────────────────────────────────────┐
│ Layer 6: Client Apps                │
│ Web App | User Portal | Admin Panel │
├─────────────────────────────────────┤
│ Layer 5: Backend API                │
│ Auth | Forecast API | Symptoms API  │
│ Correlation Service                 │
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

## 11.2 Layer Specifications

### Layer 1: Data Ingestion

| Component | Function |
|-----------|----------|
| Ingestion API | REST endpoint receiving sensor payloads; validates schema and deduplicates |
| Buffer Manager | Offline queue during WiFi outages; flushes in chronological order on reconnect |
| Data Logger | Writes validated readings to time-series DB |

### Layer 2: Data Processing

| Component | Function |
|-----------|----------|
| QA Validator | Range checks, sensor fault detection, outlier flagging |
| Gap Fill | Linear interpolation for < 2h gaps; marks longer gaps |
| Feature Engineering | Builds lagged variables, rolling stats, cyclical encodings, exposure weights |

### Layer 3: ML Engine

| Component | Function |
|-----------|----------|
| LSTM Forecaster | Loads trained weights; returns 6h/12h/24h forecasts per call |
| Anomaly Detector | Continuous Z-score, velocity, CUSUM monitoring |
| Risk Calculation Engine | Converts forecasts to AR probability |

### Layer 4: Alert Engine

| Component | Function |
|-----------|----------|
| Risk Classifier | Maps risk probability to tier; applies pediatric offset |
| Alert Dispatcher | Routes alerts by channel and urgency |

### Layer 5: Backend API

| Component | Function |
|-----------|----------|
| Auth Service | JWT-based login and session management |
| Forecast API | Serves cached forecasts and current risk |
| Symptoms API | Receives voluntary VAS/TNSS entries |
| Correlation Service | Computes Spearman rho between risk tiers and symptoms |

### Layer 6: Client Apps

| Component | Function |
|-----------|----------|
| User Portal | Risk dashboard, forecast screen, symptom diary, history view |
| Admin Panel | Sensor management, model performance monitoring |

## 11.3 Database Design

### Time-Series Store (InfluxDB / TimescaleDB)

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | DATETIME | Record timestamp |
| `sensor_id` | INT | Foreign key |
| `pm2_5` | DECIMAL | Fine PM concentration |
| `pm10` | DECIMAL | Coarse PM concentration |
| `temperature` | DECIMAL | Temperature |
| `humidity` | DECIMAL | Relative humidity |
| `quality_flag` | VARCHAR | OK / SENSOR_ERROR / OUT_OF_RANGE |
| `imputation_flag` | VARCHAR | ORIGINAL / INTERPOLATED / GAP |

### Relational Store (PostgreSQL / MySQL)

Key tables:
- `SENSORS`
- `MEASUREMENTS`
- `FORECASTS`
- `RISK_CALCULATIONS`
- `ALERTS`
- `USERS`
- `MODEL_METADATA`

## 11.4 Core REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ingest` | POST | Sensor data ingestion |
| `/api/v1/forecast/{sensor_id}` | GET | 6h/12h/24h forecasts + risk |
| `/api/v1/risk/{user_id}` | GET | Current risk tier and P_risk |
| `/api/v1/alerts/{user_id}` | GET | Alert history |
| `/api/v1/symptoms` | POST | Submit VAS/TNSS score |
| `/api/v1/correlation/{user_id}` | GET | Spearman rho results |
| `/api/v1/admin/sensors` | GET/POST | Sensor management |
| `/api/v1/admin/models` | GET | Model versions and metrics |

## 11.5 Notification Delivery

### Channels by Tier

| Tier | Channels |
|------|----------|
| Watch | App push only |
| Advisory | App push + Email |
| Warning | App push + Email + SMS |
| Active/Emergency | All channels + emergency broadcast |

### Smart Throttling

- Dispatch only on distinct tier escalations.
- Suppress duplicate alerts within the same event.
- Respect quiet hours for Watch/Advisory; override for Warning/Emergency.

## 11.6 Operational Targets

| Target | Value |
|--------|-------|
| Sensor uptime | ≈ 99% |
| End-to-end latency | < 30 seconds |
| Data availability | ≥ 95% hourly completeness |
| API response (cached) | < 500 ms |
| Retraining cycle | Every 30 days |

## 11.7 Security

- HTTPS/TLS for all sensor-to-cloud communication
- JWT tokens for user authentication
- Device tokens registered per sensor
- Symptom data anonymized; no PII linked to environmental records

## 11.8 Future Extensibility

- Multi-region cloud deployment
- Edge ML inference on ESP32-S3 for offline operation
- Multi-sensor spatial coverage across Iloilo City
- Integration with DENR-EMB and OpenAQ APIs

---

*Next: read [`12_IMPLEMENTATION_CHECKLIST.md`](12_IMPLEMENTATION_CHECKLIST.md) for the step-by-step build plan.*
