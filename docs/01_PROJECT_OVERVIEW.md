# 01 — Project Overview

## 1.1 System Identity

| Field | Detail |
|-------|--------|
| **Full Name** | ALLERT — Airborne particulate forecasting and Environmental Risk aLeRT system |
| **Domain** | Environmental health informatics, IoT, machine learning |
| **Target Condition** | Allergic Rhinitis (AR) |
| **Deployment City** | Iloilo City, Western Visayas, Philippines |
| **Research Type** | Developmental–Experimental |
| **Validation Framework** | ISO/IEC 25010 Software Quality Model |

## 1.2 Problem Statement

Allergic rhinitis management in the Philippines is almost entirely reactive. Patients must choose between perpetual medication use or uninformed exposure to harmful particulate conditions. In Iloilo City, perennial allergens persist year-round without temperate seasonal variation, and no localized, clinically-calibrated environmental risk information exists.

## 1.3 Three Gaps ALLERT Solves

1. **Single-step forecasts only** — Existing systems predict only the next hour. ALLERT predicts 6h, 12h, and 24h ahead.
2. **Concentration thresholds, not risk** — Existing systems flag PM2.5 = 35 µg/m³ as "unhealthy" regardless of who is exposed. ALLERT converts concentration into age-stratified AR probability.
3. **No integrated anomaly detection** — Existing systems miss sudden emission spikes. ALLERT runs dual-mode real-time anomaly detection alongside scheduled forecasts.

## 1.4 Research Questions

| RQ | Question | Success Criterion |
|----|----------|-----------------|
| **RQ1** | Can a Bi-LSTM forecast PM2.5/PM10 at 6h, 12h, and 24h with clinically useful accuracy? | MAPE < 5% (6h), < 7% (12h), < 10% (24h); Bi-LSTM outperforms ARIMA. |
| **RQ2** | Can dual-mode anomaly detection distinguish emission events from sensor noise? | Sensitivity > 95%, False Positive Rate < 10%. |
| **RQ3** | Does the risk engine's output correlate with actual health symptoms? | Spearman rho > 0.30, p < 0.05 vs. VAS/TNSS scores. |

## 1.5 Core Premise

AR symptom burden can be reduced through **anticipatory rather than responsive action**. ALLERT combines real-time PM sensing with Bi-LSTM forecasting to enable:

- Preemptive protective strategies before exposure peaks
- Timely medication pre-dosing
- Activity rescheduling
- Indoor air filtration activation
- Emergency shelter-seeking during acute spikes

## 1.6 Four Technical Innovations

1. **Allergy-specific risk mapping** — Uses Asia-specific epidemiological odds ratios calibrated to Filipino baseline prevalence.
2. **Age-stratified alert thresholds** — Pediatric thresholds are stricter than adult thresholds.
3. **Dual-mode alerting** — Scheduled 6h/12h/24h forecasts + continuous real-time anomaly detection.
4. **Cloud-based portable deployment** — ESP32-S3 + SPS30 + SHT40 with REST ingestion and centralized model management.

## 1.7 Hardware Stack

| Component | Model | Role |
|-----------|-------|------|
| Particulate sensor | Sensirion SPS30 | PM2.5 and PM10 via laser scattering |
| Temp/Humidity sensor | Sensirion SHT40 | Microclimate capture |
| Microcontroller | ESP32-S3 | Edge processing, WiFi transmission |
| Communication | REST API | Cloud ingestion |
| Sampling rate | Every 10 seconds | Edge capture |
| Transmission rate | Every 5 minutes | Cloud ingestion (block averages) |

## 1.8 Data Variables

| Variable | Unit | Description |
|----------|------|-------------|
| `pm2_5` | µg/m³ | Fine particulate matter ≤ 2.5 µm |
| `pm10` | µg/m³ | Coarse particulate matter ≤ 10 µm |
| `temperature` | °C | Ambient temperature |
| `humidity` | %RH | Relative humidity |

## 1.9 Dataset Scope

| Attribute | Value |
|-----------|-------|
| Duration | ~3 months continuous |
| Temporal resolution | 5-minute block averages |
| Records per day | 288 |
| Total records (approx.) | ~25,920 |
| Availability target | ≥ 95% hourly completeness |
| Calibration | Co-located with DENR-EMB / OpenAQ for ≥ 72 hours |

## 1.10 Explicit Positioning

ALLERT is a **supportive environmental risk awareness tool**, not a diagnostic or prescribing system. It estimates **population-level environmental risk** using PM as a proxy for particle-bound allergenic exposure. Individual symptom expression depends on many unmeasured factors (IgE profile, medication adherence, indoor allergens, infections, stress).

## 1.11 Key Terminology

| Term | Definition |
|------|------------|
| PM2.5 / PM10 | Fine and coarse particulate matter |
| Bi-LSTM | Bidirectional Long Short-Term Memory neural network |
| TPA | Temporal Pattern Attention mechanism |
| CUSUM | Cumulative Sum statistical control chart |
| VAS | Visual Analog Scale (0–100) symptom severity |
| TNSS | Total Nasal Symptom Score (0–12) |
| OR | Odds Ratio |
| P_baseline | Age-specific AR prevalence baseline |
| P_risk | Final computed AR risk probability (%) |

---

*Next: read [`02_FOLDER_STRUCTURE.md`](02_FOLDER_STRUCTURE.md) to set up the project directories.*
