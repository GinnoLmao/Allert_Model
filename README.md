<div align="center">

# ALLERT

### Air-quality monitoring research · Multi-horizon particulate forecasting

**An undergraduate AI research project focused on PM2.5 and PM10 forecasting**

![Python](https://img.shields.io/badge/Python-Data%20Science-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-BiLSTM-EE4C2C?logo=pytorch&logoColor=white)
![Forecasting](https://img.shields.io/badge/Forecasts-6h%20%7C%2012h%20%7C%2024h-0F766E)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-64748B)

</div>

## Overview

**ALLERT** investigates machine-learning-assisted monitoring and forecasting of airborne particulate pollution. This repository contains the **PyTorch forecasting and evaluation pipeline**, including data-quality checks, feature engineering, attention-based bidirectional LSTM models, anomaly detection, and supporting research analysis.

The broader ALLERT thesis explores how sensor readings and air-quality forecasts could support environmental awareness. **This repository is the modeling/research component**, not a complete IoT hardware, web-application, or clinically validated deployment.

### Research questions

- Forecast **PM2.5 and PM10** concentrations at **6-, 12-, and 24-hour horizons**.
- Compare the neural forecasting model with statistical baselines such as ARIMA and persistence.
- Investigate anomaly detection and interpretable indicators for elevated particulate conditions.
- Evaluate data handling, chronological splits, model behavior, and the limitations of risk modeling.

## Architecture

```text
5-minute particulate / weather measurements
          |
          v
Data validation + quality / imputation flags
          |
          v
Lags, rolling statistics, cyclical features
          |
          v
Chronological splitting + training-only scaling
          |
          v
Sequence construction
          |
          v
Attention-based bidirectional LSTM (PyTorch)
          |
          +------> PM2.5 + PM10 at 6h
          +------> PM2.5 + PM10 at 12h
          +------> PM2.5 + PM10 at 24h
          |
          v
Evaluation + baseline comparison + analysis
```

**Architecture in source:** `src/pytorch_model.py` implements stacked bidirectional LSTMs with optional learned temporal attention and separate forecast heads. `src/orchestrator.py` coordinates preprocessing, training, evaluation, tuning, and artifact generation. Configurable experiments are defined in `config/config.yaml`.

## Technology & research methods

| Area | Tools or methods |
| --- | --- |
| Modeling | PyTorch, stacked BiLSTM, temporal attention, multi-output regression |
| Data preparation | pandas, NumPy, timestamp alignment, gap detection, feature engineering |
| Evaluation | scikit-learn, forecasting error metrics, ARIMA/persistence comparisons |
| Experimentation | Optuna hyperparameter search, checkpoint/resume, chronological validation |
| Reporting | Matplotlib, Seaborn, research figures and analysis tables |
| Quality checks | pytest and modular pipeline tests |

## Repository structure

```text
ALLERT_Master_Notebook.ipynb    # Interactive research workflow
config/config.yaml              # Experiment configuration
src/
  preprocessing.py              # Input validation and gap handling
  feature_engineering.py        # Predictive features
  pytorch_model.py              # Attention-based BiLSTM architecture
  pytorch_training.py           # Training and checkpoint utilities
  evaluation.py                 # Forecast quality and baselines
  anomaly_detection.py          # Spike/anomaly experiments
  orchestrator.py               # Command-line pipeline
docs/                            # Research and implementation notes
tests/                           # Pipeline and unit tests
requirements.txt                 # Python dependencies
```

## Run locally

This repository expects **input data provided separately**. The default file path is `data/raw/air_quality_data_5m.csv`; the `data/` directory and generated results are excluded from version control.

### 1. Set up Python

```bash
git clone https://github.com/GinnoLmao/Allert_Model.git
cd Allert_Model
python -m venv .venv
```

Activate the environment using your operating system's venv activation command, then install dependencies:

```bash
pip install -r requirements.txt
```

> `requirements.txt` selects a CUDA 12.6 PyTorch wheel index, which is primarily intended for compatible NVIDIA GPU environments. Adapt the PyTorch installation to your device and OS if necessary.

### 2. Provide your own eligible dataset

At the configured location, prepare a CSV with at least:

```text
timestamp,pm2_5,pm10,temperature,humidity
```

`temperature_c` and `humidity_percent` are also accepted as source column names and normalized during loading. For meaningful results, use sufficient genuine time-stamped data and document its provenance. **No training dataset or raw sensor records are published here.**

### 3. Run tests and experiments

```bash
python -m pytest tests -q
python src/orchestrator.py --mode train --quick --subset-days 20 --stride 2 --epochs 2 --batch-size 8 --seed 42
```

Other modes exposed by the orchestrator include `evaluate`, `full`, `hpo`, and `bundle`. Resource use can be substantial with the default long input sequences.

These are **setup instructions, not a claim that a fresh run was executed during this documentation update**.

## Results and limitations

The repository's [historical system summary](docs/15_SYSTEM_SUMMARY_AND_IMPROVEMENTS.md) documents a **20-day subset / 3-epoch correctness run** in which the BiLSTM performed **worse than the ARIMA baseline**. That limited run verifies aspects of execution, **not research success or production-grade forecasting accuracy**. Additional data provenance checks, independent evaluation, and longer experiments are necessary before drawing conclusions.

Any age-stratified allergic-rhinitis risk outputs are **exploratory environmental-risk modeling**, not clinically established individual health predictions. This project does not diagnose conditions or recommend treatment.

## Read the research notes

- [Project goals and limitations](docs/01_PROJECT_OVERVIEW.md)
- [Preprocessing and data handling](docs/04_DATA_PIPELINE.md)
- [Model architecture](docs/06_MODEL_ARCHITECTURE.md)
- [Training strategy](docs/07_TRAINING_STRATEGY.md)
- [Evaluation methodology](docs/10_EVALUATION.md)
- [Historical run results and issues](docs/15_SYSTEM_SUMMARY_AND_IMPROVEMENTS.md)
- [Improvement roadmap](docs/16_IMPROVEMENT_PLAN.md)

---

**Academic research prototype · West Visayas State University**

*For more projects, visit [Ginno Arostique's GitHub profile](https://github.com/GinnoLmao).*
