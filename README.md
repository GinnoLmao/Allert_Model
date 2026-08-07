# ALLERT Rebuild — PyTorch GPU Pipeline with Checkpoint/Resume

> **Purpose:** This folder contains the complete ALLERT air-quality risk forecasting system rebuilt in **PyTorch with CUDA GPU support** and **automatic checkpoint/resume**. The master notebook and CLI orchestrator run the full pipeline end-to-end, and training can be paused and resumed at any time without losing progress.

## Start Here

| File | What It Covers | Read This If You... |
|------|----------------|---------------------|
| [`docs/01_PROJECT_OVERVIEW.md`](docs/01_PROJECT_OVERVIEW.md) | Problem, research questions, system identity, hardware | Need context on what ALLERT is and why it exists |
| [`docs/02_FOLDER_STRUCTURE.md`](docs/02_FOLDER_STRUCTURE.md) | Recommended directory layout | Are setting up the project from scratch |
| [`docs/03_CONFIGURATION.md`](docs/03_CONFIGURATION.md) | `config.yaml` design and config loader | Want one source of truth for all parameters |
| [`docs/04_DATA_PIPELINE.md`](docs/04_DATA_PIPELINE.md) | Preprocessing, QA, gap filling, scaling | Need to clean raw sensor data correctly |
| [`docs/05_FEATURE_ENGINEERING.md`](docs/05_FEATURE_ENGINEERING.md) | Lags, rolling stats, cyclical encodings, velocity, exposure | Need to build the 56-feature input matrix |
| [`docs/06_MODEL_ARCHITECTURE.md`](docs/06_MODEL_ARCHITECTURE.md) | Bi-LSTM, attention, output heads, loss function | Are building or debugging the neural network |
| [`docs/07_TRAINING_STRATEGY.md`](docs/07_TRAINING_STRATEGY.md) | Splits, regularization, Optuna HPO, walk-forward CV | Need to train the model without overfitting |
| [`docs/08_ANOMALY_DETECTION.md`](docs/08_ANOMALY_DETECTION.md) | Z-score, velocity override, CUSUM | Need real-time spike detection |
| [`docs/09_RISK_ENGINE.md`](docs/09_RISK_ENGINE.md) | Epidemiological scoring, 72h lag, tier classification | Need to turn PM forecasts into health risk |
| [`docs/10_EVALUATION.md`](docs/10_EVALUATION.md) | Metrics, ARIMA baseline, clinical validity | Need to prove the system works |
| [`docs/11_DEPLOYMENT.md`](docs/11_DEPLOYMENT.md) | Six-layer architecture, API, notifications | Are deploying to production |
| [`docs/12_IMPLEMENTATION_CHECKLIST.md`](docs/12_IMPLEMENTATION_CHECKLIST.md) | Step-by-step build list + known pitfalls | Want a concrete execution plan |
| [`docs/13_FILE_MANIFEST.md`](docs/13_FILE_MANIFEST.md) | Files to copy from the original project | Need to know what existing code to reuse |
| [`docs/15_SYSTEM_SUMMARY_AND_IMPROVEMENTS.md`](docs/15_SYSTEM_SUMMARY_AND_IMPROVEMENTS.md) | Run results, known issues, and improvement roadmap | Want a hand-off snapshot of the current system state |
| [`docs/16_IMPROVEMENT_PLAN.md`](docs/16_IMPROVEMENT_PLAN.md) | Comprehensive thesis improvement roadmap | Need a detailed, phased plan to make the system thesis-ready |

### New/Updated Modules

| Module | Purpose |
|--------|---------|
| `src/pytorch_model.py` | PyTorch Bi-LSTM architecture, attention, composite loss |
| `src/pytorch_training.py` | PyTorch training loop with automatic checkpoint/resume |
| `src/utils/timing.py` | Per-phase latency instrumentation |
| `src/anomaly_calibration.py` | Validation-based CUSUM k/h grid search |
| `src/risk_sensitivity.py` | Epidemiological coefficient perturbation analysis |
| `src/interpretability.py` | Permutation importance, attention weights, optional SHAP |
| `src/thesis_exporter.py` | Bundle all tables/figures/models into a zip |
| `src/target_scaling.py` | Explicit target scaling/inversion for metric-unit correctness |
| `tests/test_performance_regression.py` | Pinned-metric regression test scaffold |
| `tests/test_target_scaling.py` | Target-unit correctness tests |
| `tests/test_anomaly_calibration.py` | Calibration and per-pollutant threshold tests |
| `tests/test_risk_sensitivity.py` | Sensitivity output tests |
| `tests/test_interpretability.py` | Permutation-importance tests |
| `tests/test_thesis_exporter.py` | Bundle export tests |
| `tests/test_timing.py` | Latency instrumentation tests |
| `tests/test_training_augmentation.py` | Augmentation and walk-forward CV tests |
| `tests/test_evaluation_extras.py` | Persistence baseline and regime-error tests |
| `tests/test_hpo_config.py` | HPO config validation tests |

## Quick Orientation

- **Original project:** `C:\Users\USER\Codes\model (tensorflow)`
- **This rebuild folder:** `C:\Users\USER\Codes\allert_rebuild`
- **Goal:** Rebuild the ALLERT Bi-LSTM air quality risk system in PyTorch (GPU enabled) with automatic checkpoint/resume, cleaner and with critical bugs fixed.
- **Interactive walkthrough:** [`ALLERT_Master_Notebook.ipynb`](ALLERT_Master_Notebook.ipynb) runs the full pipeline end-to-end with explanations and visualizations.

## Most Important Changes vs. Original

1. **Framework migration:** The entire model, training, HPO, evaluation, and interpretability stack has been migrated from TensorFlow/Keras to **PyTorch with CUDA GPU support**.
2. **Automatic checkpoint/resume:** Long training runs save model, optimizer, scheduler, and RNG state to `model/checkpoints/` every epoch. Re-run the notebook or CLI and training resumes from the latest checkpoint — no retraining.
3. **Training bug fixed:** Validation data is never concatenated with training data. Early stopping and learning-rate reduction monitor a real held-out validation set.
4. **No magic numbers:** Every threshold, coefficient, and path lives in `config.yaml`.
5. **Modular structure:** Each pipeline stage is isolated and unit-testable.
6. **Explicit provenance:** Data flags (`OK`, `INTERPOLATED`, `GAP`, `SENSOR_ERROR`, `OUT_OF_RANGE`) are preserved instead of silently dropped.
7. **Weight normalization asserted:** Distributed lag weights and loss weights are validated to sum to 1.0.
8. **Metric-unit correctness:** Evaluation metrics are explicitly reported in original µg/m³ units; target scaling is optional and invertible.
9. **Reproducibility controls:** `--seed` and optional `--deterministic` mode pin randomness for PyTorch, NumPy, and Python.
10. **Latency instrumentation:** `--with-timing` exports `table_4_13_latency.json` and a latency breakdown figure.
11. **Interpretability:** Permutation importance and attention-weight extraction are built in; SHAP is used only if already installed.
12. **HPO enhancements:** Configurable sampler/pruner, `l2_lambda` search, resumable SQLite storage, optimization-history plots.
13. **Data augmentation & walk-forward CV:** Jitter, window slicing, mixup, and chronological CV folds.
14. **Error analysis by regime:** Hour-of-day, day-of-week, and PM-level breakdowns.
15. **Additional baselines:** Persistence forecast alongside ARIMA.
16. **Thesis-ready figures:** CUSUM decision plot, confusion matrices, risk heatmap.

## Quick Commands

```powershell
# Run all tests
.venv\Scripts\python -m pytest tests -q

# Fast smoke test (tiny model, small subset)
.venv\Scripts\python src\orchestrator.py --mode train --quick `
  --subset-days 20 --stride 2 --epochs 2 --batch-size 8 --seed 42

# Evaluate/report with latency instrumentation
.venv\Scripts\python src\orchestrator.py --mode evaluate `
  --subset-days 20 --stride 2 --batch-size 64 --with-timing

# Export thesis bundle
.venv\Scripts\python src\orchestrator.py --mode bundle

# HPO smoke check
.venv\Scripts\python src\orchestrator.py --mode hpo --hpo-trials 2 --epochs 2 --batch-size 64 --seed 42

# Full GPU run (PyTorch, auto-checkpoint/resume)
.venv\Scripts\python src\orchestrator.py --mode full --seed 42 --epochs 200 --batch-size 32
```

## How to Use These Docs

1. Read `01_PROJECT_OVERVIEW.md` for context.
2. Set up the folder structure from `02_FOLDER_STRUCTURE.md`.
3. Copy or create the files listed in `13_FILE_MANIFEST.md`.
4. Follow the implementation checklist in `12_IMPLEMENTATION_CHECKLIST.md`.
5. Reference the deep-dive files as you implement each module.

---

*This documentation suite replaces the single `ALLERT_REBUILD_MASTER_GUIDE.md` with focused, editable sub-files.*
