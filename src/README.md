# Source Code Notes

## Status

These modules are the PyTorch rebuild of the ALLERT system. The legacy
implementation in `C:\Users\USER\Codes\model (tensorflow)` is retained only for
historical reference; do not use it for new work.

## Critical Bug (Historical)

The original `train_final.py` from the legacy project concatenated training and
validation sets before calling `model.fit()`, disabling early stopping. The
PyTorch rebuild in `src/pytorch_training.py` always uses a held-out validation
set and supports checkpoint/resume.

## Refactoring Priority

1. `src/pytorch_training.py` — training loop with held-out validation, early
   stopping, and checkpoint/resume
2. `src/sequence_builder.py` — sliding-window sequences
3. `src/pytorch_model.py` — PyTorch Bi-LSTM + attention architecture
4. `src/preprocessing.py` — provenance flags and train-only scaler fitting
5. `src/feature_engineering.py` — lag/rolling/cyclical features
6. `src/risk_engine.py` — distributed-lag risk scoring
7. `src/anomaly_detection.py` — CUSUM parameters and reset policy

## Modular Build Order

Follow the phases in `PROMPT.md`:

1. Config loader
2. Preprocessing
3. Feature engineering
4. Scaling
5. Sequence builder
6. PyTorch Bi-LSTM model (`src/pytorch_model.py`)
7. Training (`src/pytorch_training.py`)
8. Evaluation
9. Anomaly detection
10. Risk engine
11. Clinical validity
12. Orchestrator + notebook
13. Unit tests

Build one phase at a time. Verify each before moving on.
