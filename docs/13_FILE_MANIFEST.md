# 13 — File Manifest: What to Bring from the Original Project

This document lists the files in the original project (`C:\Users\USER\Codes\model (tensorflow)`) that should be copied, adapted, or referenced when rebuilding ALLERT in this new folder.

## 13.1 Original Project Structure

```
model (tensorflow)/
├── allert_docs/                  # All design documentation
├── ALLERT_Master_Notebook.ipynb
├── CODE_TO_THESIS_MAPPING.md
├── config.yaml
├── data/
│   └── air_quality_data.csv
├── DEFENSE_WALKTHROUGH.md
├── model/
├── notebook/
├── requirements.txt
├── results/
├── src/
│   ├── __init__.py
│   ├── anomaly_detection.py
│   ├── bilstm_model.py
│   ├── evaluation.py
│   ├── export_results.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── regenerate_sequences.py
│   ├── risk_engine.py
│   ├── run_evaluation.py
│   ├── train_final.py
│   └── validity.py
└── tests/
```

## 13.2 Files to Copy (No Changes Needed)

| File | Destination | Why |
|------|-------------|-----|
| `data/air_quality_data.csv` | `data/raw/air_quality_data.csv` | Raw sensor data; immutable |
| `requirements.txt` | `requirements.txt` | Python dependencies (review versions) |
| `config.yaml` | `config/config.yaml` | Base config to adapt and improve |

## 13.3 Files to Copy and Refactor

These modules are mostly correct but need cleanup and the critical training bug fix:

| Original File | New Location | Key Changes Needed |
|---------------|--------------|-------------------|
| `src/preprocessing.py` | `src/preprocessing.py` | Add provenance flags; ensure scaler fits on training only |
| `src/feature_engineering.py` | `src/feature_engineering.py` | Add inline comments; verify lag offsets |
| `src/evaluation.py` | `src/evaluation.py` | Verify chronological split; ensure metrics handle all horizons |
| `src/risk_engine.py` | `src/risk_engine.py` | Verify distributed lag weight normalization |
| `src/anomaly_detection.py` | `src/anomaly_detection.py` | Verify CUSUM k/h parameters and reset policy |
| `src/validity.py` | `src/validity.py` | Add figure generation for Fig 4.14 |
| `src/export_results.py` | `src/export_results.py` | Add missing figure automation |

## 13.4 Files to Rewrite

| Original File | Reason | New Implementation |
|---------------|--------|-------------------|
| `src/bilstm_model.py` | Legacy implementation | Rewrite as `src/pytorch_model.py` in PyTorch |
| `src/train_final.py` | **Critical bug:** concatenates train+validation, disabling early stopping | Rewrite as `src/pytorch_training.py` with held-out validation and checkpoint/resume |
| `src/run_evaluation.py` | Should be folded into orchestrator | Create `src/orchestrator.py` |
| `ALLERT_Master_Notebook.ipynb` | Needs markdown discussion cells and corrected training flow | Update `ALLERT_Master_Notebook.ipynb` at repository root |

## 13.5 Files to Read for Reference Only

These are documentation files that have been distilled into this `docs/` suite. You do not need to copy them, but they may contain extra detail:

| File | Contents |
|------|----------|
| `allert_docs/PROJECT_CONTEXT.md` | Background, problem statement, hardware stack |
| `allert_docs/MODEL_DESIGN.md` | Original Bi-LSTM architecture spec |
| `allert_docs/DATA_PIPELINE.md` | Preprocessing pipeline details |
| `allert_docs/FEATURE_ENGINEERING.md` | Feature formulas |
| `allert_docs/TRAINING_STRATEGY.md` | Original training approach |
| `allert_docs/ANOMALY_DETECTION.md` | Anomaly detection formulas |
| `allert_docs/RISK_ENGINE.md` | Risk engine formulas |
| `allert_docs/EVALUATION_PLAN.md` | Evaluation metrics and outputs |
| `allert_docs/DEPLOYMENT_ARCHITECTURE.md` | Production architecture |
| `allert_docs/LIMITATIONS_AND_FUTURE_WORK.md` | Limitations and roadmap |
| `allert_docs/ALLERT_Model_Audit_Report.md` | Audit findings and critical bug identification |
| `allert_docs/misc/ALLERT_Model_Implementation_Audit.md` | 129-item checklist |
| `DEFENSE_WALKTHROUGH.md` | Defense presentation guide |
| `CODE_TO_THESIS_MAPPING.md` | Code-to-thesis traceability index |

## 13.6 What Not to Copy

| Item | Reason |
|------|--------|
| Old `model/*.pt` / `*.json` | Retrain on the new pipeline |
| Old `results/*` | Regenerate with corrected pipeline |
| `src/__pycache__/` | Auto-generated; not needed |
| WSL-specific diagnostic scripts | Not part of core model |

## 13.7 Recommended Copy Commands

Run these from PowerShell to copy the essential files:

```powershell
$src = "C:\Users\USER\Codes\model (tensorflow)"
$dst = "C:\Users\USER\Codes\allert_rebuild"

# Copy raw data
Copy-Item "$src\data\air_quality_data.csv" "$dst\data\raw\air_quality_data.csv" -Force

# Copy config and requirements as starting points
Copy-Item "$src\config.yaml" "$dst\config\config.yaml" -Force
Copy-Item "$src\requirements.txt" "$dst\requirements.txt" -Force

# Copy source modules to refactor
$modules = @(
    "preprocessing.py",
    "feature_engineering.py",
    "evaluation.py",
    "anomaly_detection.py",
    "risk_engine.py",
    "validity.py",
    "export_results.py"
)
foreach ($m in $modules) {
    Copy-Item "$src\src\$m" "$dst\src\$m" -Force
}
```

## 13.8 Important Notes

1. **Do not copy `train_final.py` blindly.** It contains the critical validation bug.
2. **Review `config.yaml` after copying.** This rebuild uses an improved config structure.
3. **Keep original documentation intact** in the old folder for reference.
4. **Do not copy old model weights or results.** They were produced by the buggy original pipeline.

---

*End of documentation suite. Return to [`README.md`](../README.md) for the index.*
