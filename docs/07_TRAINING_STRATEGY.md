# 07 — Training Strategy

This document defines how to split the data, regularize the model, optimize hyperparameters, and train the final Bi-LSTM without the bugs found in the original implementation.

## 7.1 Dataset Volume Estimate

| Metric | Value |
|--------|-------|
| Collection period | ~90 days |
| 5-minute records | ~25,920 |
| Sequence length | 2,016 steps |
| Total sliding windows (stride=1) | ~23,900 |
| Training sequences (~70%) | ~16,700 |
| Validation sequences (~15%) | ~3,600 |
| Test sequences (~15%) | ~3,600 |

## 7.2 Chronological Split

```
Chronological Time →
|────────────── Training (~70%) ──────────────|── Validation (15%) ──|── Test (15%) ──|
Day 1                                         Day 63                Day 76           Day 90
```

**Critical rules:**
- Split **before** scaling or feature fitting.
- **No random shuffling.**
- Validation is used for early stopping and hyperparameter tuning.
- Test set is used only for final metrics.

```python
def temporal_train_val_test_split(df, train_frac, val_frac, test_frac):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test
```

## 7.3 The Critical Bug from Original Implementation

The original `train_final.py` did this:

```python
# WRONG — do not do this
X_trainval = np.concatenate([X_train, X_val])
model.fit(X_trainval, y_trainval, validation_data=None, ...)
```

This disabled early stopping and learning-rate reduction because there was no validation signal.

### Corrected Training Function

Implemented in `src/pytorch_training.py`. The PyTorch loop uses `ReduceLROnPlateau`, manual early stopping, gradient clipping, and automatic checkpoint/resume.

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


def train_model(cfg, X, y, train_mask, val_mask):
    set_random_seeds(cfg)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    model = build_bilstm((seq_len, n_features), cfg).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=cfg["model"]["learning_rate"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["training"]["lr_reduction_factor"],
        patience=cfg["training"]["lr_reduction_patience"],
        min_lr=cfg["training"]["min_lr"],
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=cfg["model"]["batch_size"], shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
        batch_size=cfg["model"]["batch_size"], shuffle=False,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = cfg["training"]["early_stopping_patience"]
    history = []

    for epoch in range(cfg["model"]["max_epochs"]):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = composite_loss(pred, yb, weights=tuple(cfg["model"]["loss_weights"]))
            loss = loss + model.l2_regularization()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["model"]["gradient_clip_norm"])
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = composite_loss(pred, yb, weights=tuple(cfg["model"]["loss_weights"]))
                val_losses.append(loss.item())

        train_loss = float(torch.mean(torch.tensor(train_losses)))
        val_loss = float(torch.mean(torch.tensor(val_losses)))
        scheduler.step(val_loss)

        history.append({"epoch": epoch + 1, "loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "model/best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return history
```

`src/pytorch_training.py` adds automatic checkpoint/resume: periodic `model/checkpoints/checkpoint_epoch_*.pt` files are written, and the latest checkpoint is loaded automatically when training restarts.

## 7.4 Regularization Techniques

| Technique | Configuration | Purpose |
|-----------|--------------|---------|
| Dropout | 0.25 recurrent | Prevent co-adaptation |
| L2 weight decay | 1e-4 on kernels | Penalize large weights |
| Early stopping | patience=10 on val_loss | Stop before overfitting |
| Learning rate reduction | factor=0.5, patience=5 | Finer convergence |
| Gradient clipping | `nn.utils.clip_grad_norm_(..., max_norm=1.0)` | Prevent gradient explosion |
| Compact architecture | 2 layers only | Match model size to data volume |

## 7.5 Data Augmentation (Optional)

If overfitting is severe, apply augmentation **only to training data**:

```python
def add_jitter(X, sigma=0.01):
    return X + np.random.normal(0, sigma, X.shape)
```

## 7.6 Hyperparameter Optimization: Optuna

Use Bayesian search with **temporal cross-validation**, not random k-fold.

```python
import optuna

def objective(trial, cfg):
    params = {
        "learning_rate": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "lstm_units_layer1": trial.suggest_categorical("u1", [64, 96, 128, 192, 256]),
        "lstm_units_layer2": trial.suggest_categorical("u2", [32, 48, 64, 96, 128]),
        "dropout_rate": trial.suggest_float("drop", 0.1, 0.5),
        "batch_size": trial.suggest_categorical("bs", [16, 32, 64]),
    }
    # Run 5-fold temporal CV and return mean validation loss
    return mean_val_loss

study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg["hpo"]["n_trials"])
```

## 7.7 Walk-Forward Cross-Validation

```
Fold 1: Train [Days 1–40]   → Validate [Days 41–50]
Fold 2: Train [Days 1–50]   → Validate [Days 51–60]
Fold 3: Train [Days 1–60]   → Validate [Days 61–70]
Fold 4: Train [Days 1–70]   → Validate [Days 71–78]
Fold 5: Train [Days 1–78]   → Validate [Days 79–86]
```

This preserves chronological integrity. Future data never leaks into training.

```python
def walk_forward_cv_indices(n_days, n_folds=5):
    # Returns list of (train_end_day, val_end_day) tuples
    ...
```

## 7.8 Training Configuration Summary

| Setting | Value |
|---------|-------|
| Max epochs | 200 |
| Optimizer | Adam |
| Batch size | 32 (or Optuna result) |
| Loss | Weighted composite MAE |
| Gradient clip norm | 1.0 |
| LR reduction factor | 0.5 |
| LR patience | 5 epochs |
| Early stopping patience | 10 epochs |
| Monitor | `val_loss` |

## 7.9 Retraining Strategy

| Aspect | Recommendation |
|--------|----------------|
| Frequency | Every 30 days |
| Data window | Most recent available data |
| Scaler | Refit on new training window |
| Model comparison | Compare new model vs. incumbent on fixed holdout before deployment |
| Versioning | Log every model in `model/training_results.json` and `MODEL_METADATA` table |

## 7.10 Overfitting Risk Assessment

**Risk Level:** ELEVATED due to 3-month dataset.

| Risk Factor | Mitigation |
|-------------|------------|
| Small dataset | Compact architecture, strong regularization |
| High feature dimensionality | Feature importance review; prune if needed |
| Deep Bi-LSTM | Limited to 2 layers |
| Attention complexity | Monitor attention entropy; simplify if it collapses |
| 24h horizon | Lowest loss weight; accept higher error |

## 7.11 Expected Training Output

```
Epoch 1/80   train_loss=0.1050 val_loss=0.0950 lr=1.00e-03
Epoch 10/80  train_loss=0.0650 val_loss=0.0650 lr=1.00e-03
Epoch 28/80  train_loss=0.0420 val_loss=0.0530 lr=5.00e-04 ← LR drops
Epoch 45/80  train_loss=0.0380 val_loss=0.0510 lr=2.50e-04 ← best epoch
Epoch 68/80  Early stopping triggered
```

The gap between training and validation loss after the best epoch is expected on a small dataset. In the full 3-month run, the gap is smaller.

---

*Next: read [`08_ANOMALY_DETECTION.md`](08_ANOMALY_DETECTION.md) for the real-time monitoring pipeline.*
