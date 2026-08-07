# 06 — Bi-LSTM Model Architecture

This document specifies the neural network architecture, layer configuration, attention mechanism, output heads, loss function, and optimizer settings.

## 6.1 Input Specification

| Attribute | Value |
|-----------|-------|
| Sequence length | 2016 time steps |
| Resolution | 5 minutes |
| Coverage | 7 days |
| Input shape | `(batch_size, 2016, F)` where F ≈ 56 |
| Output shape | `(batch_size, 6)` |

## 6.2 Output Targets

The model predicts PM2.5 and PM10 at three horizons simultaneously:

```python
TARGET_ORDER = [
    "pm2_5_6h", "pm10_6h",
    "pm2_5_12h", "pm10_12h",
    "pm2_5_24h", "pm10_24h",
]
```

## 6.3 Layer Stack

```
Input (batch, 2016, F)
    ↓
Bidirectional LSTM — 128 units, return_sequences=True
    ↓
Recurrent Dropout — 0.25
    ↓
Bidirectional LSTM — 64 units, return_sequences=True
    ↓
Recurrent Dropout — 0.25
    ↓
Temporal Pattern Attention (optional)
    ↓
Global Average Pooling across time dimension
    ↓
Three Parallel Dense Heads:
      ├── Head 6h  → (pm2_5_6h,  pm10_6h)
      ├── Head 12h → (pm2_5_12h, pm10_12h)
      └── Head 24h → (pm2_5_24h, pm10_24h)
```

## 6.4 Layer Details

### Bidirectional LSTM Layer 1
- Units: 128 (64 forward + 64 backward)
- `return_sequences=True`
- Captures long-range dependencies in both time directions

### Bidirectional LSTM Layer 2
- Units: 64 (32 forward + 32 backward)
- `return_sequences=True`
- Refines temporal encoding for attention/pooling

### Dropout
- Rate: 0.25
- Applied recurrently after each LSTM layer
- Primary regularization mechanism

### Temporal Pattern Attention (TPA)
- Lightweight attention over the time dimension
- Helps the model emphasize critical timesteps (e.g., recent spikes)

### Parallel Dense Output Heads
- 3 independent heads, one per horizon
- Each head outputs 2 values: PM2.5 and PM10

## 6.5 Model Builder Template

Implemented in `src/pytorch_model.py`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPatternAttention(nn.Module):
    """Lightweight attention over the time dimension."""
    def __init__(self, hidden_dim: int, attention_units: int = 64):
        super().__init__()
        self.score_dense = nn.Linear(hidden_dim, attention_units)
        self.weight_dense = nn.Linear(attention_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.tanh(self.score_dense(x))
        scores = self.weight_dense(scores).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return x * weights


class ALLERTBiLSTM(nn.Module):
    """PyTorch Bi-LSTM forecasting model for PM2.5/PM10 multi-horizon prediction."""
    def __init__(self, input_size: int, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        self.use_attention = mcfg.get("use_attention", True)
        units_1 = mcfg["lstm_units_layer1"]
        units_2 = mcfg["lstm_units_layer2"]
        dropout = mcfg.get("dropout_rate", 0.25)

        self.lstm1 = nn.LSTM(input_size, units_1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(units_1 * 2, units_2, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)

        self.attention = (
            TemporalPatternAttention(units_2 * 2, mcfg.get("attention_units", 64))
            if self.use_attention else None
        )

        pooled_dim = units_2 * 2
        self.head_6h = nn.Linear(pooled_dim, 2)
        self.head_12h = nn.Linear(pooled_dim, 2)
        self.head_24h = nn.Linear(pooled_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        if self.attention is not None:
            x = self.attention(x)
        x = x.mean(dim=1)  # global average pooling
        out6 = self.head_6h(x)
        out12 = self.head_12h(x)
        out24 = self.head_24h(x)
        return torch.cat([out6, out12, out24], dim=1)


def build_bilstm(input_shape, cfg):
    seq_len, n_features = input_shape
    return ALLERTBiLSTM(n_features, cfg)
```

## 6.6 Temporal Pattern Attention (Lightweight)

Implemented in `src/pytorch_model.py` as `TemporalPatternAttention`.

```python
class TemporalPatternAttention(nn.Module):
    def __init__(self, hidden_dim: int, attention_units: int = 64):
        super().__init__()
        self.score_dense = nn.Linear(hidden_dim, attention_units)
        self.weight_dense = nn.Linear(attention_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.tanh(self.score_dense(x))
        scores = self.weight_dense(scores).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return x * weights
```

## 6.7 Composite Loss Function

Implemented in `src/pytorch_model.py`.

```python
import torch


def composite_loss(y_pred, y_true, weights=(0.5, 0.3, 0.2)):
    """
    Weighted MAE across 6h/12h/24h horizons.
    y_true and y_pred have shape (batch, 6).
    """
    weights_t = torch.tensor(weights, dtype=y_pred.dtype, device=y_pred.device)
    y_true_h = y_true.reshape(-1, 3, 2)
    y_pred_h = y_pred.reshape(-1, 3, 2)
    mae_per_horizon = torch.mean(torch.abs(y_true_h - y_pred_h), dim=-1)
    weighted = mae_per_horizon * weights_t
    return torch.mean(weighted, dim=-1).mean()
```

**Rationale for weights:**
- 6h gets 50% — highest confidence needed for immediate advisories
- 12h gets 30% — medium-horizon planning
- 24h gets 20% — expected lower accuracy, deprioritized

## 6.8 Optimizer and Training Controls

| Setting | Value | Rationale |
|---------|-------|-----------|
| Optimizer | Adam | Adaptive, well-suited for LSTMs |
| Learning rate | 0.001 | Default starting point; tuned by Optuna |
| Gradient clipping | `nn.utils.clip_grad_norm_(..., max_norm=1.0)` | Prevents explosion during PM spikes |
| L2 weight decay | 1e-4 | Reduces overfitting on 3-month dataset |
| ReduceLROnPlateau | factor=0.5, patience=5 | Finer optimization near convergence |
| Early stopping | patience=10, monitor val_loss | Prevents overfitting |

The PyTorch implementations live in `src/pytorch_training.py`.

## 6.9 Hyperparameter Search Space (Optuna)

| Hyperparameter | Range |
|----------------|-------|
| Learning rate | [1e-4, 1e-2] log uniform |
| LSTM Layer 1 units | {64, 96, 128, 192, 256} |
| LSTM Layer 2 units | {32, 48, 64, 96, 128} |
| Dropout rate | [0.1, 0.5] |
| Batch size | {16, 32, 64} |

## 6.10 Why This Architecture?

### Why Bidirectional LSTM?
Pollution buildup patterns have backward-informative context. An elevated reading at t=100 is more meaningful if t=90–99 were also elevated.

### Why Three Output Heads?
- Simultaneous multi-horizon prediction in one forward pass
- Avoids iterative error accumulation of autoregressive models
- Each head specializes for its horizon

### Why TPA?
Not all 2016 timesteps are equally relevant. TPA up-weights critical timesteps such as recent spikes.

## 6.11 Model Summary (Expected)

Using `torchinfo.summary(model, input_size=(1, 2016, 56))` or printing the model produces a layer-by-layer breakdown. The expected trainable parameter count is ~355K:

```
ALLERTBiLSTM(
  (lstm1): LSTM(56, 128, batch_first=True, bidirectional=True)
  (dropout1): Dropout(p=0.25, inplace=False)
  (lstm2): LSTM(256, 64, batch_first=True, bidirectional=True)
  (dropout2): Dropout(p=0.25, inplace=False)
  (attention): TemporalPatternAttention(
    (score_dense): Linear(in_features=128, out_features=64, bias=True)
    (weight_dense): Linear(in_features=64, out_features=1, bias=True)
  )
  (head_6h): Linear(in_features=128, out_features=2, bias=True)
  (head_12h): Linear(in_features=128, out_features=2, bias=True)
  (head_24h): Linear(in_features=128, out_features=2, bias=True)
)
Total params: ~355K
```

## 6.12 Required Training Outputs

Every training run must log to `model/training_results.json`:

```json
{
  "model_version": "bilstm_pytorch_v1",
  "framework": "PyTorch",
  "training_date": "YYYY-MM-DD",
  "hyperparameters": {
    "learning_rate": 0.001,
    "lstm_units_layer1": 128,
    "lstm_units_layer2": 64,
    "dropout_rate": 0.25,
    "batch_size": 32,
    "sequence_length": 2016,
    "early_stopping_patience": 10,
    "total_epochs_run": 87,
    "best_epoch": 77,
    "training_duration_seconds": 3420
  },
  "forecast_metrics": {
    "PM2_5": {
      "6h": {"MAE": null, "RMSE": null, "MAPE": null, "R2": null},
      "12h": {"MAE": null, "RMSE": null, "MAPE": null, "R2": null},
      "24h": {"MAE": null, "RMSE": null, "MAPE": null, "R2": null}
    },
    "PM10": {
      "6h": {"MAE": null, "RMSE": null, "MAPE": null, "R2": null},
      "12h": {"MAE": null, "RMSE": null, "MAPE": null, "R2": null},
      "24h": {"MAE": null, "RMSE": null, "MAPE": null, "R2": null}
    }
  }
}
```

The trained weights are saved as `model/best_model.pt`.

---

*Next: read [`07_TRAINING_STRATEGY.md`](07_TRAINING_STRATEGY.md) for training procedures.*
