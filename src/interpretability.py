"""ALLERT interpretability module.

Provides:
    - Permutation importance on the validation set
    - Optional SHAP summary if ``shap`` is installed
    - Attention-weight extraction/plot helper for the PyTorch attention module
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.config_loader import get_project_root
from src.evaluation import compute_metrics, _predict_in_batches
from src.pytorch_model import build_bilstm, DEVICE, load_model_with_config


def _model_predict(model, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Dispatch prediction to PyTorch module or any object with ``predict``."""
    import torch.nn as nn

    if isinstance(model, nn.Module):
        return _predict_in_batches(model, X, batch_size)
    # Fallback for Keras/sklearn-style models (used in tests)
    return np.asarray(model.predict(X))


def permutation_importance(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    baseline_mae: float,
    feature_names: List[str],
    n_repeats: int = 5,
    seed: int = 42,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Compute permutation importance by feature.

    Parameters
    ----------
    model:
        Trained PyTorch model or any object exposing ``predict(X)``.
    X_val, y_val:
        Validation sequences and targets.
    baseline_mae:
        MAE on unpermuted validation data.
    feature_names:
        Ordered list of feature names matching the last axis of ``X_val``.
    n_repeats:
        Number of permutation repeats per feature.
    seed:
        RNG seed for reproducibility.
    batch_size:
        Inference batch size (PyTorch path only).

    Returns
    -------
    dict
        Sorted importances with mean/std MAE drop.
    """
    rng = np.random.default_rng(seed)
    importances = []
    n_features = X_val.shape[2]
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match X_val features ({n_features})"
        )

    for i, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            perm_idx = rng.permutation(X_perm.shape[0])
            X_perm[:, :, i] = X_perm[perm_idx, :, i]
            y_pred = _model_predict(model, X_perm, batch_size)
            mae = float(np.mean(np.abs(y_val - y_pred)))
            drops.append(mae - baseline_mae)
        importances.append({
            "feature": name,
            "mae_drop_mean": float(np.mean(drops)),
            "mae_drop_std": float(np.std(drops)),
        })

    importances.sort(key=lambda x: x["mae_drop_mean"], reverse=True)
    return {
        "baseline_mae": float(baseline_mae),
        "n_repeats": n_repeats,
        "importances": importances,
    }


def extract_attention_weights(model, X_sample: np.ndarray) -> Optional[np.ndarray]:
    """Extract attention weights from the PyTorch attention module.

    Returns
    -------
    np.ndarray or None
        Attention weights of shape ``(n_samples, seq_len)`` if the model uses
        attention, otherwise None.
    """
    if not hasattr(model, "attention") or model.attention is None:
        return None

    model.eval()
    weights = []
    with torch.no_grad():
        for i in range(0, len(X_sample), 64):
            xb = torch.from_numpy(np.asarray(X_sample[i : i + 64], dtype=np.float32)).to(DEVICE)
            # Forward through LSTM stack to get attended input
            x = model._lstm_stack(xb)
            scores = torch.tanh(model.attention.score_dense(x))
            scores = model.attention.weight_dense(scores).squeeze(-1)
            w = torch.softmax(scores, dim=1).cpu().numpy()
            weights.append(w)
    return np.concatenate(weights, axis=0)


def plot_attention_weights(
    attention_weights: np.ndarray,
    out_path: Path,
    n_show: int = 200,
) -> Path:
    """Plot mean attention weight over time steps for a sample of sequences."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mean_weights = attention_weights[:n_show].mean(axis=0)
    x = np.arange(len(mean_weights))

    plt.figure(figsize=(12, 4))
    plt.plot(x, mean_weights, color="steelblue", linewidth=1.5)
    plt.fill_between(x, mean_weights, alpha=0.3, color="steelblue")
    plt.xlabel("Time step")
    plt.ylabel("Mean attention weight")
    plt.title("Fig 4.8d — Temporal Pattern Attention Weights")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_permutation_importance(
    importances: List[Dict[str, Any]],
    out_path: Path,
    top_n: int = 20,
) -> Path:
    """Generate a horizontal bar chart of permutation importances."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top = importances[:top_n]
    names = [x["feature"] for x in top]
    means = [x["mae_drop_mean"] for x in top]
    stds = [x["mae_drop_std"] for x in top]

    y = np.arange(len(names))
    plt.figure(figsize=(10, max(4, 0.4 * len(names) + 2)))
    plt.barh(y, means, xerr=stds, color="steelblue", edgecolor="white")
    plt.yticks(y, names)
    plt.xlabel("MAE increase (µg/m³)")
    plt.title(f"Fig 4.8b — Permutation Importance (top {top_n})")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def run_interpretability(
    cfg: Dict[str, Any],
    out_dir: Optional[Path] = None,
    n_repeats: int = 3,
    seed: int = 42,
    max_val_samples: int = 100,
) -> Dict[str, Any]:
    """Run permutation importance and attention extraction on a validation sample."""
    from src import sequence_builder as sb

    if out_dir is None:
        out_dir = Path(cfg["paths"]["results_dir"])
        if not out_dir.is_absolute():
            out_dir = get_project_root() / out_dir
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    viz_dir = out_dir / "visualizations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    results_dir = get_project_root() / cfg["paths"]["results_dir"]
    X, y, ts, train_mask, val_mask, test_mask = sb.load_sequences(results_dir)
    model_dir = get_project_root() / cfg["paths"]["model_dir"]
    model, _ = load_model_with_config(
        model_dir, (X.shape[1], X.shape[2]), fallback_cfg=cfg, device=DEVICE
    )

    # Subsample validation for speed
    val_idx = np.where(val_mask)[0]
    if len(val_idx) > max_val_samples:
        rng = np.random.default_rng(seed)
        val_idx = rng.choice(val_idx, size=max_val_samples, replace=False)
    X_val = X[val_idx]
    y_val = y[val_idx]

    # Baseline MAE
    y_pred_baseline = _predict_in_batches(model, X_val, cfg["model"]["batch_size"])
    baseline_mae = float(np.mean(np.abs(y_val - y_pred_baseline)))

    # Load feature names
    feature_cols_path = get_project_root() / cfg["paths"]["model_dir"] / "feature_cols.json"
    with open(feature_cols_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    perm = permutation_importance(
        model, X_val, y_val, baseline_mae, feature_names,
        n_repeats=n_repeats, seed=seed, batch_size=cfg["model"]["batch_size"],
    )

    # Attention weights
    attn = None
    attn_weights = extract_attention_weights(model, X_val[:min(50, len(X_val))])
    if attn_weights is not None:
        attn = {
            "shape": list(attn_weights.shape),
            "mean": float(attn_weights.mean()),
            "max": float(attn_weights.max()),
            "path": str(plot_attention_weights(attn_weights, viz_dir / "fig_4_8d_attention_weights.png")),
        }

    # Optional SHAP (skip cleanly if not installed; PyTorch DeepExplainer is supported)
    shap_result = None
    try:
        import shap
        background = torch.from_numpy(np.asarray(X_val[:min(10, len(X_val))], dtype=np.float32)).to(DEVICE)
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(
            torch.from_numpy(np.asarray(X_val[:min(30, len(X_val))], dtype=np.float32)).to(DEVICE)
        )
        shap_result = {
            "type": "deep_explainer",
            "samples": min(30, len(X_val)),
            "shap_values_shape": [np.asarray(sv).shape for sv in shap_values] if isinstance(shap_values, list) else str(type(shap_values)),
        }
    except Exception as exc:
        shap_result = {"type": "skipped", "reason": str(exc)}

    report = {
        "permutation_importance": perm,
        "attention_weights": attn,
        "shap": shap_result,
    }

    with open(tables_dir / "table_4_8b_permutation_importance.json", "w", encoding="utf-8") as f:
        json.dump(perm, f, indent=2)

    plot_permutation_importance(perm["importances"], viz_dir / "fig_4_8b_permutation_importance.png")

    return report


__all__ = [
    "permutation_importance",
    "extract_attention_weights",
    "plot_attention_weights",
    "plot_permutation_importance",
    "run_interpretability",
]
