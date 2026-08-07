"""ALLERT Optuna hyperparameter optimization module (PyTorch).

Provides a Bayesian search over the model hyperparameter space defined in
``config.yaml``. The objective trains a candidate model for a small number of
epochs and returns the best validation loss. This is intended primarily for
GPU-backed runs; CPU runs are very slow.

Artifacts:
    - ``model/optuna_best_params.json``
    - ``results/optuna_study.pkl``
    - ``results/visualizations/fig_4_4_hpo_optimization_history.png``
    - ``results/visualizations/fig_4_4b_hpo_param_importance.png``
"""
import json
import copy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset

# Mixed-precision helpers: use the PyTorch 2.4+ API when available,
# fall back to the older cuda.amp API for torch>=2.1.
try:
    from torch.amp import autocast, GradScaler

    def _autocast(enabled: bool):
        return autocast("cuda", enabled=enabled)

    def _grad_scaler(enabled: bool):
        return GradScaler("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

    def _autocast(enabled: bool):
        return autocast(enabled=enabled)

    def _grad_scaler(enabled: bool):
        return GradScaler(enabled=enabled)
from src.config_loader import get_project_root
from src.pytorch_model import build_bilstm, set_random_seeds, composite_loss, DEVICE
from src.pytorch_training import (
    MemmapDataset,
    make_spike_targets,
    _build_optimizer,
    _build_scheduler,
)
from src.training import augment_training_data


def _get_sampler(name: Optional[str], seed: int):
    """Return an Optuna sampler by name."""
    name = (name or "tpe").lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    raise ValueError(f"Unknown HPO sampler: {name}")


def _get_pruner(name: Optional[str]):
    """Return an Optuna pruner by name."""
    name = (name or "median").lower()
    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "halving":
        return optuna.pruners.SuccessiveHalvingPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    if name in {"none", "null"}:
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unknown HPO pruner: {name}")


def _build_cfg_for_trial(cfg: Dict[str, Any], trial: optuna.Trial) -> Dict[str, Any]:
    """Create a config copy with hyperparameters sampled by Optuna."""
    trial_cfg = copy.deepcopy(cfg)
    space = cfg["hpo"]["search_space"]

    trial_cfg["model"]["learning_rate"] = trial.suggest_float(
        "learning_rate", *space["learning_rate"], log=True
    )

    if "n_lstm_layers" in space:
        trial_cfg["model"]["n_lstm_layers"] = trial.suggest_categorical(
            "n_lstm_layers", space["n_lstm_layers"]
        )

    trial_cfg["model"]["lstm_units_layer1"] = trial.suggest_categorical(
        "lstm_units_layer1", space["lstm_units_layer1"]
    )
    trial_cfg["model"]["lstm_units_layer2"] = trial.suggest_categorical(
        "lstm_units_layer2", space["lstm_units_layer2"]
    )

    n_layers = int(trial_cfg["model"].get("n_lstm_layers", 2))
    for i in range(3, n_layers + 1):
        key = f"lstm_units_layer{i}"
        if key in space:
            trial_cfg["model"][key] = trial.suggest_categorical(key, space[key])
        else:
            trial_cfg["model"][key] = trial_cfg["model"]["lstm_units_layer2"]

    trial_cfg["model"]["dropout_rate"] = trial.suggest_float(
        "dropout_rate", *space["dropout_rate"]
    )

    if "recurrent_dropout_rate" in space and space["recurrent_dropout_rate"] is not None:
        trial_cfg["model"]["recurrent_dropout"] = trial.suggest_float(
            "recurrent_dropout_rate", *space["recurrent_dropout_rate"]
        )
    else:
        trial_cfg["model"]["recurrent_dropout"] = trial_cfg["model"]["dropout_rate"]

    trial_cfg["model"]["batch_size"] = trial.suggest_categorical(
        "batch_size", space["batch_size"]
    )

    if "l2_lambda" in space:
        trial_cfg["model"]["l2_lambda"] = trial.suggest_float(
            "l2_lambda", *space["l2_lambda"], log=True
        )

    return trial_cfg


def objective(
    trial: optuna.Trial,
    cfg: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    max_epochs: int = 30,
) -> float:
    """Optuna objective: train a candidate and return validation loss."""
    trial_cfg = _build_cfg_for_trial(cfg, trial)
    rep = trial_cfg.get("reproducibility", {})
    set_random_seeds(
        trial_cfg,
        seed=rep.get("random_seed"),
        deterministic=rep.get("deterministic_mode", False),
    )

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    # Optional subset for fast HPO while keeping chronological split ratios
    subset_frac = cfg.get("hpo", {}).get("subset_fraction")
    if subset_frac is not None and 0 < subset_frac < 1.0:
        rng = np.random.default_rng(seed=cfg.get("hpo", {}).get("seed", 42) + trial.number)
        train_indices = rng.choice(train_indices, size=int(len(train_indices) * subset_frac), replace=False)
        val_indices = rng.choice(val_indices, size=int(len(val_indices) * subset_frac), replace=False)
        print(f"[HPO subset] trial {trial.number}: train={len(train_indices)}, val={len(val_indices)}")

    aug_cfg = trial_cfg["training"].get("data_augmentation", {})
    if aug_cfg.get("enabled", False):
        print("[WARNING] Data augmentation is enabled but will be skipped in lazy-memmap HPO mode.")

    model = build_bilstm((X.shape[1], X.shape[2]), trial_cfg).to(DEVICE)

    batch_size = trial_cfg["model"]["batch_size"]
    train_dataset = MemmapDataset(X, y, train_indices)
    val_dataset = MemmapDataset(X, y, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = _build_optimizer(model, trial_cfg)
    scheduler = _build_scheduler(optimizer, trial_cfg, steps_per_epoch=len(train_loader))
    scheduler_name = trial_cfg["model"].get("scheduler", "plateau").lower()

    use_spike_head = trial_cfg["model"].get("use_spike_head", False)
    spike_threshold = trial_cfg["model"].get("spike_threshold", 35.0)
    spike_weight = trial_cfg["model"].get("spike_head_weight", 0.3)
    loss_type = trial_cfg["model"].get("loss_type", "mae")
    loss_alpha = trial_cfg["model"].get("loss_alpha", 0.0)

    patience = trial_cfg["hpo"].get("early_stopping_patience", 5)
    weights = tuple(trial_cfg["model"]["loss_weights"])
    gradient_clip_norm = trial_cfg["model"]["gradient_clip_norm"]
    use_amp = trial_cfg["model"].get("mixed_precision", False) and torch.cuda.is_available()
    scaler = _grad_scaler(use_amp) if use_amp else None

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with _autocast(enabled=use_amp):
                out = model(xb)
                if use_spike_head:
                    pred, spike_logits = out
                    spike_true = make_spike_targets(yb, spike_threshold)
                    loss = composite_loss(
                        pred, yb, weights=weights,
                        spike_pred=spike_logits, spike_true=spike_true,
                        spike_weight=spike_weight,
                        loss_type=loss_type,
                        spike_threshold=spike_threshold,
                        loss_alpha=loss_alpha,
                    )
                else:
                    pred = out
                    loss = composite_loss(
                        pred, yb, weights=weights,
                        loss_type=loss_type,
                        spike_threshold=spike_threshold,
                        loss_alpha=loss_alpha,
                    )
            loss = loss + model.l2_regularization()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            if scheduler_name == "onecycle":
                scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with _autocast(enabled=use_amp):
                    out = model(xb)
                    if use_spike_head:
                        pred, spike_logits = out
                        spike_true = make_spike_targets(yb, spike_threshold)
                        loss = composite_loss(
                            pred, yb, weights=weights,
                            spike_pred=spike_logits, spike_true=spike_true,
                            spike_weight=spike_weight,
                            loss_type=loss_type,
                            spike_threshold=spike_threshold,
                            loss_alpha=loss_alpha,
                        )
                    else:
                        pred = out
                        loss = composite_loss(
                            pred, yb, weights=weights,
                            loss_type=loss_type,
                            spike_threshold=spike_threshold,
                            loss_alpha=loss_alpha,
                        )
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        if scheduler_name == "plateau":
            scheduler.step(val_loss)
        elif scheduler_name in {"cosine"}:
            scheduler.step()

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    return float(best_val_loss)


def plot_hpo_results(study: optuna.Study, out_dir: Path) -> Dict[str, Optional[Path]]:
    """Generate HPO visualization figures."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.figure.savefig(out_dir / "fig_4_4_hpo_optimization_history.png", dpi=150, bbox_inches="tight")
        plt.close(fig.figure)
        paths["optimization_history"] = out_dir / "fig_4_4_hpo_optimization_history.png"
    except Exception:
        paths["optimization_history"] = None

    try:
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.figure.savefig(out_dir / "fig_4_4b_hpo_param_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig.figure)
        paths["param_importance"] = out_dir / "fig_4_4b_hpo_param_importance.png"
    except Exception:
        paths["param_importance"] = None

    return paths


def run_hpo(
    cfg: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    n_trials: Optional[int] = None,
    max_epochs: int = None,
) -> Dict[str, Any]:
    """Run the Optuna study and save the best parameters."""
    if n_trials is None:
        n_trials = cfg["hpo"]["n_trials"]
    if max_epochs is None:
        max_epochs = cfg["hpo"].get("max_epochs_per_trial", 30)

    hpo_cfg = cfg.get("hpo", {})
    seed = hpo_cfg.get("seed", cfg.get("reproducibility", {}).get("random_seed", 42))

    study = optuna.create_study(
        direction=hpo_cfg.get("direction", "minimize"),
        sampler=_get_sampler(hpo_cfg.get("sampler"), seed),
        pruner=_get_pruner(hpo_cfg.get("pruner")),
        study_name=hpo_cfg.get("study_name", "allert_hpo"),
        storage=hpo_cfg.get("storage"),
        load_if_exists=hpo_cfg.get("storage") is not None,
    )
    timeout = hpo_cfg.get("timeout_seconds")
    study.optimize(
        lambda trial: objective(trial, cfg, X, y, train_mask, val_mask, max_epochs),
        n_trials=n_trials,
        timeout=timeout,
    )

    if len(study.trials) == 0:
        print("[HPO] No trials completed in this session. The study is saved and can be resumed.")
        return {"completed_trials": 0, "message": "No trials completed yet"}

    best = study.best_params
    best["val_loss"] = study.best_value

    model_dir = Path(cfg["paths"]["model_dir"])
    if not model_dir.is_absolute():
        model_dir = get_project_root() / model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    results_dir = Path(cfg["paths"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = get_project_root() / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    study_path = results_dir / "optuna_study.pkl"
    joblib_dump = __import__("joblib")
    joblib_dump.dump(study, study_path)

    summary = {
        "n_trials": n_trials,
        "max_epochs_per_trial": max_epochs,
        "timeout_seconds": timeout,
        "completed_trials": len(study.trials),
        "direction": hpo_cfg.get("direction", "minimize"),
        "sampler": hpo_cfg.get("sampler", "tpe"),
        "pruner": hpo_cfg.get("pruner", "median"),
        "best_params": best,
        "best_trial_number": study.best_trial.number,
    }
    with open(results_dir / "tables" / "table_4_4_hpo_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    viz_paths = plot_hpo_results(study, results_dir / "visualizations")
    summary["viz_paths"] = {k: str(v) for k, v in viz_paths.items() if v is not None}

    return summary


__all__ = ["run_hpo", "objective", "plot_hpo_results"]
