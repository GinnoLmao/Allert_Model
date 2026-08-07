"""ALLERT PyTorch training module.

Trains the Bi-LSTM on a held-out validation set with early stopping,
learning-rate reduction, and **automatic checkpoint/resume** so that long
runs can be paused and continued without retraining.

Artifacts produced:
    - ``model/best_model.pt``
    - ``model/training_results.json``
    - ``model/training_progress.json`` (if enabled)
    - ``model/checkpoints/checkpoint_epoch_*.pt`` (if enabled)
"""
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

from src.config_loader import get_project_root
from src.pytorch_model import build_bilstm, set_random_seeds, composite_loss, DEVICE
from src.training import (
    augment_training_data,
    walk_forward_splits,
    build_curriculum_y,
)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def get_checkpoint_dir(cfg: Dict[str, Any]) -> Path:
    checkpoint_dir = Path(cfg["training"].get("checkpoint", {}).get("checkpoint_dir", "model/checkpoints"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = get_project_root() / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_epoch: int,
    best_val_loss: float,
    history: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    scaler: Optional[GradScaler] = None,
    best_state_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a full training checkpoint (model + optimizer + scheduler + state)."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "model_state_dict": model.state_dict(),
        "best_model_state_dict": best_state_dict if best_state_dict is not None else {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": cfg,
        "timestamp": datetime.now().isoformat(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(state, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint and restore RNG state."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])
    rng_state = state["rng_state"]
    if isinstance(rng_state, torch.Tensor):
        rng_state = rng_state.cpu().byte()
    torch.set_rng_state(rng_state)
    if torch.cuda.is_available() and state.get("cuda_rng_state"):
        cuda_states = state["cuda_rng_state"]
        if isinstance(cuda_states, list):
            cuda_states = [s.cpu().byte() if isinstance(s, torch.Tensor) else s for s in cuda_states]
        torch.cuda.set_rng_state_all(cuda_states)
    return state


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    """Keep only the most recent N checkpoints to save disk space."""
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) > keep_last_n:
        for old in checkpoints[:-keep_last_n]:
            try:
                old.unlink()
            except Exception:
                pass


def _safe_torch_save(obj, path: Path) -> None:
    """Save a PyTorch object, removing any existing file first on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass
    torch.save(obj, path)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)


class MemmapDataset(TensorDataset):
    """Lazy PyTorch Dataset that reads sequences from a memory-mapped array.

    Avoids materializing the entire train/val split in RAM, which is essential
    for long sequences (2016 steps) on machines with limited memory.
    """

    def __init__(self, X, y, indices: np.ndarray):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        self._len = len(self.indices)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        x = torch.from_numpy(np.array(self.X[i], dtype=np.float32, copy=True))
        y = torch.from_numpy(np.array(self.y[i], dtype=np.float32, copy=True))
        return x, y


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    indices: np.ndarray = None,
    shuffle: bool = False,
) -> DataLoader:
    if indices is not None:
        dataset = MemmapDataset(X, y, indices)
    else:
        dataset = TensorDataset(
            torch.from_numpy(np.asarray(X, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def make_spike_targets(y: torch.Tensor, threshold: float) -> torch.Tensor:
    """Build binary spike targets for PM2.5 at 6h, 12h, 24h horizons.

    y layout: [pm2_5_6h, pm10_6h, pm2_5_12h, pm10_12h, pm2_5_24h, pm10_24h].
    Spike target: [pm2_5_spike_6h, pm2_5_spike_12h, pm2_5_spike_24h].
    """
    pm25_idx = [0, 2, 4]
    return (y[:, pm25_idx] > threshold).float()


def _build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = cfg["model"]["learning_rate"]
    name = cfg["model"].get("optimizer", "adam").lower()
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=cfg["model"].get("weight_decay", 1e-4))
    return Adam(model.parameters(), lr=lr)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    steps_per_epoch: int,
) -> Optional[Any]:
    name = cfg["model"].get("scheduler", "plateau").lower()
    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg["training"]["lr_reduction_factor"],
            patience=cfg["training"]["lr_reduction_patience"],
            min_lr=cfg["training"]["min_lr"],
        )
    if name == "cosine":
        epochs = cfg["model"].get("scheduler_max_epochs", cfg["model"]["max_epochs"])
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg["training"]["min_lr"])
    if name == "onecycle":
        epochs = cfg["model"].get("scheduler_max_epochs", cfg["model"]["max_epochs"])
        return OneCycleLR(
            optimizer,
            max_lr=cfg["model"]["learning_rate"],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )
    return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(
    cfg: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    model_path: Optional[Path] = None,
    max_epochs: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Train the PyTorch Bi-LSTM with held-out validation and checkpoint/resume.

    Returns
    -------
    history:
        List of per-epoch metric dictionaries.
    results:
        Dictionary of training metadata and metrics.
    """
    rep = cfg.get("reproducibility", {})
    set_random_seeds(
        cfg,
        seed=rep.get("random_seed"),
        deterministic=rep.get("deterministic_mode", False),
    )

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("Train or validation split is empty — cannot train")

    # Apply augmentation to a small in-memory sample for memory safety.
    # For the full training loop we read lazily from the memmap.
    aug_cfg = cfg["training"].get("data_augmentation", {})
    if aug_cfg.get("enabled", False):
        # NOTE: in-memory augmentation of the full training set is incompatible
        # with lazy memmap loading. If augmentation is required, the data must
        # be materialized (requires ~6 GB RAM on this dataset). For now we
        # warn and keep the lazy loader, which is the default configuration.
        print("[WARNING] Data augmentation is enabled but will be skipped in lazy-memmap mode.")

    seq_len = X.shape[1]
    n_features = X.shape[2]

    model = build_bilstm((seq_len, n_features), cfg).to(DEVICE)

    if model_path is None:
        model_path = Path(cfg["paths"]["model_dir"]) / "best_model.pt"
    if not model_path.is_absolute():
        model_path = get_project_root() / model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_cfg = cfg["training"].get("checkpoint", {})
    checkpoint_enabled = checkpoint_cfg.get("enabled", True)
    checkpoint_dir = get_checkpoint_dir(cfg)
    checkpoint_freq = checkpoint_cfg.get("checkpoint_frequency_epochs", 1)
    keep_last_n = checkpoint_cfg.get("keep_last_n", 3)
    resume_if_exists = checkpoint_cfg.get("resume_if_checkpoint_exists", True)

    epochs = max_epochs if max_epochs is not None else cfg["model"]["max_epochs"]
    batch_size = cfg["model"]["batch_size"]
    learning_rate = cfg["model"]["learning_rate"]
    gradient_clip_norm = cfg["model"]["gradient_clip_norm"]
    use_amp = cfg["model"].get("mixed_precision", False) and torch.cuda.is_available()
    use_spike_head = cfg["model"].get("use_spike_head", False)
    spike_threshold = cfg["model"].get("spike_threshold", 35.0)
    spike_weight = cfg["model"].get("spike_head_weight", 0.3)
    loss_type = cfg["model"].get("loss_type", "mae")
    loss_alpha = cfg["model"].get("loss_alpha", 0.0)

    train_loader = make_dataloader(X, y, batch_size, indices=train_indices, shuffle=True)
    val_loader = make_dataloader(X, y, batch_size, indices=val_indices, shuffle=False)

    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scheduler_name = cfg["model"].get("scheduler", "plateau").lower()
    scaler = _grad_scaler(use_amp) if use_amp else None

    monitor = cfg["training"]["early_stopping_monitor"]
    patience = cfg["training"]["early_stopping_patience"]
    restore_best_weights = cfg["training"]["restore_best_weights"]

    history: List[Dict[str, Any]] = []
    start_epoch = 0
    best_epoch = 0
    epochs_no_improve = 0
    best_val_loss = float("inf")
    best_state_dict = None

    # Resume from latest checkpoint if available
    latest_checkpoint = None
    if resume_if_exists and checkpoint_enabled and epochs > 0:
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"[RESUME] Loading checkpoint: {latest_checkpoint}")
            try:
                state = load_checkpoint(latest_checkpoint, model, optimizer, scheduler, DEVICE, scaler=scaler)
                start_epoch = state["epoch"] + 1
                best_epoch = state["best_epoch"]
                best_val_loss = state["best_val_loss"]
                history = state["history"]
                if "best_model_state_dict" in state:
                    best_state_dict = state["best_model_state_dict"]
                    print(f"[RESUME] Restored best weights from epoch {best_epoch} (val_loss={best_val_loss:.6f})")
                else:
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = start_epoch - best_epoch
                print(f"[RESUME] Resuming at epoch {start_epoch + 1}, best val loss so far: {best_val_loss:.6f}")
            except RuntimeError as e:
                print(f"[RESUME] Checkpoint incompatible with current model ({e}); starting fresh.")
                start_epoch = 0

    start_time = time.time()

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            train_losses = []
            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} train", leave=False):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                # Curriculum learning: mask longer horizons in early epochs
                yb = build_curriculum_y(yb.cpu().numpy(), cfg, epoch)
                yb = _to_tensor(yb, DEVICE)
                optimizer.zero_grad()
                with _autocast(enabled=use_amp):
                    out = model(xb)
                    if use_spike_head:
                        pred, spike_logits = out
                        spike_true = make_spike_targets(yb, spike_threshold)
                        loss = composite_loss(
                            pred, yb,
                            weights=tuple(cfg["model"]["loss_weights"]),
                            spike_pred=spike_logits,
                            spike_true=spike_true,
                            spike_weight=spike_weight,
                            loss_type=loss_type,
                            spike_threshold=spike_threshold,
                            loss_alpha=loss_alpha,
                        )
                    else:
                        pred = out
                        loss = composite_loss(
                            pred, yb,
                            weights=tuple(cfg["model"]["loss_weights"]),
                            loss_type=loss_type,
                            spike_threshold=spike_threshold,
                            loss_alpha=loss_alpha,
                        )
                loss = loss + model.l2_regularization()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()
                if scheduler_name == "onecycle":
                    scheduler.step()
                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} val", leave=False):
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    with _autocast(enabled=use_amp):
                        out = model(xb)
                        if use_spike_head:
                            pred, spike_logits = out
                            spike_true = make_spike_targets(yb, spike_threshold)
                            loss = composite_loss(
                                pred, yb,
                                weights=tuple(cfg["model"]["loss_weights"]),
                                spike_pred=spike_logits,
                                spike_true=spike_true,
                                spike_weight=spike_weight,
                                loss_type=loss_type,
                                spike_threshold=spike_threshold,
                                loss_alpha=loss_alpha,
                            )
                        else:
                            pred = out
                            loss = composite_loss(
                                pred, yb,
                                weights=tuple(cfg["model"]["loss_weights"]),
                                loss_type=loss_type,
                                spike_threshold=spike_threshold,
                                loss_alpha=loss_alpha,
                            )
                    val_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            current_lr = optimizer.param_groups[0]["lr"]

            entry = {
                "epoch": epoch + 1,
                "loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "timestamp": datetime.now().isoformat(),
            }
            history.append(entry)

            # Optional progress logger
            progress_cfg = cfg["training"].get("progress_log", {})
            if progress_cfg.get("enabled", False):
                progress_path = Path(progress_cfg.get("path", "model/training_progress.json"))
                if not progress_path.is_absolute():
                    progress_path = get_project_root() / progress_path
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)

            print(
                f"Epoch {epoch + 1}/{epochs} — train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} lr={current_lr:.2e}"
            )

            # Scheduler step
            if scheduler_name == "plateau":
                scheduler.step(val_loss)
            elif scheduler_name in {"cosine"}:
                scheduler.step()

            # Early stopping
            improved = val_loss < best_val_loss - 1e-9
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_no_improve = 0
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                _safe_torch_save(model.state_dict(), model_path)
            else:
                epochs_no_improve += 1

            # Periodic checkpoint (always save the most recent state)
            if checkpoint_enabled and (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_epoch,
                    best_val_loss,
                    history,
                    cfg,
                    scaler=scaler,
                    best_state_dict=best_state_dict,
                )
                cleanup_old_checkpoints(checkpoint_dir, keep_last_n)

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs (patience={patience}).")
                break

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Keyboard interrupt detected. Saving checkpoint before exit...")
        if checkpoint_enabled:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}_interrupted.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_epoch,
                best_val_loss,
                history,
                cfg,
                scaler=scaler,
                best_state_dict=best_state_dict,
            )
        raise

    duration = time.time() - start_time

    # Restore best weights if requested and available
    if restore_best_weights and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        _safe_torch_save(model.state_dict(), model_path)
        print(f"Restored best weights from epoch {best_epoch} and saved to {model_path}")

    results = {
        "model_version": "bilstm_pytorch_v1",
        "framework": "PyTorch",
        "device": str(DEVICE),
        "training_date": datetime.now().isoformat(),
        "hyperparameters": {
            "learning_rate": learning_rate,
            "n_lstm_layers": cfg["model"].get("n_lstm_layers", 2),
            "lstm_units_layer1": cfg["model"]["lstm_units_layer1"],
            "lstm_units_layer2": cfg["model"]["lstm_units_layer2"],
            "lstm_units_layer3": cfg["model"].get("lstm_units_layer3"),
            "lstm_units_layer4": cfg["model"].get("lstm_units_layer4"),
            "dropout_rate": cfg["model"]["dropout_rate"],
            "recurrent_dropout": cfg["model"]["recurrent_dropout"],
            "l2_lambda": cfg["model"]["l2_lambda"],
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "n_features": n_features,
            "max_epochs": epochs,
            "early_stopping_patience": patience,
            "early_stopping_monitor": monitor,
            "use_attention": cfg["model"].get("use_attention", True),
            "attention_type": cfg["model"].get("attention_type", "vector"),
            "use_spike_head": use_spike_head,
            "spike_threshold": spike_threshold,
            "spike_head_weight": spike_weight,
            "optimizer": cfg["model"].get("optimizer", "adam"),
            "scheduler": scheduler_name,
            "augmentation_enabled": aug_cfg.get("enabled", False),
            "walk_forward_cv_enabled": cfg["training"].get("walk_forward_cv", {}).get("enabled", False),
            "curriculum_enabled": cfg["training"].get("curriculum", {}).get("enabled", False),
        },
        "total_epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_duration_seconds": round(duration, 2),
        "history": history,
        "model_path": str(model_path),
    }

    # Save the config used to build this model so evaluation/interpretability can
    # reconstruct the exact architecture (important when quick mode overrides sizes).
    config_path = Path(cfg["paths"]["model_dir"]) / "model_config.json"
    if not config_path.is_absolute():
        config_path = get_project_root() / config_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)

    results_path = Path(cfg["paths"]["model_dir"]) / "training_results.json"
    if not results_path.is_absolute():
        results_path = get_project_root() / results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return history, results


def run_walk_forward_training(
    cfg: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    max_epochs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Train a model for each walk-forward split and return fold results."""
    splits = walk_forward_splits(X, cfg)
    fold_results = []
    for fold_i, (train_mask, val_mask) in enumerate(splits, start=1):
        print(f"\n=== Walk-forward fold {fold_i}/{len(splits)} ===")
        fold_path = Path(cfg["paths"]["model_dir"]) / f"fold_{fold_i}_model.pt"
        _, results = train_model(
            cfg, X, y, train_mask, val_mask,
            model_path=fold_path,
            max_epochs=max_epochs,
        )
        fold_results.append({"fold": fold_i, **results})

    # Copy last fold to best_model.pt
    last_fold = Path(cfg["paths"]["model_dir"]) / f"fold_{len(splits)}_model.pt"
    best_path = Path(cfg["paths"]["model_dir"]) / "best_model.pt"
    if last_fold.exists():
        if best_path.exists():
            try:
                best_path.unlink()
            except Exception:
                pass
        shutil.copy(str(last_fold), str(best_path))

    return fold_results


__all__ = [
    "train_model",
    "run_walk_forward_training",
    "save_checkpoint",
    "load_checkpoint",
    "get_checkpoint_dir",
]
