"""ALLERT end-to-end orchestrator.

CLI entry point that ties the modular PyTorch pipeline together:

    python src/orchestrator.py --mode train      # preprocess + features + scale + sequences + train
    python src/orchestrator.py --mode evaluate   # evaluate model + anomaly + risk + validity
    python src/orchestrator.py --mode full       # train then evaluate

A ``--quick`` flag is provided for local smoke tests; it trains a tiny model
for a small number of epochs so the pipeline can be verified without a GPU.
For real experiments, use the full configuration and GPU acceleration.
Training is checkpointed automatically so long runs can be paused and resumed.
"""
import argparse
import copy
import sys
from pathlib import Path

# Allow running this script directly as `python src/orchestrator.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config_loader import load_config
from src import preprocessing as pp
from src import feature_engineering as fe
from src import scaling
from src import sequence_builder as sb
from src import training
from src import evaluation as ev
from src import anomaly_detection as ad
from src import risk_engine as re
from src import validity
from src import hpo
from src import risk_sensitivity as rs
from src import interpretability as interp
from src.utils import timing


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def run_preprocessing(cfg, subset_days: int = None):
    print("\n=== Phase 2: Preprocessing ===")
    if subset_days:
        print(f"[SUBSET] Using the most recent {subset_days} days of raw data.")
    df, stats = pp.preprocess(cfg, subset_days=subset_days)
    pp.save_processed_data(df, cfg)
    pp.export_dataset_statistics(stats, cfg)
    print("Preprocessing complete.")
    return df


def run_feature_engineering(cfg, df):
    print("\n=== Phase 3: Feature Engineering ===")
    features = fe.build_feature_matrix(df, cfg)
    cols = fe.get_feature_column_names(features)
    import json
    from src.config_loader import get_project_root
    model_dir = get_project_root() / cfg["paths"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(cols, f, indent=2)
    print(f"Feature matrix: {features.shape} -> saved {len(cols)} feature columns")
    return features, cols


def run_scaling(cfg):
    print("\n=== Phase 4: Scaling & Splits ===")
    full, train, val, test, scaler, target_scaler = scaling.build_scaled_dataset(cfg)
    from src.config_loader import get_project_root
    root = get_project_root()
    scaler.save(root / cfg["paths"]["model_dir"] / "scaler.joblib")
    target_scaler.save(root / cfg["paths"]["model_dir"] / "target_scaler.joblib")
    full.to_csv(root / cfg["paths"]["data_processed"])
    scaling.save_split_indices(
        len(full), train, val,
        root / "data" / "splits" / "train_val_test_indices.json",
    )
    print(f"Scaler fitted on {len(train)} training rows; saved scaled dataset.")
    print(f"Target scaling enabled={target_scaler.scaled}; metrics will be reported in original µg/m³.")


def run_sequence_builder(cfg, stride: int = 1):
    print("\n=== Phase 5: Sequence Builder ===")
    if stride != 1:
        print(f"[STRIDE] Building sequences with stride={stride}.")
    X, y, ts, train_mask, val_mask, test_mask, meta = sb.build_sequences(cfg, stride=stride)
    print(f"Sequences: X={X.shape}, y={y.shape}")
    print(f"Split counts: train={meta['train_count']}, val={meta['val_count']}, test={meta['test_count']}")


def _load_sequences(cfg):
    from src.config_loader import get_project_root
    results_dir = get_project_root() / cfg["paths"]["results_dir"]
    return sb.load_sequences(results_dir)


def load_sequences_for_training(cfg):
    """Return X, y, train_mask, val_mask, test_mask for training modes."""
    X, y, ts, train_mask, val_mask, test_mask = _load_sequences(cfg)
    return X, y, train_mask, val_mask, test_mask


def run_training(cfg, quick: bool = False):
    print("\n=== Phase 7: Training ===")
    from src.config_loader import get_project_root
    X, y, ts, train_mask, val_mask, test_mask = _load_sequences(cfg)

    train_cfg = cfg
    if quick:
        print("[QUICK MODE] Using a tiny architecture and small subset for fast verification.")
        train_cfg = copy.deepcopy(cfg)
        train_cfg["model"]["lstm_units_layer1"] = 8
        train_cfg["model"]["lstm_units_layer2"] = 4
        train_cfg["model"]["attention_units"] = 4
        train_cfg["model"]["batch_size"] = 4
        train_cfg["training"]["early_stopping_patience"] = 5
        train_cfg["training"]["lr_reduction_patience"] = 3

        import numpy as np
        n_train = min(500, train_mask.sum())
        n_val = min(100, val_mask.sum())
        sub_train = np.zeros(len(train_mask), dtype=bool)
        sub_train[np.where(train_mask)[0][:n_train]] = True
        sub_val = np.zeros(len(val_mask), dtype=bool)
        sub_val[np.where(val_mask)[0][:n_val]] = True
        train_mask = sub_train
        val_mask = sub_val

    history, results = training.train_model(train_cfg, X, y, train_mask, val_mask)
    print(f"Training complete. Best epoch: {results['best_epoch']}, total epochs: {results['total_epochs_run']}")


def run_evaluation(cfg):
    print("\n=== Phase 8: Forecasting Evaluation ===")
    ev.run_evaluation(cfg)
    print("Evaluation artifacts exported.")


def run_anomaly(cfg):
    print("\n=== Phase 9: Anomaly Detection ===")
    import numpy as np
    from src.config_loader import get_project_root
    from src.target_scaling import load_target_scaler
    from src.pytorch_model import load_model_with_config
    from src.evaluation import _predict_in_batches
    results_dir = get_project_root() / cfg["paths"]["results_dir"]
    X, y, ts, train_mask, val_mask, test_mask = sb.load_sequences(results_dir)
    model_dir = get_project_root() / cfg["paths"]["model_dir"]
    model, _ = load_model_with_config(
        model_dir, (X.shape[1], X.shape[2]), fallback_cfg=cfg
    )
    y_pred = _predict_in_batches(model, X[test_mask], cfg["model"]["batch_size"])
    target_scaler = load_target_scaler(get_project_root() / cfg["paths"]["model_dir"])
    y_test_orig = target_scaler.inverse_transform(y[test_mask])
    y_pred_orig = target_scaler.inverse_transform(y_pred)
    # PM2.5 is target column 0, PM10 is target column 1
    for idx, pollutant in [(0, "pm2_5"), (1, "pm10")]:
        print(f"Running anomaly detection for {pollutant}...")
        ad.run_anomaly_pipeline(
            cfg,
            y_test_orig[:, idx],
            y_pred_orig[:, idx],
            timestamps=ts[test_mask],
            out_dir=results_dir,
            pollutant=pollutant,
        )
    print("Anomaly detection artifacts exported.")


def run_risk(cfg):
    print("\n=== Phase 10: Risk Engine ===")
    re.run_risk_pipeline(cfg)
    rs.run_risk_sensitivity(cfg)
    print("Risk engine artifacts exported.")


def run_interpretability(cfg):
    print("\n=== Phase 11b: Interpretability ===")
    interp.run_interpretability(cfg)
    print("Interpretability artifacts exported.")


def run_validity(cfg):
    print("\n=== Phase 11: Clinical Validity ===")
    validity.run_validity_analysis(cfg)
    print("Clinical validity artifacts exported.")


def run_hpo(cfg, n_trials: int = None, max_epochs: int = 20):
    print("\n=== Hyperparameter Optimization (Optuna) ===")
    from src.config_loader import get_project_root
    results_dir = get_project_root() / cfg["paths"]["results_dir"]
    X, y, ts, train_mask, val_mask, test_mask = sb.load_sequences(results_dir)
    best = hpo.run_hpo(cfg, X, y, train_mask, val_mask, n_trials=n_trials, max_epochs=max_epochs)
    print("Best params:", best)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ALLERT end-to-end pipeline orchestrator")
    p.add_argument("--mode", choices=["train", "evaluate", "full", "hpo", "bundle"], required=True,
                   help="Pipeline mode: train, evaluate, full, hpo, or bundle")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke-test mode for local CPU verification")
    p.add_argument("--subset-days", type=int, default=None,
                   help="Use only the most recent N days of raw data (fast CPU runs)")
    p.add_argument("--stride", type=int, default=1,
                   help="Sliding-window stride for sequence generation (default=1)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override max_epochs from config")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch_size from config")
    p.add_argument("--hpo-trials", type=int, default=None,
                   help="Number of Optuna trials (default from config)")
    p.add_argument("--config", default=None, help="Optional override path to config.yaml")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for Python/NumPy/TensorFlow (overrides config)")
    p.add_argument("--deterministic", action="store_true",
                   help="Enable TensorFlow deterministic ops (slower but reproducible)")
    p.add_argument("--with-timing", action="store_true",
                   help="Record per-phase latencies and export table_4_13_latency.json")
    p.add_argument("--walk-forward-cv", action="store_true",
                   help="Use walk-forward cross-validation instead of single split")
    p.add_argument("--augment", action="store_true",
                   help="Enable data augmentation from config")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config)) if args.config else load_config()

    if args.batch_size is not None:
        cfg["model"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["model"]["max_epochs"] = args.epochs

    # Apply reproducibility overrides from CLI
    if "reproducibility" not in cfg:
        cfg["reproducibility"] = {}
    if args.seed is not None:
        cfg["reproducibility"]["random_seed"] = args.seed
        cfg["project"]["random_seed"] = args.seed
    if args.deterministic:
        cfg["reproducibility"]["deterministic_mode"] = True

    if args.augment:
        cfg["training"]["data_augmentation"]["enabled"] = True

    timing_store = {} if args.with_timing else None

    if args.mode == "train":
        with timing.timed_section("preprocessing", timing_store):
            df = run_preprocessing(cfg, subset_days=args.subset_days)
        with timing.timed_section("feature_engineering", timing_store):
            run_feature_engineering(cfg, df)
        with timing.timed_section("scaling", timing_store):
            run_scaling(cfg)
        with timing.timed_section("sequence_builder", timing_store):
            run_sequence_builder(cfg, stride=args.stride)
        with timing.timed_section("training", timing_store):
            if args.walk_forward_cv:
                from src.training import run_walk_forward_training
                run_walk_forward_training(cfg, *load_sequences_for_training(cfg))
            else:
                run_training(cfg, quick=args.quick)

    elif args.mode == "evaluate":
        with timing.timed_section("evaluation", timing_store):
            run_evaluation(cfg)
        with timing.timed_section("anomaly_detection", timing_store):
            run_anomaly(cfg)
        with timing.timed_section("risk_engine", timing_store):
            run_risk(cfg)
        with timing.timed_section("validity", timing_store):
            run_validity(cfg)
        with timing.timed_section("interpretability", timing_store):
            run_interpretability(cfg)

    elif args.mode == "full":
        with timing.timed_section("preprocessing", timing_store):
            df = run_preprocessing(cfg, subset_days=args.subset_days)
        with timing.timed_section("feature_engineering", timing_store):
            run_feature_engineering(cfg, df)
        with timing.timed_section("scaling", timing_store):
            run_scaling(cfg)
        with timing.timed_section("sequence_builder", timing_store):
            run_sequence_builder(cfg, stride=args.stride)
        with timing.timed_section("training", timing_store):
            if args.walk_forward_cv:
                from src.training import run_walk_forward_training
                run_walk_forward_training(cfg, *load_sequences_for_training(cfg))
            else:
                run_training(cfg, quick=args.quick)
        with timing.timed_section("evaluation", timing_store):
            run_evaluation(cfg)
        with timing.timed_section("anomaly_detection", timing_store):
            run_anomaly(cfg)
        with timing.timed_section("risk_engine", timing_store):
            run_risk(cfg)
        with timing.timed_section("validity", timing_store):
            run_validity(cfg)
        with timing.timed_section("interpretability", timing_store):
            run_interpretability(cfg)

    elif args.mode == "hpo":
        with timing.timed_section("hpo", timing_store):
            run_hpo(cfg, n_trials=args.hpo_trials, max_epochs=args.epochs)

    elif args.mode == "bundle":
        from src import thesis_exporter
        from src.config_loader import get_project_root
        thesis_exporter.export_thesis_bundle(
            get_project_root(),
            get_project_root() / cfg["paths"]["results_dir"] / "thesis_bundle.zip",
        )
        print("Thesis bundle exported.")

    if args.with_timing and timing_store:
        from src.config_loader import get_project_root
        tables_dir = get_project_root() / cfg["paths"]["results_dir"] / "tables"
        timing.export_latency_report(
            timing_store,
            tables_dir / "table_4_13_latency.json",
            extra={"mode": args.mode, "seed": args.seed},
        )
        timing.plot_latency_breakdown(
            timing_store,
            get_project_root() / cfg["paths"]["results_dir"] / "visualizations" / "fig_4_13_latency_breakdown.png",
        )
        print("Latency report exported.")

    print("\n=== Orchestrator finished ===")


if __name__ == "__main__":
    main()
