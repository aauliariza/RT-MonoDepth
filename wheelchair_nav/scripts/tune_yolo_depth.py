"""Hyperparameter search (Optuna, TPE sampler) for YOLO26n-depth /
YOLO26s-depth -- run this BEFORE the full training in
scripts/train_yolo_depth.py.

Each trial fine-tunes a fresh copy of --pretrained (or a from-scratch
yolo26{variant}-depth.yaml) for a small number of epochs
(--epochs_per_trial) on --data, and is scored on validation abs_rel
(minimized) -- read from Ultralytics' native DepthValidator, the same
metrics.results_dict['metrics/abs_rel'] reported by
`yolo depth val`/model.val(). Trials tune the optimizer (lr0, lrf,
momentum, weight_decay, warmup_epochs), the depth-specific loss weights
(dlog: SILog gain, dgrad: gradient-loss gain, dlam: SILog scale-invariance
focus) and the augmentation pipeline (hsv/translate/scale/fliplr/mosaic).

The best hyperparameters found are written to a JSON file that maps
directly onto Ultralytics' model.train() kwargs, so the full run can pick
them up with a single flag:

    python -m wheelchair_nav.scripts.train_yolo_depth \
        --variant n --data ./data/sunrgbd_yolo_depth/depth_comparison.yaml --epochs 60 \
        --hparams_json ./log_yolo_depth/optuna_best_yolo26n_depth_hparams.json

Usage:
    python -m wheelchair_nav.scripts.tune_yolo_depth \
        --variant n --data ./data/sunrgbd_yolo_depth/depth_comparison.yaml \
        --n_trials 30 --epochs_per_trial 10 \
        --out_json ./log_yolo_depth/optuna_best_yolo26n_depth_hparams.json
"""
from __future__ import annotations

import argparse
import json
import os

import optuna
from ultralytics import YOLO


def objective_factory(args):
    def objective(trial: optuna.Trial) -> float:
        hparams = {
            "lr0": trial.suggest_float("lr0", 1e-5, 1e-1, log=True),
            "lrf": trial.suggest_float("lrf", 0.01, 1.0, log=True),
            "momentum": trial.suggest_float("momentum", 0.6, 0.98),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "warmup_epochs": trial.suggest_float("warmup_epochs", 0.0, 5.0),
            "dlog": trial.suggest_float("dlog", 0.2, 4.0),
            "dgrad": trial.suggest_float("dgrad", 0.0, 2.0),
            "dlam": trial.suggest_float("dlam", 0.0, 1.0),
            "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.1),
            "hsv_s": trial.suggest_float("hsv_s", 0.0, 0.9),
            "hsv_v": trial.suggest_float("hsv_v", 0.0, 0.9),
            "translate": trial.suggest_float("translate", 0.0, 0.9),
            "scale": trial.suggest_float("scale", 0.0, 0.9),
            "fliplr": trial.suggest_float("fliplr", 0.0, 0.5),
            "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
        }

        weights = args.pretrained if args.pretrained is not None else f"yolo26{args.variant}-depth.pt"
        model = YOLO(weights) if weights else YOLO(f"yolo26{args.variant}-depth.yaml")
        trial_name = f"trial_{trial.number:03d}"

        model.train(
            data=args.data,
            epochs=args.epochs_per_trial,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.tune_project,
            name=trial_name,
            verbose=False,
            plots=False,
            **hparams,
        )

        metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False, plots=False)
        abs_rel = float(metrics.results_dict["metrics/abs_rel"])
        return abs_rel

    return objective


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["n", "s"], default="n", help="YOLO26 scale: n (nano) or s (small)")
    p.add_argument("--data", default="./data/sunrgbd_yolo_depth/depth_comparison.yaml")
    p.add_argument("--pretrained", default=None,
                    help="Pretrained weights each trial starts from (default: yolo26{variant}-depth.pt); "
                         "pass '' for random init")
    p.add_argument("--epochs_per_trial", type=int, default=10,
                    help="Short proxy fine-tuning budget per trial -- the full budget is used "
                         "later in train_yolo_depth.py")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--timeout", type=int, default=None, help="Optional wall-clock budget in seconds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study_name", default=None, help="default: yolo26{variant}_depth_tpe")
    p.add_argument("--storage", default=None,
                    help="e.g. sqlite:///./log_yolo_depth/optuna_yolo_depth.db -- enables resuming/parallel trials")
    p.add_argument("--tune_project", default="./log_yolo_depth/optuna_tuning",
                    help="Each trial's Ultralytics run artifacts are written under here")
    p.add_argument("--out_json", default=None, help="default: ./log_yolo_depth/optuna_best_yolo26{variant}_depth_hparams.json")
    return p.parse_args()


def main():
    args = parse_args()
    study_name = args.study_name or f"yolo26{args.variant}_depth_tpe"
    out_json = args.out_json or f"./log_yolo_depth/optuna_best_yolo26{args.variant}_depth_hparams.json"
    print(f"Tuning {args.n_trials} trials x {args.epochs_per_trial} epochs each on device={args.device}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="minimize",  # minimize validation abs_rel
        sampler=sampler,
    )
    study.optimize(objective_factory(args), n_trials=args.n_trials, timeout=args.timeout)

    print("\nBest trial:")
    print(f"  abs_rel: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"best_value_abs_rel": study.best_value, "best_params": study.best_params}, f, indent=2)
    print(f"\nSaved best hyperparameters to {out_json}")

    try:
        csv_path = os.path.splitext(out_json)[0] + "_trials.csv"
        study.trials_dataframe().to_csv(csv_path, index=False)
        print(f"Saved full trial history to {csv_path}")
    except Exception:
        pass  # trials_dataframe() needs pandas -- optional, tuning result is already saved above


if __name__ == "__main__":
    main()
