"""Hyperparameter search (Optuna, TPE sampler) for the YOLO26-nano
class-agnostic "obstacle" detector -- run this BEFORE the full training in
scripts/train_yolo_obstacle.py.

Each trial trains a fresh, randomly-initialized yolo26n.yaml (no
pretrained checkpoint -- same "from scratch" policy as
train_yolo_obstacle.py) for a small number of epochs (--epochs_per_trial)
on --data (SUN RGB-D's own 2D-annotated boxes, converted by
prepare_sunrgbd.py --make_yolo_labels; no other dataset), and is scored on
validation mAP50-95 (maximized), read the same way
evaluation/eval_detection_metrics.py reads it. Trials tune the optimizer
(lr0, lrf, momentum, weight_decay, warmup_epochs), the loss weighting
(box, cls) and the augmentation pipeline (hsv/translate/scale/fliplr/
mosaic) -- everything else (epochs, imgsz, batch, single_cls) stays fixed
so trials are comparable.

The best hyperparameters found are written to a JSON file that maps
directly onto Ultralytics' model.train() kwargs, so the full run can pick
them up with a single flag:

    python -m wheelchair_nav.scripts.train_yolo_obstacle \
        --data ./data/sunrgbd_yolo/obstacle.yaml --epochs 200 \
        --hparams_json ./log_yolo/optuna_best_yolo_hparams.json

Usage:
    python -m wheelchair_nav.scripts.tune_yolo_obstacle \
        --data ./data/sunrgbd_yolo/obstacle.yaml \
        --n_trials 30 --epochs_per_trial 10 \
        --out_json ./log_yolo/optuna_best_yolo_hparams.json
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
            "box": trial.suggest_float("box", 0.02, 0.2),
            "cls": trial.suggest_float("cls", 0.2, 4.0),
            "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.1),
            "hsv_s": trial.suggest_float("hsv_s", 0.0, 0.9),
            "hsv_v": trial.suggest_float("hsv_v", 0.0, 0.9),
            "translate": trial.suggest_float("translate", 0.0, 0.9),
            "scale": trial.suggest_float("scale", 0.0, 0.9),
            "fliplr": trial.suggest_float("fliplr", 0.0, 0.5),
            "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
        }

        model = YOLO(args.pretrained) if args.pretrained else YOLO("yolo26n.yaml")
        trial_name = f"trial_{trial.number:03d}"

        model.train(
            data=args.data,
            epochs=args.epochs_per_trial,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.tune_project,
            name=trial_name,
            single_cls=True,
            verbose=False,
            plots=False,
            **hparams,
        )

        metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False, plots=False)
        map50_95 = float(metrics.box.map)
        return map50_95

    return objective


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="./data/sunrgbd_yolo/obstacle.yaml")
    p.add_argument("--pretrained", default="",
                    help="Empty (default) starts each trial from random init (yolo26n.yaml) -- no "
                         "pretrained checkpoint is used")
    p.add_argument("--epochs_per_trial", type=int, default=10,
                    help="Short proxy fine-tuning budget per trial -- the full budget is used "
                         "later in train_yolo_obstacle.py")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--timeout", type=int, default=None, help="Optional wall-clock budget in seconds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study_name", default="yolo26n_obstacle_tpe")
    p.add_argument("--storage", default=None,
                    help="e.g. sqlite:///./log_yolo/optuna_yolo.db -- enables resuming/parallel trials")
    p.add_argument("--tune_project", default="./log_yolo/optuna_tuning",
                    help="Each trial's Ultralytics run artifacts are written under here")
    p.add_argument("--out_json", default="./log_yolo/optuna_best_yolo_hparams.json")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Tuning {args.n_trials} trials x {args.epochs_per_trial} epochs each on device={args.device}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="maximize",  # maximize validation mAP50-95
        sampler=sampler,
    )
    study.optimize(objective_factory(args), n_trials=args.n_trials, timeout=args.timeout)

    print("\nBest trial:")
    print(f"  mAP50-95: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"best_value_map50_95": study.best_value, "best_params": study.best_params}, f, indent=2)
    print(f"\nSaved best hyperparameters to {args.out_json}")

    try:
        csv_path = os.path.splitext(args.out_json)[0] + "_trials.csv"
        study.trials_dataframe().to_csv(csv_path, index=False)
        print(f"Saved full trial history to {csv_path}")
    except Exception:
        pass  # trials_dataframe() needs pandas -- optional, tuning result is already saved above


if __name__ == "__main__":
    main()
