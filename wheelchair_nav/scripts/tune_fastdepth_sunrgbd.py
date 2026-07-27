"""Hyperparameter search (Optuna, TPE sampler) for the FastDepth comparison
baseline (networks/FastDepth/model.py: MobileNetSkipAdd, unmodified) -- run
this BEFORE the full training in scripts/train_fastdepth_sunrgbd.py.

Mirrors scripts/tune_depth_sunrgbd.py exactly (same search space, same
proxy-training protocol, same objective: validation masked-L1 depth error
in metres) so the two models are tuned under identical conditions -- the
whole point of using FastDepth as an apples-to-apples baseline against
RT-MonoDepth.

Usage:
    python -m wheelchair_nav.scripts.tune_fastdepth_sunrgbd \
        --splits_dir ./splits_sunrgbd --n_trials 30 --epochs_per_trial 5 \
        --out_json ./log_fastdepth/optuna_best_fastdepth_hparams.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import optuna
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from layers import get_smooth_loss  # noqa: E402  (repo root, unmodified)
from networks.FastDepth.model import MobileNetSkipAdd  # noqa: E402  (repo root, unmodified)

from wheelchair_nav.datasets.sunrgbd_dataset import SUNRGBDDepthDataset  # noqa: E402
from wheelchair_nav.scripts.train_depth_sunrgbd import masked_l1, scale_invariant_log_loss  # noqa: E402


def build_loaders(args, batch_size: int):
    train_set = SUNRGBDDepthDataset(
        os.path.join(args.splits_dir, "train.txt"), args.height, args.width,
        args.min_depth, args.max_depth, is_train=True,
    )
    val_set = SUNRGBDDepthDataset(
        os.path.join(args.splits_dir, "val.txt"), args.height, args.width,
        args.min_depth, args.max_depth, is_train=False,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader


def objective_factory(args, device: torch.device):
    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
        smoothness_weight = trial.suggest_float("smoothness_weight", 1e-4, 1e-1, log=True)
        si_lambda = trial.suggest_float("si_lambda", 0.0, 1.0)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        train_loader, val_loader = build_loaders(args, batch_size)

        # Fresh, randomly-initialized model per trial -- same "from scratch"
        # convention as the full training run.
        model = MobileNetSkipAdd().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        val_loss = float("inf")
        for epoch in range(args.epochs_per_trial):
            model.train()
            for batch in train_loader:
                color = batch["color"].to(device)
                depth_gt = batch["depth_gt"].to(device)
                valid = batch["valid_mask"].to(device)

                raw = model(color)
                depth_pred = raw.clamp(args.min_depth, args.max_depth)

                loss = (
                    masked_l1(depth_pred, depth_gt, valid)
                    + scale_invariant_log_loss(depth_pred, depth_gt, valid, lam=si_lambda)
                    + smoothness_weight * get_smooth_loss(raw, color)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss, n_batches = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    color = batch["color"].to(device)
                    depth_gt = batch["depth_gt"].to(device)
                    valid = batch["valid_mask"].to(device)
                    depth_pred = model(color).clamp(args.min_depth, args.max_depth)
                    val_loss += masked_l1(depth_pred, depth_gt, valid).item()
                    n_batches += 1
            val_loss /= max(1, n_batches)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_loss

    return objective


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_dir", default="./splits_sunrgbd")
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--min_depth", type=float, default=0.1)
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--epochs_per_trial", type=int, default=5,
                    help="Short proxy training budget per trial -- keep small, "
                         "the full budget is used later in train_fastdepth_sunrgbd.py")
    p.add_argument("--timeout", type=int, default=None, help="Optional wall-clock budget in seconds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study_name", default="fastdepth_sunrgbd_tpe")
    p.add_argument("--storage", default=None,
                    help="e.g. sqlite:///./log_fastdepth/optuna_fastdepth.db -- enables resuming/parallel trials")
    p.add_argument("--out_json", default="./log_fastdepth/optuna_best_fastdepth_hparams.json")
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Tuning on {device} | {args.n_trials} trials x {args.epochs_per_trial} epochs each")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="minimize",  # minimize validation masked-L1 depth error (metres)
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective_factory(args, device), n_trials=args.n_trials, timeout=args.timeout)

    print("\nBest trial:")
    print(f"  val_L1: {study.best_value:.4f} m")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"best_value_val_l1_m": study.best_value, "best_params": study.best_params}, f, indent=2)
    print(f"\nSaved best hyperparameters to {args.out_json}")

    try:
        csv_path = os.path.splitext(args.out_json)[0] + "_trials.csv"
        study.trials_dataframe().to_csv(csv_path, index=False)
        print(f"Saved full trial history to {csv_path}")
    except Exception:
        pass  # trials_dataframe() needs pandas -- optional, tuning result is already saved above


if __name__ == "__main__":
    main()
