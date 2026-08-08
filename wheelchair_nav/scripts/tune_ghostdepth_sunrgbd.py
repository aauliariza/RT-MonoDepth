"""Hyperparameter search (Optuna, TPE sampler) for the Ghost-Depth
comparison baseline (networks/GhostDepth/ghost_depth.py, reimplemented
from Quan et al., CNIOT '23) -- run this BEFORE the full training in
scripts/train_ghostdepth_sunrgbd.py.

Mirrors scripts/tune_depth_sunrgbd.py and tune_fastdepth_sunrgbd.py:
same proxy-training protocol, same pruner, and the same objective
(validation masked-L1 depth error in metres at full resolution), so all
three depth models are tuned under identical conditions -- the whole
point of the apples-to-apples comparison in
evaluation/eval_depth_comparison.py.

The search space differs in one respect: Ghost-Depth trains with its own
berHu loss (paper Eq. 3), so the tuned knob is berhu_c_frac -- the
fraction of the per-image max error at which berHu switches from L1 to
L2 behaviour -- instead of smoothness_weight/si_lambda. Pass
--loss l1_silog to tune under this repo's shared objective instead.

Usage:
    python -m wheelchair_nav.scripts.tune_ghostdepth_sunrgbd \
        --splits_dir ./wheelchair_nav/splits_sunrgbd --n_trials 30 --epochs_per_trial 5 \
        --out_json ./wheelchair_nav/log_ghostdepth/optuna_best_ghostdepth_hparams.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from layers import get_smooth_loss  # noqa: E402  (repo root, unmodified)
from networks.GhostDepth.ghost_depth import GhostDepth  # noqa: E402

from wheelchair_nav.config import (  # noqa: E402
    INPUT_HEIGHT, INPUT_WIDTH, MAX_DEPTH_M, MIN_DEPTH_M,
)
from wheelchair_nav.datasets.sunrgbd_dataset import SUNRGBDDepthDataset  # noqa: E402
from wheelchair_nav.scripts.train_depth_sunrgbd import masked_l1, scale_invariant_log_loss  # noqa: E402
from wheelchair_nav.scripts.train_ghostdepth_sunrgbd import berhu_loss  # noqa: E402


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
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        if args.loss == "berhu":
            berhu_c_frac = trial.suggest_float("berhu_c_frac", 0.05, 0.5)
            smoothness_weight = si_lambda = 0.0
        else:
            berhu_c_frac = 0.2
            smoothness_weight = trial.suggest_float("smoothness_weight", 1e-4, 1e-1, log=True)
            si_lambda = trial.suggest_float("si_lambda", 0.0, 1.0)

        train_loader, val_loader = build_loaders(args, batch_size)

        # Fresh, randomly-initialized model per trial -- same "from scratch"
        # convention as the full training run.
        # Pre-bound so the finally block's del cannot raise NameError and
        # mask an OOM raised by the model construction itself.
        model = optimizer = None
        try:
            model = GhostDepth(
                width=args.width_mult,
                iaff_channels=args.iaff_channels,
                keep_final_conv=not args.no_final_conv,
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate,
                betas=(args.beta1, args.beta2), weight_decay=weight_decay,
            )

            val_loss = float("inf")
            for epoch in range(args.epochs_per_trial):
                model.train()
                for batch in train_loader:
                    color = batch["color"].to(device)
                    depth_gt = batch["depth_gt"].to(device)
                    valid = batch["valid_mask"].to(device)

                    raw = model(color)
                    depth_pred = raw.clamp(args.min_depth, args.max_depth)

                    # Ghost-Depth predicts at half resolution -- bring targets down to it.
                    size = raw.shape[-2:]
                    depth_gt_s = F.interpolate(depth_gt, size=size, mode="nearest")
                    valid_s = F.interpolate(valid, size=size, mode="nearest")

                    if args.loss == "berhu":
                        loss = berhu_loss(depth_pred, depth_gt_s, valid_s, c_frac=berhu_c_frac)
                    else:
                        color_s = F.interpolate(color, size=size, mode="bilinear", align_corners=False)
                        loss = (
                            masked_l1(depth_pred, depth_gt_s, valid_s)
                            + scale_invariant_log_loss(depth_pred, depth_gt_s, valid_s, lam=si_lambda)
                            + smoothness_weight * get_smooth_loss(raw, color_s)
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
                        raw = model(color)
                        depth_pred = F.interpolate(
                            raw, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False,
                        ).clamp(args.min_depth, args.max_depth)
                        val_loss += masked_l1(depth_pred, depth_gt, valid).item()
                        n_batches += 1
                val_loss /= max(1, n_batches)

                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return val_loss
        except torch.OutOfMemoryError:
            # The batch_size search space deliberately probes sizes that may
            # not fit this GPU. Prune that trial instead of aborting the whole
            # study, so the search simply learns to avoid those sizes.
            print(f"  trial {trial.number}: CUDA OOM at batch_size={batch_size}, pruning")
            raise optuna.TrialPruned()
        finally:
            # Optuna keeps a reference to whatever the objective leaves
            # alive, so N trials would otherwise stack N models on the GPU.
            # Dropping them and emptying the cache here also stops the CUDA
            # allocator fragmenting across trials of differing batch size --
            # the exact failure mode the OOM message warns about. The finally
            # block matters because TrialPruned exits by exception.
            del model, optimizer, train_loader, val_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return objective


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_dir", default="./wheelchair_nav/splits_sunrgbd")
    p.add_argument("--height", type=int, default=INPUT_HEIGHT)
    p.add_argument("--width", type=int, default=INPUT_WIDTH)
    p.add_argument("--min_depth", type=float, default=MIN_DEPTH_M)
    p.add_argument("--max_depth", type=float, default=MAX_DEPTH_M)
    p.add_argument("--loss", choices=["berhu", "l1_silog"], default="berhu",
                    help="Objective to tune under; must match what train_ghostdepth_sunrgbd.py runs")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--width_mult", type=float, default=1.0)
    p.add_argument("--iaff_channels", type=int, default=40)
    p.add_argument("--no_final_conv", action="store_true")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--epochs_per_trial", type=int, default=5,
                    help="Short proxy training budget per trial -- keep small, "
                         "the full budget is used later in train_ghostdepth_sunrgbd.py")
    p.add_argument("--timeout", type=int, default=None, help="Optional wall-clock budget in seconds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study_name", default="ghostdepth_sunrgbd_tpe")
    p.add_argument("--storage", default=None,
                    help="e.g. sqlite:///./wheelchair_nav/log_ghostdepth/optuna_ghostdepth.db "
                         "-- enables resuming/parallel trials")
    p.add_argument("--out_json", default="./wheelchair_nav/log_ghostdepth/optuna_best_ghostdepth_hparams.json")
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
