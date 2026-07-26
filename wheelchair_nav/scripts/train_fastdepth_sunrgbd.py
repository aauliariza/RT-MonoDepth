"""Trains FastDepth -- the UNMODIFIED MobileNetSkipAdd model from
networks/FastDepth/model.py -- from scratch on the exact same SUN RGB-D
train/val/test split as RT-MonoDepth (splits_sunrgbd/, see
scripts/prepare_sunrgbd.py). This is a comparison-only baseline: the
result is meant to be evaluated apples-to-apples against RT-MonoDepth in
evaluation/eval_depth_comparison.py, so it deliberately reuses the same
supervised objective, dataset, resolution and epoch budget conventions as
scripts/train_depth_sunrgbd.py -- masked L1 + scale-invariant log loss
(Eigen et al.) + the repo's own edge-aware smoothness term
(layers.get_smooth_loss, unmodified). FastDepth's own final layer is a
plain Conv+BatchNorm+ReLU (see MobileNetSkipAdd.decode_conv6), so its raw
output is already non-negative; here it is only clamped into the same
indoor metric range RT-MonoDepth is trained over, not remapped through a
sigmoid/disparity formula.

Usage:
    python -m wheelchair_nav.scripts.train_fastdepth_sunrgbd \
        --splits_dir ./splits_sunrgbd --num_epochs 40 --batch_size 16

Hyperparameters found by scripts/tune_fastdepth_sunrgbd.py (Optuna, TPE
sampler) can be applied directly with --hparams_json:

    python -m wheelchair_nav.scripts.train_fastdepth_sunrgbd \
        --splits_dir ./splits_sunrgbd --num_epochs 40 \
        --hparams_json ./log_fastdepth/optuna_best_fastdepth_hparams.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from layers import get_smooth_loss  # noqa: E402  (repo root, unmodified)
from networks.FastDepth.model import MobileNetSkipAdd  # noqa: E402  (repo root, unmodified)

from wheelchair_nav.datasets.sunrgbd_dataset import SUNRGBDDepthDataset  # noqa: E402
from wheelchair_nav.scripts.train_depth_sunrgbd import masked_l1, scale_invariant_log_loss  # noqa: E402


def save_model(save_dir: str, model: MobileNetSkipAdd, height: int, width: int) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "height": height, "width": width},
        os.path.join(save_dir, "fastdepth.pth"),
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_dir", default="./splits_sunrgbd")
    p.add_argument("--log_dir", default="./log_fastdepth")
    p.add_argument("--model_name", default="FastDepth_sunrgbd")
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--min_depth", type=float, default=0.1)
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=40)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--scheduler_step_size", type=int, default=25)
    p.add_argument("--smoothness_weight", type=float, default=1e-3)
    p.add_argument("--si_lambda", type=float, default=0.5, help="scale-invariant log loss weight (Eigen et al.)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--hparams_json", default=None,
                    help="JSON from tune_fastdepth_sunrgbd.py ({'best_params': {...}}); overrides "
                         "learning_rate/batch_size/smoothness_weight/si_lambda/weight_decay")
    args = p.parse_args()

    if args.hparams_json:
        with open(args.hparams_json, "r") as f:
            best_params = json.load(f).get("best_params", {})
        for key in ("learning_rate", "batch_size", "smoothness_weight", "si_lambda", "weight_decay"):
            if key in best_params:
                setattr(args, key, best_params[key])
        print(f"Loaded tuned hyperparameters from {args.hparams_json}: {best_params}")

    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"Training on {device}")

    train_set = SUNRGBDDepthDataset(
        os.path.join(args.splits_dir, "train.txt"), args.height, args.width,
        args.min_depth, args.max_depth, is_train=True,
    )
    val_set = SUNRGBDDepthDataset(
        os.path.join(args.splits_dir, "val.txt"), args.height, args.width,
        args.min_depth, args.max_depth, is_train=False,
    )
    print(f"train: {len(train_set)} pairs | val: {len(val_set)} pairs")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # Random weight initialization only -- no pretrained checkpoint is ever
    # loaded, matching the "from scratch" training convention used for
    # RT-MonoDepth, so the comparison isolates architecture, not pretraining.
    model = MobileNetSkipAdd().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_step_size, 0.1)

    save_root = os.path.join(args.log_dir, args.model_name, "models")
    best_val = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for batch in train_loader:
            color = batch["color"].to(device)
            depth_gt = batch["depth_gt"].to(device)
            valid = batch["valid_mask"].to(device)

            raw = model(color)
            depth_pred = raw.clamp(args.min_depth, args.max_depth)

            loss = (
                masked_l1(depth_pred, depth_gt, valid)
                + scale_invariant_log_loss(depth_pred, depth_gt, valid, lam=args.si_lambda)
                + args.smoothness_weight * get_smooth_loss(raw, color)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                color = batch["color"].to(device)
                depth_gt = batch["depth_gt"].to(device)
                valid = batch["valid_mask"].to(device)
                depth_pred = model(color).clamp(args.min_depth, args.max_depth)
                val_loss += masked_l1(depth_pred, depth_gt, valid).item()
        val_loss /= max(1, len(val_loader))

        dt = time.time() - t0
        print(f"epoch {epoch:03d} | train_loss {train_loss:.4f} | val_L1 {val_loss:.4f} m | {dt:.1f}s")

        save_model(os.path.join(save_root, f"weights_{epoch}"), model, args.height, args.width)
        if val_loss < best_val:
            best_val = val_loss
            save_model(os.path.join(save_root, "best"), model, args.height, args.width)

    print(f"Done. Best val L1: {best_val:.4f} m. Weights saved under {save_root}")


if __name__ == "__main__":
    main()
