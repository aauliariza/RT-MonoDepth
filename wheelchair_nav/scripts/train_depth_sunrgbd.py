"""Trains RT-MonoDepth -- the FULL model from networks/RTMonoDepth/RTMonoDepth.py,
architecture completely unmodified -- from scratch (random Kaiming init,
no pretrained checkpoint ever loaded) on SUN RGB-D metric depth pairs
produced by scripts/prepare_sunrgbd.py.

Why a different training recipe than trainer.py: the original repo trains
RT-MonoDepth self-supervised on KITTI video with a photometric
reprojection loss between consecutive frames, which needs posed monocular
(or stereo) sequences. SUN RGB-D is a collection of single RGB-D snapshots
of static indoor scenes with dense sensor-measured ground-truth depth, so
there is no ego-motion to exploit -- we instead train directly against
that ground truth with a masked L1 + scale-invariant log loss (Eigen et
al.) plus the repo's own edge-aware smoothness term. Only the training
objective is new; DepthEncoder/DepthDecoder are imported unchanged.

Usage:
    python -m wheelchair_nav.scripts.train_depth_sunrgbd \
        --splits_dir ./splits_sunrgbd --num_epochs 40 --batch_size 16

Hyperparameters found by scripts/tune_depth_sunrgbd.py (Optuna, TPE
sampler) can be applied directly with --hparams_json, which overrides
--learning_rate/--batch_size/--smoothness_weight/--si_lambda/--weight_decay
with the tuned values before training starts:

    python -m wheelchair_nav.scripts.train_depth_sunrgbd \
        --splits_dir ./splits_sunrgbd --num_epochs 40 \
        --hparams_json ./log_sunrgbd/optuna_best_depth_hparams.json
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

from layers import disp_to_depth, get_smooth_loss  # noqa: E402  (repo root, unmodified)
from networks.RTMonoDepth.RTMonoDepth import DepthDecoder, DepthEncoder  # noqa: E402  (repo root, unmodified)

from wheelchair_nav.datasets.sunrgbd_dataset import SUNRGBDDepthDataset  # noqa: E402


def masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - gt) * mask).sum() / denom


def scale_invariant_log_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, lam: float = 0.5) -> torch.Tensor:
    eps = 1e-6
    diff = (torch.log(pred.clamp_min(eps)) - torch.log(gt.clamp_min(eps))) * mask
    n = mask.sum().clamp_min(1.0)
    term1 = (diff ** 2).sum() / n
    term2 = (diff.sum() / n) ** 2
    return term1 - lam * term2


def save_models(save_dir: str, encoder: DepthEncoder, decoder: DepthDecoder, height: int, width: int) -> None:
    os.makedirs(save_dir, exist_ok=True)
    enc_state = encoder.state_dict()
    enc_state["height"] = height
    enc_state["width"] = width
    torch.save(enc_state, os.path.join(save_dir, "encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(save_dir, "depth.pth"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_dir", default="./splits_sunrgbd")
    p.add_argument("--log_dir", default="./log_sunrgbd")
    p.add_argument("--model_name", default="RTMonoDepth_sunrgbd")
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
                    help="JSON from tune_depth_sunrgbd.py ({'best_params': {...}}); overrides "
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

    # Random weight initialization only (Kaiming init happens inside the
    # classes' own __init__) -- no pretrained checkpoint is ever loaded,
    # satisfying the "trained from scratch" requirement.
    encoder = DepthEncoder().to(device)
    decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_step_size, 0.1)

    save_root = os.path.join(args.log_dir, args.model_name, "models")
    best_val = float("inf")

    for epoch in range(args.num_epochs):
        encoder.train()
        decoder.train()
        t0 = time.time()
        running_loss = 0.0

        for batch in train_loader:
            color = batch["color"].to(device)
            depth_gt = batch["depth_gt"].to(device)
            valid = batch["valid_mask"].to(device)

            outputs = decoder(encoder(color))
            disp = outputs[("disp", 0)]
            _, depth_pred = disp_to_depth(disp, args.min_depth, args.max_depth)

            loss = (
                masked_l1(depth_pred, depth_gt, valid)
                + scale_invariant_log_loss(depth_pred, depth_gt, valid, lam=args.si_lambda)
                + args.smoothness_weight * get_smooth_loss(disp, color)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))

        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                color = batch["color"].to(device)
                depth_gt = batch["depth_gt"].to(device)
                valid = batch["valid_mask"].to(device)
                outputs = decoder(encoder(color))
                _, depth_pred = disp_to_depth(outputs[("disp", 0)], args.min_depth, args.max_depth)
                val_loss += masked_l1(depth_pred, depth_gt, valid).item()
        val_loss /= max(1, len(val_loader))

        dt = time.time() - t0
        print(f"epoch {epoch:03d} | train_loss {train_loss:.4f} | val_L1 {val_loss:.4f} m | {dt:.1f}s")

        save_models(os.path.join(save_root, f"weights_{epoch}"), encoder, decoder, args.height, args.width)
        if val_loss < best_val:
            best_val = val_loss
            save_models(os.path.join(save_root, "best"), encoder, decoder, args.height, args.width)

    print(f"Done. Best val L1: {best_val:.4f} m. Weights saved under {save_root}")


if __name__ == "__main__":
    main()
