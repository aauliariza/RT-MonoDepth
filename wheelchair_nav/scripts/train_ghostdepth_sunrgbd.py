"""Trains Ghost-Depth (networks/GhostDepth/ghost_depth.py, reimplemented
from Quan et al., CNIOT '23, doi:10.1145/3603781.3603861) from scratch on
the exact same SUN RGB-D train/val/test split as RT-MonoDepth and
FastDepth (splits_sunrgbd/, see scripts/prepare_sunrgbd.py).

The paper's own training recipe is followed where it is stated, so this
is as close to Ghost-Depth as the SUN RGB-D setting allows:

  - berHu (reverse Huber) loss, the paper's Eq. 3 (sec. 3.2), with the
    per-image threshold c = 0.2 * max|x| -- see berhu_loss() below. This
    is the loss Ghost-Depth uses, NOT the masked-L1 + scale-invariant
    objective the other depth models in this repo train with, so
    --loss is provided to switch if you want to ablate that difference.
  - Adam with beta1=0.9, beta2=0.999, weight_decay=1e-4, initial LR 1e-4
    decayed to 10% every 30 epochs, batch size 8 (paper sec. 4.2).
    Those are the defaults below.

Two deliberate departures from the paper, both to keep the comparison in
evaluation/eval_depth_comparison.py apples-to-apples:

  - Trained FROM SCRATCH. The paper initialises the GhostNet encoder from
    ImageNet; here no pretrained checkpoint is loaded, matching how
    RT-MonoDepth and FastDepth are trained in this repo. The comparison
    therefore isolates architecture rather than pretraining, and the
    accuracy is NOT expected to match the paper's published numbers.
  - SUN RGB-D at this repo's resolution and metric range, not NYU-Depth
    V2 at 304x228.

Ghost-Depth predicts depth at HALF the fed resolution (the paper trains
304x228 inputs against 152x114 targets). The loss is computed at that
native half resolution by downsampling the ground truth to meet it --
nearest-neighbour for depth and mask so no invalid pixel leaks into a
neighbour -- rather than upsampling the prediction.

Usage:
    python -m wheelchair_nav.scripts.train_ghostdepth_sunrgbd \
        --splits_dir ./wheelchair_nav/splits_sunrgbd --num_epochs 55 --batch_size 8

Hyperparameters found by scripts/tune_ghostdepth_sunrgbd.py (Optuna, TPE
sampler) can be applied directly with --hparams_json:

    python -m wheelchair_nav.scripts.train_ghostdepth_sunrgbd \
        --splits_dir ./wheelchair_nav/splits_sunrgbd --num_epochs 55 \
        --hparams_json ./wheelchair_nav/log_ghostdepth/optuna_best_ghostdepth_hparams.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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
from wheelchair_nav.training_log import save_training_curve  # noqa: E402


def berhu_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, c_frac: float = 0.2) -> torch.Tensor:
    """berHu (reverse Huber) loss -- Ghost-Depth Eq. 3, as introduced by
    Laina et al. (3DV 2016):

        B(x) = |x|                  if |x| <= c
             = (x^2 + c^2) / (2c)   if |x| >  c

    with x the prediction error and c = c_frac * max_i(|x|). The paper
    takes that maximum over "any pixels in each image" of the batch, so
    the threshold is computed PER IMAGE here, over valid pixels only, and
    detached -- c indexes which regime each pixel falls in, it is not
    itself a quantity to backpropagate through.

    Behaves as L1 on small errors and as L2 on large ones, so pixels with
    large errors get proportionally more weight.
    """
    diff = (pred - gt) * mask
    abs_diff = diff.abs()

    # Per-image max over valid pixels -> (N, 1, 1, 1), broadcast back out.
    per_image_max = abs_diff.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
    c = (c_frac * per_image_max).detach().clamp_min(1e-6)

    l2_branch = (abs_diff ** 2 + c ** 2) / (2.0 * c)
    loss = torch.where(abs_diff <= c, abs_diff, l2_branch) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def compute_loss(depth_pred: torch.Tensor, raw: torch.Tensor, color: torch.Tensor,
                 depth_gt: torch.Tensor, valid: torch.Tensor, args) -> torch.Tensor:
    """Ghost-Depth's own objective (berHu) by default; the repo's shared
    masked-L1 + scale-invariant-log + smoothness objective when
    --loss l1_silog, so the architecture and the training objective can
    be ablated independently.
    """
    if args.loss == "berhu":
        return berhu_loss(depth_pred, depth_gt, valid, c_frac=args.berhu_c_frac)

    return (
        masked_l1(depth_pred, depth_gt, valid)
        + scale_invariant_log_loss(depth_pred, depth_gt, valid, lam=args.si_lambda)
        + args.smoothness_weight * get_smooth_loss(raw, color)
    )


def save_model(save_dir: str, model: GhostDepth, args) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "height": args.height,
            "width": args.width,
            "width_mult": args.width_mult,
            "iaff_channels": args.iaff_channels,
            "keep_final_conv": not args.no_final_conv,
        },
        os.path.join(save_dir, "ghostdepth.pth"),
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_dir", default="./wheelchair_nav/splits_sunrgbd")
    p.add_argument("--log_dir", default="./wheelchair_nav/log_ghostdepth")
    p.add_argument("--model_name", default="GhostDepth_sunrgbd")
    p.add_argument("--height", type=int, default=INPUT_HEIGHT)
    p.add_argument("--width", type=int, default=INPUT_WIDTH)
    p.add_argument("--min_depth", type=float, default=MIN_DEPTH_M)
    p.add_argument("--max_depth", type=float, default=MAX_DEPTH_M)
    # Paper sec. 4.2: batch 8, Adam(0.9, 0.999), wd 1e-4, lr 1e-4 /10 every 30, 55 epochs.
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=55)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler_step_size", type=int, default=30)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--loss", choices=["berhu", "l1_silog"], default="berhu",
                    help="'berhu' is Ghost-Depth's own loss (paper Eq. 3); 'l1_silog' is the "
                         "objective RT-MonoDepth/FastDepth train with in this repo")
    p.add_argument("--berhu_c_frac", type=float, default=0.2,
                    help="c = berhu_c_frac * max|error| per image (paper: 0.2)")
    p.add_argument("--smoothness_weight", type=float, default=1e-3, help="--loss l1_silog only")
    p.add_argument("--si_lambda", type=float, default=0.5, help="--loss l1_silog only")
    # Architecture switches -- see the reproduction notes in ghost_depth.py.
    p.add_argument("--width_mult", type=float, default=1.0, help="GhostNet width multiplier")
    p.add_argument("--iaff_channels", type=int, default=40,
                    help="Skip-fusion width that gets iAFF instead of addition (paper: 40)")
    p.add_argument("--no_final_conv", action="store_true",
                    help="Drop GhostNet's 160->960 ConvBnAct as well as the classifier "
                         "(2.57M params instead of 2.79M; the paper reports 2.71M)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    p.add_argument("--hparams_json", default=None,
                    help="JSON from tune_ghostdepth_sunrgbd.py ({'best_params': {...}}); overrides "
                         "learning_rate/batch_size/weight_decay/berhu_c_frac")
    args = p.parse_args()

    if args.hparams_json:
        with open(args.hparams_json, "r") as f:
            best_params = json.load(f).get("best_params", {})
        for key in ("learning_rate", "batch_size", "weight_decay", "berhu_c_frac",
                    "smoothness_weight", "si_lambda"):
            if key in best_params:
                setattr(args, key, best_params[key])
        print(f"Loaded tuned hyperparameters from {args.hparams_json}: {best_params}")

    return args


def main():
    args = parse_args()

    # iAFF's global attention branch is BatchNorm applied after a global
    # average pool, so its input is 1x1 spatially and BatchNorm has only
    # the batch axis left to estimate variance over. PyTorch raises a
    # confusing "Expected more than 1 value per channel" deep inside the
    # model for batch_size=1; catch it here with an actionable message.
    if args.batch_size < 2:
        raise SystemExit(
            f"--batch_size must be >= 2 (got {args.batch_size}): Ghost-Depth's iAFF module "
            "applies BatchNorm to globally-pooled features, which cannot compute a variance "
            "from a single sample. Use --batch_size 2 or higher."
        )

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
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

    # Random weight initialization only -- no ImageNet-pretrained GhostNet
    # encoder, unlike the paper, so this matches the "from scratch"
    # convention the other depth models in this repo are trained under.
    model = GhostDepth(
        width=args.width_mult,
        iaff_channels=args.iaff_channels,
        keep_final_conv=not args.no_final_conv,
    ).to(device)
    print(f"Ghost-Depth parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M "
          f"(paper reports 2.71 M)")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate,
        betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_step_size, 0.1)

    model_dir = os.path.join(args.log_dir, args.model_name)
    save_root = os.path.join(model_dir, "models")
    best_val = float("inf")
    history_epochs, history_train, history_val = [], [], []

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

            # Ghost-Depth predicts at half resolution: bring the targets
            # down to it rather than upsampling the prediction.
            size = raw.shape[-2:]
            depth_gt_s = F.interpolate(depth_gt, size=size, mode="nearest")
            valid_s = F.interpolate(valid, size=size, mode="nearest")
            color_s = F.interpolate(color, size=size, mode="bilinear", align_corners=False)

            loss = compute_loss(depth_pred, raw, color_s, depth_gt_s, valid_s, args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))

        # Validation reports masked L1 in meters at FULL resolution, the
        # same quantity train_depth_sunrgbd.py / train_fastdepth_sunrgbd.py
        # report, so "best" means the same thing across all three models.
        model.eval()
        val_loss = 0.0
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
        val_loss /= max(1, len(val_loader))

        dt = time.time() - t0
        print(f"epoch {epoch:03d} | train_loss {train_loss:.4f} | val_L1 {val_loss:.4f} m | {dt:.1f}s")

        save_model(os.path.join(save_root, f"weights_{epoch}"), model, args)
        if val_loss < best_val:
            best_val = val_loss
            save_model(os.path.join(save_root, "best"), model, args)

        history_epochs.append(epoch)
        history_train.append(train_loss)
        history_val.append(val_loss)
        save_training_curve(
            model_dir, history_epochs, history_train, history_val,
            train_label=f"train_loss ({args.loss})", val_label="val_L1 (m)",
            title=f"{args.model_name} training curve",
        )

    print(f"Done. Best val L1: {best_val:.4f} m. Weights saved under {save_root}")
    print(f"Training curve: {os.path.join(model_dir, 'training_curve.png')}")


if __name__ == "__main__":
    main()
