"""Depth-estimation evaluation for the from-scratch RT-MonoDepth (full)
model trained on SUN RGB-D: accuracy/error metrics (abs_rel, sq_rel, rmse,
rmse_log, a1/a2/a3 -- the same definitions used by evaluate_depth_full.py
in the repo root for KITTI), parameter count, and inference FPS.

Usage:
    python -m wheelchair_nav.evaluation.eval_depth_metrics \
        --weights_dir ./wheelchair_nav/log_sunrgbd/RTMonoDepth_sunrgbd/models/best \
        --test_list ./wheelchair_nav/splits_sunrgbd/test.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from layers import disp_to_depth  # noqa: E402  (repo root, unmodified)

from wheelchair_nav.datasets.sunrgbd_dataset import SUNRGBDDepthDataset  # noqa: E402
from wheelchair_nav.perception.depth_estimator import DepthEstimator  # noqa: E402


def compute_errors(gt: np.ndarray, pred: np.ndarray):
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights_dir", required=True, help="Folder with encoder.pth + depth.pth")
    p.add_argument("--test_list", default="./wheelchair_nav/splits_sunrgbd/test.txt")
    p.add_argument("--min_depth", type=float, default=0.1)
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps_cycles", type=int, default=200)
    p.add_argument("--fps_warmup", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    estimator = DepthEstimator(
        args.weights_dir, device=args.device, min_depth_m=args.min_depth, max_depth_m=args.max_depth,
    )

    dataset = SUNRGBDDepthDataset(
        args.test_list, estimator.feed_height, estimator.feed_width,
        args.min_depth, args.max_depth, is_train=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    errors = []
    with torch.no_grad():
        for batch in loader:
            color = batch["color"].to(estimator.device)
            depth_gt = batch["depth_gt"].numpy()[0, 0]
            valid = batch["valid_mask"].numpy()[0, 0].astype(bool)
            if valid.sum() == 0:
                continue

            outputs = estimator.decoder(estimator.encoder(color))
            _, depth_pred = disp_to_depth(outputs[("disp", 0)], args.min_depth, args.max_depth)
            depth_pred = depth_pred.cpu().numpy()[0, 0]

            errors.append(compute_errors(depth_gt[valid], depth_pred[valid]))

    if not errors:
        raise SystemExit("No valid-depth pixels found across the test split -- check --test_list.")

    mean_errors = np.array(errors).mean(0)
    names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    print(f"\nDepth accuracy on SUN RGB-D test split ({len(errors)} images):")
    print(("  " + "{:>10}" * 7).format(*names))
    print(("  " + "{:>10.4f}" * 7).format(*mean_errors.tolist()))

    n_params = estimator.num_parameters()
    print(f"\nParameters: encoder+decoder = {n_params / 1e6:.3f} M ({n_params} total)")

    device = estimator.device
    dummy = torch.randn(1, 3, estimator.feed_height, estimator.feed_width).to(device)
    with torch.no_grad():
        for _ in range(args.fps_warmup):
            estimator.decoder(estimator.encoder(dummy))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.fps_cycles):
            estimator.decoder(estimator.encoder(dummy))
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
    fps = args.fps_cycles / dt
    print(f"Inference FPS ({device}, {estimator.feed_width}x{estimator.feed_height}): {fps:.1f}")


if __name__ == "__main__":
    main()
