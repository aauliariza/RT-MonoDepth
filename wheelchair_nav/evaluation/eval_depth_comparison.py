"""Apples-to-apples comparison of RT-MonoDepth (full, this project's chosen
depth model) against four baselines trained/tuned on the exact same SUN
RGB-D train/val/test split (see scripts/prepare_sunrgbd.py):

  - FastDepth              (networks/FastDepth/model.py, unmodified)
  - Ghost-Depth            (networks/GhostDepth/ghost_depth.py, reimplemented
                            from Quan et al., CNIOT '23)
  - YOLO26n-depth           (Ultralytics native depth task, unmodified)
  - YOLO26s-depth           (Ultralytics native depth task, unmodified)

All models are scored with the exact same metric definitions
(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 -- identical formulas to
evaluate_depth_full.py in the repo root and
evaluation/eval_depth_metrics.py), the same valid-pixel mask, and the same
test images, so the resulting table isolates architecture/training
differences rather than evaluation-protocol differences. Parameter count
and inference FPS are reported alongside for a full accuracy-vs-cost
picture.

Pass only the weights for the models you want in the table; any model
whose weights flag is omitted is skipped.

Usage:
    python -m wheelchair_nav.evaluation.eval_depth_comparison \
        --test_list ./wheelchair_nav/splits_sunrgbd/test.txt \
        --rtmonodepth_weights_dir ./wheelchair_nav/log_sunrgbd/RTMonoDepth_sunrgbd/models/best \
        --fastdepth_weights_dir ./wheelchair_nav/log_fastdepth/FastDepth_sunrgbd/models/best \
        --ghostdepth_weights_dir ./wheelchair_nav/log_ghostdepth/GhostDepth_sunrgbd/models/best \
        --yolo26n_depth_weights ./wheelchair_nav/log_yolo_depth/yolo26n_depth_sunrgbd/weights/best.pt \
        --yolo26s_depth_weights ./wheelchair_nav/log_yolo_depth/yolo26s_depth_sunrgbd/weights/best.pt \
        --out_csv ./wheelchair_nav/log_sunrgbd/depth_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wheelchair_nav.config import MAX_DEPTH_M, MIN_DEPTH_M  # noqa: E402


def compute_errors(gt: np.ndarray, pred: np.ndarray):
    """Identical formulas to evaluate_depth_full.py / eval_depth_metrics.py."""
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def read_test_pairs(test_list: str):
    pairs = []
    with open(test_list, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rgb, depth = line.split()
            pairs.append((rgb, depth))
    return pairs


def build_estimators(args):
    """Returns an ordered dict of {name: estimator} for every model whose
    weights were provided on the CLI. Each estimator exposes
    .infer(frame_bgr) -> (H, W) float32 metric depth and .num_parameters().
    """
    estimators = {}

    if args.rtmonodepth_weights_dir:
        from wheelchair_nav.perception.depth_estimator import DepthEstimator

        estimators["RT-MonoDepth (full)"] = DepthEstimator(
            args.rtmonodepth_weights_dir, device=args.device,
            min_depth_m=args.min_depth, max_depth_m=args.max_depth,
        )

    if args.fastdepth_weights_dir:
        from wheelchair_nav.baselines.fastdepth_estimator import FastDepthEstimator

        estimators["FastDepth"] = FastDepthEstimator(
            args.fastdepth_weights_dir, device=args.device,
            min_depth_m=args.min_depth, max_depth_m=args.max_depth,
        )

    if args.ghostdepth_weights_dir:
        from wheelchair_nav.baselines.ghostdepth_estimator import GhostDepthEstimator

        estimators["Ghost-Depth"] = GhostDepthEstimator(
            args.ghostdepth_weights_dir, device=args.device,
            min_depth_m=args.min_depth, max_depth_m=args.max_depth,
        )

    if args.yolo26n_depth_weights:
        from wheelchair_nav.baselines.yolo_depth_estimator import YoloDepthEstimator

        estimators["YOLO26n-depth"] = YoloDepthEstimator(
            args.yolo26n_depth_weights, device=args.device, imgsz=args.yolo_imgsz,
        )

    if args.yolo26s_depth_weights:
        from wheelchair_nav.baselines.yolo_depth_estimator import YoloDepthEstimator

        estimators["YOLO26s-depth"] = YoloDepthEstimator(
            args.yolo26s_depth_weights, device=args.device, imgsz=args.yolo_imgsz,
        )

    if not estimators:
        raise SystemExit(
            "No model weights provided -- pass at least one of "
            "--rtmonodepth_weights_dir / --fastdepth_weights_dir / "
            "--ghostdepth_weights_dir / --yolo26n_depth_weights / "
            "--yolo26s_depth_weights."
        )
    return estimators


def evaluate_model(name: str, estimator, pairs, args):
    errors = []
    for rgb_path, depth_path in pairs:
        frame = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        gt = np.load(depth_path).astype(np.float32)

        pred = estimator.infer(frame)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        mask = (gt > args.min_depth) & (gt < args.max_depth) & np.isfinite(pred)
        if mask.sum() == 0:
            continue

        pred_valid = np.clip(pred[mask], args.min_depth, args.max_depth)
        errors.append(compute_errors(gt[mask], pred_valid))

    if not errors:
        print(f"[{name}] no valid-depth pixels evaluated -- skipping.")
        return None

    mean_errors = np.array(errors).mean(0)
    n_params = estimator.num_parameters()

    # FPS: warmup + averaged repeated inference on the first test frame,
    # same protocol as eval_depth_metrics.py / compare_runtime.py.
    frame0 = cv2.imread(pairs[0][0], cv2.IMREAD_COLOR)
    for _ in range(args.fps_warmup):
        estimator.infer(frame0)
    t0 = time.time()
    for _ in range(args.fps_cycles):
        estimator.infer(frame0)
    dt = time.time() - t0
    fps = args.fps_cycles / dt if dt > 0 else float("inf")

    return {
        "model": name,
        "n_images": len(errors),
        "abs_rel": mean_errors[0],
        "sq_rel": mean_errors[1],
        "rmse": mean_errors[2],
        "rmse_log": mean_errors[3],
        "a1": mean_errors[4],
        "a2": mean_errors[5],
        "a3": mean_errors[6],
        "params_m": n_params / 1e6,
        "fps": fps,
    }


def print_table(rows):
    cols = ["model", "n_images", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "params_m", "fps"]
    headers = ["Model", "N", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "Params(M)", "FPS"]
    widths = [max(len(h), 12) for h in headers]
    widths[0] = max(widths[0], max(len(r["model"]) for r in rows) + 2)

    def fmt_row(values):
        return "".join(f"{str(v):>{w}}" for v, w in zip(values, widths))

    print("\nApples-to-apples depth model comparison (same test split, same metric formulas):")
    print(fmt_row(headers))
    for r in rows:
        values = [
            r["model"], r["n_images"],
            f"{r['abs_rel']:.4f}", f"{r['sq_rel']:.4f}", f"{r['rmse']:.4f}", f"{r['rmse_log']:.4f}",
            f"{r['a1']:.4f}", f"{r['a2']:.4f}", f"{r['a3']:.4f}",
            f"{r['params_m']:.3f}", f"{r['fps']:.1f}",
        ]
        print(fmt_row(values))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_list", default="./wheelchair_nav/splits_sunrgbd/test.txt")
    p.add_argument("--rtmonodepth_weights_dir", default=None)
    p.add_argument("--fastdepth_weights_dir", default=None)
    p.add_argument("--ghostdepth_weights_dir", default=None)
    p.add_argument("--yolo26n_depth_weights", default=None)
    p.add_argument("--yolo26s_depth_weights", default=None)
    p.add_argument("--min_depth", type=float, default=MIN_DEPTH_M)
    p.add_argument("--max_depth", type=float, default=MAX_DEPTH_M)
    p.add_argument("--device", default="cuda")
    p.add_argument("--yolo_imgsz", type=int, default=640)
    p.add_argument("--fps_cycles", type=int, default=100)
    p.add_argument("--fps_warmup", type=int, default=10)
    p.add_argument("--out_csv", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    pairs = read_test_pairs(args.test_list)
    if not pairs:
        raise SystemExit(f"No test pairs found in {args.test_list}")

    estimators = build_estimators(args)

    rows = []
    for name, estimator in estimators.items():
        print(f"Evaluating {name} on {len(pairs)} test images...")
        result = evaluate_model(name, estimator, pairs, args)
        if result is not None:
            rows.append(result)

    print_table(rows)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved comparison table to {args.out_csv}")


if __name__ == "__main__":
    main()
