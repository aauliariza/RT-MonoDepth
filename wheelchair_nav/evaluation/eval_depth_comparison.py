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
differences rather than evaluation-protocol differences. Parameter count,
MACs and the full per-inference LATENCY DISTRIBUTION (mean/std/min/p50/
p90/p95/p99/max, plus FPS derived from the mean) are reported alongside
for a complete accuracy-vs-cost picture -- see evaluation/latency.py for
why the tail percentiles, not the mean, are what bound a wheelchair's
worst-case reaction distance.

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

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wheelchair_nav.config import (  # noqa: E402
    INPUT_HEIGHT, INPUT_WIDTH, MAX_DEPTH_M, MIN_DEPTH_M, YOLO_DEPTH_IMGSZ,
)
from wheelchair_nav.evaluation.latency import format_latency_table, measure_latency  # noqa: E402


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


def feed_resolution(estimator):
    """(height, width) the estimator actually runs its network at.

    All depth models in this project are trained and run on the same
    IMAGE CONTENT resolution (config.INPUT_HEIGHT x INPUT_WIDTH), but the
    Ultralytics models reach it differently: they letterbox into a square
    imgsz x imgsz canvas, so their tensor is larger than the content it
    carries (at imgsz=384 a 640x480 frame occupies exactly 288x384 with the
    rest grey padding). What this function reports is the TENSOR shape,
    which is what MACs and FPS actually scale with -- hence the padding
    overhead shows up as cost, correctly, and the resolution travels with
    those numbers instead of being silently assumed equal.
    """
    h = getattr(estimator, "feed_height", None) or getattr(estimator, "imgsz", YOLO_DEPTH_IMGSZ)
    w = getattr(estimator, "feed_width", None) or getattr(estimator, "imgsz", YOLO_DEPTH_IMGSZ)
    return int(h), int(w)


def count_macs_g(estimator, hw=None):
    """Multiply-accumulate operations per forward pass, in billions.

    Measured at `hw` = (height, width) if given, else at the estimator's
    own feed resolution. Pass an explicit `hw` to get architecture-
    comparable numbers across models whose deployed resolutions differ.

    Reported alongside parameter count because the two measure different
    costs and can disagree sharply: parameters are memory, MACs are
    arithmetic, and a network can be modest in one and expensive in the
    other. Neither predicts latency on its own -- depthwise/Ghost
    convolutions are memory-bound, so a model can hold the lowest MACs in
    the table and still be the slowest, which is why latency is measured
    directly rather than inferred from MACs.

    Returns None if thop is missing or the estimator's backing module
    can't be resolved -- the column is then left blank rather than
    failing the whole evaluation.
    """
    import torch
    import torch.nn as nn

    try:
        from thop import profile
    except ImportError:
        return None

    class Wrapped(nn.Module):
        """Presents whatever the estimator holds as one profileable module."""

        def __init__(self, estimator):
            super().__init__()
            if hasattr(estimator, "encoder") and hasattr(estimator, "decoder"):
                self.encoder = estimator.encoder          # RT-MonoDepth
                self.decoder = estimator.decoder
            elif hasattr(estimator.model, "model"):
                self.net = estimator.model.model          # YOLO26{n,s}-depth
            else:
                self.net = estimator.model                # FastDepth, Ghost-Depth

        def forward(self, x):
            if hasattr(self, "encoder"):
                return self.decoder(self.encoder(x))
            return self.net(x)

    try:
        wrapped = Wrapped(estimator)
        # thop.profile() saves model.training on entry and RESTORES it on
        # exit. A freshly constructed wrapper defaults to training=True, so
        # without this eval() the restore would flip the estimator's real
        # module into training mode -- corrupting the FPS measurement and
        # every later inference (BatchNorm would update running stats, and
        # Ghost-Depth's iAFF would crash outright on batch size 1).
        wrapped.eval()

        h, w = hw if hw is not None else feed_resolution(estimator)
        device = next(wrapped.parameters()).device
        x = torch.randn(1, 3, int(h), int(w), device=device)
        with torch.no_grad():
            macs, _ = profile(wrapped, inputs=(x,), verbose=False)
    except Exception:
        return None
    return macs / 1e9


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
    feed_h, feed_w = feed_resolution(estimator)
    macs_g = count_macs_g(estimator)
    # Same architecture, one common resolution -> MACs that can be compared
    # across models whose deployed resolutions differ.
    macs_ref = count_macs_g(estimator, hw=args.macs_ref_hw) if args.macs_ref_hw else None

    # Latency + FPS: warmup, then per-call timing of the full .infer() path
    # (resize -> H2D -> forward -> D2H) on the first test frame, identically
    # for every model. FPS is derived from the mean latency, so the two can
    # never disagree. See evaluation/latency.py for why the tail matters
    # more than the mean here.
    frame0 = cv2.imread(pairs[0][0], cv2.IMREAD_COLOR)
    lat = measure_latency(
        lambda: estimator.infer(frame0),
        warmup=args.fps_warmup,
        cycles=args.fps_cycles,
        device=getattr(estimator, "device", None),
    )

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
        "macs_g": macs_g,
        "macs_g_ref": macs_ref,
        "feed_hw": f"{feed_h}x{feed_w}",
        **lat,  # lat_mean_ms ... lat_max_ms, plus fps derived from the mean
    }


def print_table(rows):
    has_ref = any(r.get("macs_g_ref") is not None for r in rows)
    headers = ["Model", "N", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3",
               "Params(M)", "GMACs", "FeedHxW", "FPS", "p95(ms)"]
    if has_ref:
        headers.insert(11, "GMACs@ref")
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
            f"{r['params_m']:.3f}",
            "n/a" if r["macs_g"] is None else f"{r['macs_g']:.3f}",
        ]
        if has_ref:
            values.append("n/a" if r.get("macs_g_ref") is None else f"{r['macs_g_ref']:.3f}")
        values += [r["feed_hw"], f"{r['fps']:.1f}", f"{r['lat_p95_ms']:.2f}"]
        print(fmt_row(values))

    # Second table: the full latency distribution. Kept separate rather than
    # bolted onto the one above, which is already at the width a terminal can
    # show. p95 appears in both because it is the number that decides whether
    # a model is fast ENOUGH, while the mean only says how fast it usually is.
    print()
    print(format_latency_table(
        rows,
        title="Per-inference latency (full .infer() path: resize -> H2D -> forward -> D2H):",
    ))


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
    p.add_argument("--yolo_imgsz", type=int, default=YOLO_DEPTH_IMGSZ)
    # Old --fps_* spellings kept as aliases so existing commands/scripts do
    # not break; both now drive the latency sampler, since FPS is derived
    # from mean latency rather than measured separately.
    p.add_argument("--latency_cycles", "--fps_cycles", dest="fps_cycles", type=int, default=200,
                    help="Timed inferences per model. p99 is interpolated from the top ~1%%, so "
                         "raise this (500+) if you intend to quote p99 in a paper.")
    p.add_argument("--latency_warmup", "--fps_warmup", dest="fps_warmup", type=int, default=20,
                    help="Untimed calls first, to absorb CUDA context creation and cuDNN autotuning")
    p.add_argument("--macs_ref_hw", default=f"{INPUT_HEIGHT}x{INPUT_WIDTH}",
                    help="HxW at which to additionally measure every model's MACs, so the "
                         "numbers are comparable across models whose TENSOR shapes differ "
                         "(the Ultralytics models letterbox onto a padded square canvas). "
                         "Defaults to config.INPUT_HEIGHT x INPUT_WIDTH. Set to 'none' to "
                         "report only as-deployed MACs.")
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    if args.macs_ref_hw and args.macs_ref_hw.lower() != "none":
        try:
            h, w = (int(v) for v in args.macs_ref_hw.lower().split("x"))
        except ValueError:
            raise SystemExit(
                f"--macs_ref_hw must look like {INPUT_HEIGHT}x{INPUT_WIDTH}, "
                f"got {args.macs_ref_hw!r}")
        args.macs_ref_hw = (h, w)
    else:
        args.macs_ref_hw = None

    return args


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
