"""End-to-end wheelchair navigation system evaluation, computed on top of
the per-frame CSV log produced by run_navigation.py (--log_csv).

Always computed from the log alone:
  - pipeline FPS (mean/min/max)
  - safety metrics: missed-stop rate (an obstacle was closer than the
    safety threshold but the system kept moving FORWARD) and false-stop
    rate (the system stopped although nothing was closer than the
    threshold)

Optional, if you provide ground truth:
  --gt_decisions CSV with columns: frame,decision
      -> decision accuracy + confusion matrix against the smoothed
         Free-Path-Selection output.

  --gt_distances JSON: [{"frame": int, "bbox": [x1,y1,x2,y2], "distance_m": float}, ...]
      (requires --video and --depth_weights to re-run RT-MonoDepth on the
      referenced frames)
      -> MAE / RMSE of the RT-MonoDepth + median-in-bbox distance estimate
         against a handful of tape-measured reference distances.

Usage:
    python -m wheelchair_nav.evaluation.eval_navigation_metrics \
        --nav_log ./out/navigation_demo_log.csv \
        --gt_decisions ./data/gt_decisions.csv \
        --gt_distances ./data/gt_distances.json \
        --video ./data/input.mp4 --depth_weights ./log_sunrgbd/RTMonoDepth_sunrgbd/models/best
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DECISION_LABELS = ["FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]


def load_nav_log(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def safety_metrics(rows, safe_distance):
    missed_stop = 0  # obstacle closer than threshold but system kept moving FORWARD
    false_stop = 0   # system stopped although nothing was closer than threshold
    total = len(rows)
    for row in rows:
        min_depth = row["min_depth_m"]
        min_depth = float("inf") if min_depth == "inf" else float(min_depth)
        decision = row["decision"]
        if min_depth < safe_distance and decision == "FORWARD":
            missed_stop += 1
        if decision == "STOP" and min_depth >= safe_distance:
            false_stop += 1
    return {
        "frames": total,
        "missed_stop_rate": missed_stop / total if total else 0.0,
        "false_stop_rate": false_stop / total if total else 0.0,
    }


def decision_accuracy(rows, gt_path):
    gt = {}
    with open(gt_path, newline="") as f:
        for row in csv.DictReader(f):
            gt[int(row["frame"])] = row["decision"]

    confusion = {a: {b: 0 for b in DECISION_LABELS} for a in DECISION_LABELS}
    correct = n = 0
    for row in rows:
        frame = int(row["frame"])
        if frame not in gt:
            continue
        pred, true = row["decision"], gt[frame]
        confusion[true][pred] += 1
        correct += int(pred == true)
        n += 1

    acc = correct / n if n else 0.0
    return acc, confusion, n


def make_depth_lookup(video_path, depth_weights, device):
    import cv2

    from wheelchair_nav.perception.depth_estimator import DepthEstimator

    estimator = DepthEstimator(depth_weights, device=device)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    def lookup(frame_idx: int, bbox):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return None
        depth_map = estimator.infer(frame)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi = depth_map[y1:y2, x1:x2]
        roi = roi[np.isfinite(roi)]
        return float(np.median(roi)) if roi.size else None

    return lookup, cap


def distance_error(gt_path, lookup):
    with open(gt_path) as f:
        entries = json.load(f)

    errs = []
    for e in entries:
        pred = lookup(e["frame"], e["bbox"])
        if pred is None:
            continue
        errs.append(abs(pred - e["distance_m"]))
    errs = np.array(errs)
    return {
        "n": len(errs),
        "mae_m": float(errs.mean()) if len(errs) else None,
        "rmse_m": float(np.sqrt((errs ** 2).mean())) if len(errs) else None,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nav_log", required=True, help="*_log.csv produced by run_navigation.py")
    p.add_argument("--safe_distance", type=float, default=1.0)
    p.add_argument("--gt_decisions", default=None)
    p.add_argument("--gt_distances", default=None)
    p.add_argument("--video", default=None, help="Required with --gt_distances")
    p.add_argument("--depth_weights", default=None, help="Required with --gt_distances")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_nav_log(args.nav_log)

    fps = np.array([float(r["fps"]) for r in rows if r["fps"]])
    print(f"Frames evaluated: {len(rows)}")
    print(f"Pipeline FPS: mean {fps.mean():.1f}, min {fps.min():.1f}, max {fps.max():.1f}")

    safety = safety_metrics(rows, args.safe_distance)
    print(
        f"Missed-stop rate (obstacle < {args.safe_distance}m but kept moving FORWARD): "
        f"{safety['missed_stop_rate'] * 100:.2f}%"
    )
    print(
        f"False-stop rate (stopped with nothing < {args.safe_distance}m): "
        f"{safety['false_stop_rate'] * 100:.2f}%"
    )

    if args.gt_decisions:
        acc, confusion, n = decision_accuracy(rows, args.gt_decisions)
        print(f"\nDecision accuracy vs {args.gt_decisions}: {acc * 100:.2f}% over {n} labeled frames")
        print("Confusion matrix (rows=ground truth, cols=predicted):")
        print("            " + "".join(f"{l:>12}" for l in DECISION_LABELS))
        for a in DECISION_LABELS:
            print(f"{a:>12}" + "".join(f"{confusion[a][b]:>12}" for b in DECISION_LABELS))

    if args.gt_distances:
        if not (args.video and args.depth_weights):
            raise SystemExit("--gt_distances requires --video and --depth_weights")
        lookup, cap = make_depth_lookup(args.video, args.depth_weights, args.device)
        result = distance_error(args.gt_distances, lookup)
        cap.release()
        print(
            f"\nDistance MAE vs {args.gt_distances}: {result['mae_m']} m "
            f"(RMSE {result['rmse_m']} m) over {result['n']} reference points"
        )


if __name__ == "__main__":
    main()
