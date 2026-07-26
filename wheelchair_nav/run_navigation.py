"""End-to-end offline test of the wheelchair navigation pipeline on a video
file:

    YOLO26-nano bboxes -> RT-MonoDepth metric depth -> Obstacle List
    (median depth per bbox inner ROI) -> Sector-Based Free-Path Selection
    (priority CTR>L>R>FL>FR>STOP, N=3 hysteresis) -> Decision
    -> simulated wheelchair drive command.

Writes an annotated output video (bboxes + per-obstacle distance, sector
grid, decision banner, FPS, a depth-colormap picture-in-picture) plus a
per-frame CSV log used by evaluation/eval_navigation_metrics.py.

Usage:
    python -m wheelchair_nav.run_navigation \
        --video path/to/input.mp4 \
        --depth_weights ./log_sunrgbd/RTMonoDepth_sunrgbd/models/best \
        --yolo_weights ./log_yolo/obstacle_yolo26n/weights/best.pt \
        --output ./out/navigation_demo.mp4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wheelchair_nav.config import SAFE_DISTANCE_M, SECTOR_NAMES  # noqa: E402
from wheelchair_nav.navigation.controller import WheelchairController  # noqa: E402
from wheelchair_nav.navigation.free_path import FreePathSelector  # noqa: E402
from wheelchair_nav.navigation.sectors import compute_sector_depths  # noqa: E402
from wheelchair_nav.perception.depth_estimator import DepthEstimator  # noqa: E402
from wheelchair_nav.perception.obstacle_detector import ObstacleDetector  # noqa: E402
from wheelchair_nav.perception.obstacle_list import build_obstacle_list  # noqa: E402

DECISION_COLOR = {
    "FORWARD": (60, 200, 60),
    "TURN_LEFT": (60, 170, 240),
    "TURN_RIGHT": (60, 170, 240),
    "STOP": (40, 40, 220),
}


def colorize_depth(depth_m: np.ndarray, max_depth: float) -> np.ndarray:
    norm = np.clip(depth_m / max_depth, 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(255 - gray, cv2.COLORMAP_JET)


def draw_overlay(frame, obstacles, decision, fps, safe_distance):
    h, w = frame.shape[:2]
    vis = frame.copy()

    for i in range(1, len(SECTOR_NAMES)):
        x = int(w * i / len(SECTOR_NAMES))
        cv2.line(vis, (x, 40), (x, h), (90, 90, 90), 1)

    for obs in obstacles:
        x1, y1, x2, y2 = obs.bbox
        color = (0, 0, 255) if obs.depth_m < safe_distance else (0, 200, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"Depth: {obs.depth_m:.2f}m", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    banner_color = DECISION_COLOR.get(decision, (255, 255, 255))
    cv2.rectangle(vis, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.putText(vis, f"Decision: {decision}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, banner_color, 2)
    cv2.putText(vis, f"FPS: {fps:.1f}", (w - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return vis


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--depth_weights", required=True, help="Folder with encoder.pth + depth.pth")
    p.add_argument("--yolo_weights", required=True,
                    help="Path to a YOLO26-nano checkpoint trained from scratch on SUN RGB-D "
                         "(scripts/train_yolo_obstacle.py) -- no COCO-pretrained or other checkpoint")
    p.add_argument("--output", default="./out/navigation_demo.mp4")
    p.add_argument("--log_csv", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--safe_distance", type=float, default=SAFE_DISTANCE_M)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--max_depth_vis", type=float, default=6.0)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    log_csv = args.log_csv or os.path.splitext(args.output)[0] + "_log.csv"

    depth_estimator = DepthEstimator(args.depth_weights, device=args.device)
    detector = ObstacleDetector(args.yolo_weights, conf=args.conf, device=args.device)
    selector = FreePathSelector(safe_distance_m=args.safe_distance)
    controller = WheelchairController()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    csv_file = open(log_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "decision", "num_obstacles", "min_depth_m", "fps"] + list(SECTOR_NAMES))

    frame_idx = 0
    fps_ema = 0.0
    stop_frames = turn_frames = forward_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()

        depth_map = depth_estimator.infer(frame)
        boxes = detector.detect(frame)
        obstacles = build_obstacle_list(depth_map, boxes)
        sector_depths = compute_sector_depths(obstacles, image_width=w)
        decision = selector.update(sector_depths)
        controller.step(decision)

        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_ema = fps if frame_idx == 0 else 0.9 * fps_ema + 0.1 * fps

        min_depth = min(sector_depths.values()) if sector_depths else float("inf")
        vis = draw_overlay(frame, obstacles, decision, fps_ema, args.safe_distance)

        depth_thumb = colorize_depth(depth_map, args.max_depth_vis)
        th, tw = h // 4, w // 4
        depth_thumb = cv2.resize(depth_thumb, (tw, th))
        vis[h - th:h, 0:tw] = depth_thumb

        writer.write(vis)
        if args.show:
            cv2.imshow("wheelchair navigation", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        csv_writer.writerow([
            frame_idx, decision, len(obstacles),
            f"{min_depth:.3f}" if np.isfinite(min_depth) else "inf",
            f"{fps:.2f}",
        ] + [
            f"{sector_depths[s]:.3f}" if np.isfinite(sector_depths[s]) else "inf" for s in SECTOR_NAMES
        ])

        if decision == "STOP":
            stop_frames += 1
        elif decision == "FORWARD":
            forward_frames += 1
        else:
            turn_frames += 1
        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames -> {args.output}")
    print(f"  FORWARD: {forward_frames} | TURN: {turn_frames} | STOP: {stop_frames}")
    print(f"  Mean pipeline FPS: {fps_ema:.1f}")
    print(f"  Per-frame log: {log_csv}")


if __name__ == "__main__":
    main()
