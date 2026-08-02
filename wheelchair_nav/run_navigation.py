"""End-to-end offline test of the wheelchair navigation pipeline on a video
file:

    YOLO26-nano bboxes -> monocular depth model -> Obstacle List
    (median depth per bbox inner ROI) -> Sector-Based Free-Path Selection
    (priority CTR>L>R>FL>FR>STOP, N=3 hysteresis) -> Decision
    -> simulated wheelchair drive command.

The depth model is selectable via --depth_model: RT-MonoDepth (full,
default), FastDepth, or YOLO26n/s-depth -- any of the four models trained
in this project can drive the same pipeline, since
DepthEstimator/FastDepthEstimator/YoloDepthEstimator all expose the same
.infer(frame_bgr) -> metric depth (meters) interface (see
evaluation/eval_depth_comparison.py, which uses the same three classes).

Writes an annotated output video: RGB (left) with per-obstacle bboxes +
distance, the 5 sectors (FL|L|CTR|R|FR) tinted by status (green = the
chosen free path, amber = free but not chosen, red = blocked), a bottom
banner (decision, nearest-obstacle distance, FPS), concatenated side by
side with the colorized depth map (right, same resolution) -- plus a
per-frame CSV log used by evaluation/eval_navigation_metrics.py.

Usage (RT-MonoDepth, default):
    python -m wheelchair_nav.run_navigation \
        --video path/to/input.mp4 \
        --depth_model rtmonodepth \
        --depth_weights ./wheelchair_nav/log_sunrgbd/RTMonoDepth_sunrgbd/models/best \
        --yolo_weights ./wheelchair_nav/log_yolo/obstacle_yolo26n/weights/best.pt \
        --output ./wheelchair_nav/out/navigation_demo.mp4

Usage (FastDepth):
    python -m wheelchair_nav.run_navigation \
        --video path/to/input.mp4 \
        --depth_model fastdepth \
        --depth_weights ./wheelchair_nav/log_fastdepth/FastDepth_sunrgbd/models/best \
        --yolo_weights ./wheelchair_nav/log_yolo/obstacle_yolo26n/weights/best.pt \
        --output ./wheelchair_nav/out/navigation_demo_fastdepth.mp4

Usage (YOLO26n-depth / YOLO26s-depth):
    python -m wheelchair_nav.run_navigation \
        --video path/to/input.mp4 \
        --depth_model yolo26n-depth \
        --depth_weights ./wheelchair_nav/log_yolo_depth/yolo26n_depth/weights/best.pt \
        --yolo_weights ./wheelchair_nav/log_yolo/obstacle_yolo26n/weights/best.pt \
        --output ./wheelchair_nav/out/navigation_demo_yolo26n_depth.mp4
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

from wheelchair_nav.config import MAX_DEPTH_M, MIN_DEPTH_M, SAFE_DISTANCE_M, SECTOR_NAMES  # noqa: E402
from wheelchair_nav.navigation.controller import WheelchairController  # noqa: E402
from wheelchair_nav.navigation.free_path import DECISION_TO_SECTOR, FreePathSelector  # noqa: E402
from wheelchair_nav.navigation.sectors import compute_sector_depths  # noqa: E402
from wheelchair_nav.perception.obstacle_detector import ObstacleDetector  # noqa: E402
from wheelchair_nav.perception.obstacle_list import build_obstacle_list  # noqa: E402

DEPTH_MODEL_CHOICES = ("rtmonodepth", "fastdepth", "yolo26n-depth", "yolo26s-depth")


def build_depth_estimator(args):
    """Instantiates the selected monocular depth model behind the common
    .infer(frame_bgr) -> (H, W) float32 metric-depth interface, so the rest
    of the pipeline (Obstacle List, sectors, hysteresis, overlay) doesn't
    need to know which model produced the depth map.

    --depth_weights means different things depending on --depth_model:
      rtmonodepth/fastdepth -> a folder (encoder.pth+depth.pth, or
        fastdepth.pth respectively) saved by the matching train_*.py script.
      yolo26n-depth/yolo26s-depth -> a single Ultralytics checkpoint file
        (e.g. runs/.../weights/best.pt) saved by train_yolo_depth.py.
    """
    if args.depth_model == "rtmonodepth":
        from wheelchair_nav.perception.depth_estimator import DepthEstimator

        return DepthEstimator(
            args.depth_weights, device=args.device,
            min_depth_m=args.min_depth, max_depth_m=args.max_depth,
        )
    if args.depth_model == "fastdepth":
        from wheelchair_nav.baselines.fastdepth_estimator import FastDepthEstimator

        return FastDepthEstimator(
            args.depth_weights, device=args.device,
            min_depth_m=args.min_depth, max_depth_m=args.max_depth,
        )
    if args.depth_model in ("yolo26n-depth", "yolo26s-depth"):
        from wheelchair_nav.baselines.yolo_depth_estimator import YoloDepthEstimator

        return YoloDepthEstimator(args.depth_weights, device=args.device, imgsz=args.yolo_depth_imgsz)

    raise SystemExit(f"Unknown --depth_model: {args.depth_model}")

DECISION_COLOR = {
    "FORWARD": (60, 200, 60),
    "TURN_LEFT": (220, 90, 220),
    "TURN_RIGHT": (220, 90, 220),
    "TURN_FAR_LEFT": (200, 60, 200),
    "TURN_FAR_RIGHT": (200, 60, 200),
    "STOP": (40, 40, 220),
}

SECTOR_LABEL = {name: name.rstrip("0123456789") for name in SECTOR_NAMES}

# BGR
COLOR_BLOCKED = (0, 0, 220)     # red -- something inside is closer than the safety threshold
COLOR_CHOSEN = (0, 200, 0)      # green -- this sector is free AND is the one the decision picked
COLOR_FREE_ALT = (0, 200, 220)  # amber -- free, but a higher-priority sector was chosen instead


def colorize_depth(depth_m: np.ndarray, max_depth: float) -> np.ndarray:
    norm = np.clip(depth_m / max_depth, 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)


def draw_sector_overlay(vis, sector_depths, chosen_sector, safe_distance, alpha=0.35):
    h, w = vis.shape[:2]
    n = len(SECTOR_NAMES)
    overlay = vis.copy()

    for i, name in enumerate(SECTOR_NAMES):
        x1, x2 = int(w * i / n), int(w * (i + 1) / n)
        depth = sector_depths.get(name, float("inf"))
        if depth < safe_distance:
            color = COLOR_BLOCKED
        elif name == chosen_sector:
            color = COLOR_CHOSEN
        else:
            color = COLOR_FREE_ALT
        cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, dst=vis)

    ty = h // 2
    for i, name in enumerate(SECTOR_NAMES):
        x1, x2 = int(w * i / n), int(w * (i + 1) / n)
        cx = (x1 + x2) // 2
        if i > 0:
            cv2.line(vis, (x1, 0), (x1, h), (110, 110, 110), 1)

        label = SECTOR_LABEL[name]
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(vis, label, (cx - lw // 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        depth = sector_depths.get(name, float("inf"))
        depth_str = f"{depth:.2f}m" if np.isfinite(depth) else "inf"
        (dw, _), _ = cv2.getTextSize(depth_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(vis, depth_str, (cx - dw // 2, ty + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def draw_overlay(frame, obstacles, decision, fps, sector_depths, safe_distance):
    h, w = frame.shape[:2]
    vis = frame.copy()

    chosen_sector = DECISION_TO_SECTOR.get(decision)
    draw_sector_overlay(vis, sector_depths, chosen_sector, safe_distance)

    for obs in obstacles:
        x1, y1, x2, y2 = obs.bbox
        color = COLOR_BLOCKED if obs.depth_m < safe_distance else COLOR_CHOSEN
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"obstacle {obs.depth_m:.2f}m", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    min_depth = min(sector_depths.values()) if sector_depths else float("inf")
    banner_color = DECISION_COLOR.get(decision, (255, 255, 255))
    cv2.rectangle(vis, (0, h - 40), (w, h), (20, 20, 20), -1)
    cv2.putText(vis, decision, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, banner_color, 2)

    obs_str = f"OBS: {min_depth:.2f}m" if np.isfinite(min_depth) else "OBS: --"
    (ow, _), _ = cv2.getTextSize(obs_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(vis, obs_str, (w // 2 - ow // 2, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 140), 2)

    fps_str = f"{fps:.1f} FPS"
    (fw, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(vis, fps_str, (w - fw - 15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis


_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


def ensure_video_extension(output_path: str) -> str:
    """Appends .mp4 if --output has no recognized video extension.

    A path with no (or an unrecognized) extension still gets a byte-for-byte
    valid video written to it by OpenCV -- but Windows Explorer (and most
    media players' file-association logic) only recognizes a video file by
    its extension, so a working file named e.g. "navigation_demo" shows up
    as a generic, unopenable "File" with no icon. This makes that class of
    mistake impossible instead of documenting it.
    """
    if not output_path.lower().endswith(_VIDEO_EXTENSIONS):
        fixed = output_path + ".mp4"
        print(f"Warning: --output '{output_path}' has no video file extension; "
              f"writing to '{fixed}' instead so your OS/player recognizes it.")
        return fixed
    return output_path


def open_video_writer(output_path: str, fps: float, size: tuple):
    """Opens a cv2.VideoWriter, trying a few fourcc/container combinations.

    cv2.VideoWriter fails *silently* when a codec isn't available: isOpened()
    is False, but write() raises nothing and just does nothing, so a full run
    can finish "successfully" while producing a 0-byte or otherwise corrupt,
    unplayable file -- no traceback, no error, just a bad file. This checks
    isOpened() explicitly and falls back to more widely-supported codecs
    before giving up with an actionable error.
    """
    candidates = [
        (output_path, "mp4v"),
        (output_path, "avc1"),
        (os.path.splitext(output_path)[0] + ".avi", "MJPG"),
        (os.path.splitext(output_path)[0] + ".avi", "XVID"),
    ]
    for path, fourcc_name in candidates:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_name), fps, size)
        if writer.isOpened():
            if path != output_path:
                print(f"Warning: could not open '{output_path}' with any MP4 codec on this "
                      f"OpenCV build; writing '{fourcc_name}' to '{path}' instead.")
            return writer, path
        writer.release()

    raise SystemExit(
        "Could not open a video writer with any of the tried codecs "
        f"({[c[1] for c in candidates]}). This means your OpenCV build has no working codec "
        "backend for video writing (check with: python3 -c \"import cv2; "
        "print(cv2.getBuildInformation())\" | grep -i ffmpeg -- it should say YES). Fix by "
        "installing an OpenCV build with FFmpeg support, e.g.:\n"
        "  pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python\n"
        "or, if that still fails:\n"
        "  conda install -c conda-forge opencv"
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--depth_model", default="rtmonodepth", choices=DEPTH_MODEL_CHOICES,
                    help="Which trained monocular depth model drives the pipeline (default: rtmonodepth)")
    p.add_argument("--depth_weights", required=True,
                    help="rtmonodepth/fastdepth: folder saved by the matching train_*.py script "
                         "(encoder.pth+depth.pth, or fastdepth.pth). "
                         "yolo26n-depth/yolo26s-depth: path to a single .pt checkpoint "
                         "from train_yolo_depth.py")
    p.add_argument("--yolo_weights", required=True,
                    help="Path to a YOLO26-nano checkpoint trained from scratch on SUN RGB-D "
                         "(scripts/train_yolo_obstacle.py) -- no COCO-pretrained or other checkpoint")
    p.add_argument("--output", default="./wheelchair_nav/out/navigation_demo.mp4")
    p.add_argument("--log_csv", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--safe_distance", type=float, default=SAFE_DISTANCE_M)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--min_depth", type=float, default=MIN_DEPTH_M,
                    help="Only used by --depth_model rtmonodepth/fastdepth")
    p.add_argument("--max_depth", type=float, default=MAX_DEPTH_M,
                    help="Only used by --depth_model rtmonodepth/fastdepth")
    p.add_argument("--yolo_depth_imgsz", type=int, default=640,
                    help="Only used by --depth_model yolo26n-depth/yolo26s-depth")
    p.add_argument("--max_depth_vis", type=float, default=6.0)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.output = ensure_video_extension(args.output)
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    log_csv = args.log_csv or os.path.splitext(args.output)[0] + "_log.csv"

    depth_estimator = build_depth_estimator(args)
    detector = ObstacleDetector(args.yolo_weights, conf=args.conf, device=args.device)
    selector = FreePathSelector(safe_distance_m=args.safe_distance)
    controller = WheelchairController()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Side-by-side output: RGB+overlay (left) and colorized depth (right), same
    # resolution, concatenated into one frame twice the input width.
    writer, output_path = open_video_writer(args.output, fps_in, (w * 2, h))

    csv_file = open(log_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "decision", "num_obstacles", "min_depth_m", "fps"] + list(SECTOR_NAMES))

    frame_idx = 0
    fps_ema = 0.0
    decision_counts = {}

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
        vis = draw_overlay(frame, obstacles, decision, fps_ema, sector_depths, args.safe_distance)
        depth_vis = colorize_depth(depth_map, args.max_depth_vis)
        combined = np.hstack([vis, depth_vis])

        writer.write(combined)
        if args.show:
            cv2.imshow("wheelchair navigation", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        csv_writer.writerow([
            frame_idx, decision, len(obstacles),
            f"{min_depth:.3f}" if np.isfinite(min_depth) else "inf",
            f"{fps:.2f}",
        ] + [
            f"{sector_depths[s]:.3f}" if np.isfinite(sector_depths[s]) else "inf" for s in SECTOR_NAMES
        ])

        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames -> {output_path}")
    print("  " + " | ".join(f"{d}: {n}" for d, n in sorted(decision_counts.items())))
    print(f"  Mean pipeline FPS: {fps_ema:.1f}")
    print(f"  Per-frame log: {log_csv}")


if __name__ == "__main__":
    main()
