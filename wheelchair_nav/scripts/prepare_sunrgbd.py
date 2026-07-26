"""Preprocessing of the SUN RGB-D dataset (https://rgbd.cs.princeton.edu/)
into the RGB / metric-depth pairs used to train RT-MonoDepth from scratch
(wheelchair_nav/datasets/sunrgbd_dataset.py), and optionally into
single-class ("obstacle") YOLO detection labels used to fine-tune
YOLO26-nano.

SUN RGB-D ships as per-scene folders under kv1/, kv2/, realsense/ and
xtion/, each containing image/*.jpg, depth_bfx/*.png (sensor-refined
depth, preferred over the raw depth/ folder) and intrinsics.txt. Depth
PNGs store a bit-rotated 16-bit Kinect encoding (see `_decode_depth_png`
for the standard SUNRGBDtoolbox formula: a 3-bit rotate then /1000 for
meters). Some RealSense/Xtion scenes in public re-releases store plain
millimeter depth instead -- pass --depth_encoding raw_mm if the bitshift
decode looks wrong (mostly-zero depth maps), or leave the default
"bitshift" and let the automatic per-scene fallback in process_depth()
catch it.

Usage:
    python -m wheelchair_nav.scripts.prepare_sunrgbd \
        --sunrgbd_root /path/to/SUNRGBD \
        --out_dir ./data/sunrgbd_processed \
        --splits_dir ./splits_sunrgbd \
        --make_yolo_labels --yolo_out_dir ./data/sunrgbd_yolo
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import cv2
import numpy as np
from tqdm import tqdm


def _decode_depth_png(depth_png: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "bitshift":
        depth = np.bitwise_or(
            np.right_shift(depth_png, 3), np.left_shift(depth_png, 16 - 3)
        ).astype(np.uint16)
        depth_m = depth.astype(np.float32) / 1000.0
        depth_m[depth_m > 10.0] = 0.0
    else:  # raw_mm
        depth_m = depth_png.astype(np.float32) / 1000.0
    return depth_m


def _find_scenes(root: str):
    scenes = []
    for sensor in ("kv1", "kv2", "realsense", "xtion"):
        sensor_dir = os.path.join(root, sensor)
        if not os.path.isdir(sensor_dir):
            continue
        for dirpath, dirnames, _filenames in os.walk(sensor_dir):
            if "image" in dirnames and ("depth_bfx" in dirnames or "depth" in dirnames):
                scenes.append(dirpath)
    return sorted(scenes)


def _scene_rgb_depth(scene_dir: str):
    image_dir = os.path.join(scene_dir, "image")
    depth_dir = os.path.join(scene_dir, "depth_bfx")
    if not os.path.isdir(depth_dir):
        depth_dir = os.path.join(scene_dir, "depth")

    rgb_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    if not rgb_files or not depth_files:
        return None
    return rgb_files[0], depth_files[0]


def process_depth(args):
    scenes = _find_scenes(args.sunrgbd_root)
    if not scenes:
        raise SystemExit(
            f"No SUN RGB-D scenes found under {args.sunrgbd_root}. "
            "Expected kv1/kv2/realsense/xtion subfolders, each scene "
            "containing image/ and depth(_bfx)/."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = []
    n_skipped = 0

    for scene_dir in tqdm(scenes, desc="Processing SUN RGB-D scenes"):
        found = _scene_rgb_depth(scene_dir)
        if found is None:
            n_skipped += 1
            continue
        rgb_path, depth_path = found

        depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            n_skipped += 1
            continue

        depth_m = _decode_depth_png(depth_png, args.depth_encoding)
        if np.count_nonzero(depth_m) < 0.05 * depth_m.size:
            # Implausible decode (almost everything invalid) -- most likely
            # this scene's depth PNG uses the other encoding. Retry once.
            alt = "raw_mm" if args.depth_encoding == "bitshift" else "bitshift"
            depth_m = _decode_depth_png(depth_png, alt)

        scene_id = os.path.relpath(scene_dir, args.sunrgbd_root).replace(os.sep, "_")
        out_npy = os.path.join(args.out_dir, f"{scene_id}.npy")
        np.save(out_npy, depth_m.astype(np.float32))
        pairs.append((rgb_path, out_npy))

    print(f"Processed {len(pairs)} RGB/depth pairs, skipped {n_skipped} incomplete scenes.")
    if not pairs:
        raise SystemExit("No usable RGB/depth pairs were produced -- check --sunrgbd_root.")

    random.seed(args.seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_val = max(1, int(n * args.val_ratio))
    n_test = max(1, int(n * args.test_ratio))
    test_pairs = pairs[:n_test]
    val_pairs = pairs[n_test:n_test + n_val]
    train_pairs = pairs[n_test + n_val:]

    os.makedirs(args.splits_dir, exist_ok=True)
    for name, split in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
        split_path = os.path.join(args.splits_dir, f"{name}.txt")
        with open(split_path, "w") as f:
            for rgb, depth in split:
                f.write(f"{rgb} {depth}\n")
        print(f"  {name}: {len(split)} pairs -> {split_path}")


def process_yolo_labels(args):
    """Best-effort conversion of SUN RGB-D 2D annotations
    (annotation2Dfinal/index.json) into single-class ("obstacle") YOLO
    labels. The annotation2Dfinal layout has drifted across SUN RGB-D
    re-releases, so scenes that don't match the expected schema are
    skipped rather than aborting the whole run. If too few scenes convert,
    fine-tune YOLO26-nano from COCO-pretrained weights directly instead
    (see README) -- SUN RGB-D box labels are an optional refinement, not a
    requirement, for the obstacle detector.
    """
    scenes = _find_scenes(args.sunrgbd_root)
    img_out = os.path.join(args.yolo_out_dir, "images")
    lbl_out = os.path.join(args.yolo_out_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    n_ok, n_skip = 0, 0
    for scene_dir in tqdm(scenes, desc="Converting SUN RGB-D 2D boxes to YOLO"):
        ann_path = os.path.join(scene_dir, "annotation2Dfinal", "index.json")
        found = _scene_rgb_depth(scene_dir)
        if not os.path.isfile(ann_path) or found is None:
            n_skip += 1
            continue
        rgb_path, _ = found
        try:
            with open(ann_path, "r") as f:
                ann = json.load(f)
            img = cv2.imread(rgb_path)
            h, w = img.shape[:2]
            frame = ann["frames"][0]

            lines = []
            for poly in frame.get("polygon", []):
                xs, ys = poly.get("x", []), poly.get("y", [])
                if not xs or not ys:
                    continue
                x1, x2 = max(min(xs), 0), min(max(xs), w)
                y1, y2 = max(min(ys), 0), min(max(ys), h)
                if x2 <= x1 or y2 <= y1:
                    continue
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not lines:
                n_skip += 1
                continue

            scene_id = os.path.relpath(scene_dir, args.sunrgbd_root).replace(os.sep, "_")
            ext = os.path.splitext(rgb_path)[1]
            cv2.imwrite(os.path.join(img_out, f"{scene_id}{ext}"), img)
            with open(os.path.join(lbl_out, f"{scene_id}.txt"), "w") as f:
                f.write("\n".join(lines))
            n_ok += 1
        except Exception:
            n_skip += 1
            continue

    print(f"YOLO obstacle labels: {n_ok} scenes converted, {n_skip} skipped.")
    if n_ok < 200:
        print(
            "Few scenes converted -- consider fine-tuning YOLO26-nano from "
            "COCO-pretrained weights directly instead (see README)."
        )

    yaml_path = os.path.join(args.yolo_out_dir, "obstacle.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {os.path.abspath(args.yolo_out_dir)}\n"
            "train: images\nval: images\nnc: 1\nnames: ['obstacle']\n"
        )
    print(f"Wrote {yaml_path} (single class: obstacle)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sunrgbd_root", required=True,
                         help="Path to extracted SUNRGBD/ root (contains kv1/, kv2/, realsense/, xtion/)")
    parser.add_argument("--out_dir", default="./data/sunrgbd_processed",
                         help="Where metric-depth .npy files are written")
    parser.add_argument("--splits_dir", default="./splits_sunrgbd",
                         help="Where train/val/test.txt pair lists are written")
    parser.add_argument("--depth_encoding", choices=["bitshift", "raw_mm"], default="bitshift")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make_yolo_labels", action="store_true")
    parser.add_argument("--yolo_out_dir", default="./data/sunrgbd_yolo")
    return parser.parse_args()


def main():
    args = parse_args()
    process_depth(args)
    if args.make_yolo_labels:
        process_yolo_labels(args)


if __name__ == "__main__":
    main()
