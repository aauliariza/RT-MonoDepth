"""Preprocessing of the SUN RGB-D dataset (https://rgbd.cs.princeton.edu/)
into:
  - the RGB / metric-depth pairs used to train RT-MonoDepth from scratch
    (wheelchair_nav/datasets/sunrgbd_dataset.py);
  - single-class ("obstacle") YOLO detection labels converted from SUN
    RGB-D's OWN 2D bounding-box annotations (annotation2Dfinal/index.json,
    --make_yolo_labels). This is the only dataset used to train
    YOLO26-nano in this project -- no COCO, no other dataset, and
    scripts/train_yolo_obstacle.py trains from random weights, never from
    a pretrained checkpoint;
  - (optional, comparison baselines only) the images/{split}+depth/{split}
    layout Ultralytics' native depth task expects, mirroring the exact
    same train/val/test split used for RT-MonoDepth/FastDepth, so
    YOLO26n-depth/YOLO26s-depth can be trained and compared
    apples-to-apples against RT-MonoDepth on identical images
    (--make_yolo_depth_layout; see evaluation/eval_depth_comparison.py).

SUN RGB-D ships as per-scene folders under kv1/, kv2/, realsense/ and
xtion/, each containing image/*.jpg, depth_bfx/*.png (sensor-refined
depth, preferred over the raw depth/ folder), intrinsics.txt, and
annotation2Dfinal/index.json (2D polygon annotations). Depth PNGs store a
bit-rotated 16-bit Kinect encoding (see `_decode_depth_png` for the
standard SUNRGBDtoolbox formula: a 3-bit rotate then /1000 for meters).
Some RealSense/Xtion scenes in public re-releases store plain millimeter
depth instead -- pass --depth_encoding raw_mm if the bitshift decode looks
wrong (mostly-zero depth maps), or leave the default "bitshift" and let
the automatic per-scene fallback in process_depth() catch it.

annotation2Dfinal/index.json schema (verified against community SUN RGB-D
parsers, e.g. Mask_RCNN-for-SUN-RGB-D's samples/sun/SUNRGBD.py):
    {"frames": [{"polygon": [{"x": [...], "y": [...], "object": <int index
     into "objects">}, ...]}], "objects": [{"name": "chair"}, null, ...]}
x/y are pixel coordinates in the associated image/*.jpg. "objects" entries
can be null (deleted/merged objects); see _parse_annotation2d().

Usage:
    python -m wheelchair_nav.scripts.prepare_sunrgbd \
        --sunrgbd_root /path/to/SUNRGBD \
        --out_dir ./wheelchair_nav/data/sunrgbd_processed \
        --splits_dir ./wheelchair_nav/splits_sunrgbd \
        --make_yolo_labels --yolo_out_dir ./wheelchair_nav/data/sunrgbd_yolo \
        --make_yolo_depth_layout --yolo_depth_out_dir ./wheelchair_nav/data/sunrgbd_yolo_depth
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from wheelchair_nav.config import MAX_DEPTH_M


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
    splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    for name, split in splits.items():
        split_path = os.path.join(args.splits_dir, f"{name}.txt")
        with open(split_path, "w") as f:
            for rgb, depth in split:
                f.write(f"{rgb} {depth}\n")
        print(f"  {name}: {len(split)} pairs -> {split_path}")

    return splits


def _link_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst) or os.path.islink(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def _split_items(items: list, val_ratio: float, test_ratio: float, seed: int) -> dict:
    """Shuffles a list with its own local RNG (so it doesn't perturb the
    global random.seed() state process_depth() also relies on) and splits
    it the same way process_depth() splits RGB/depth pairs.
    """
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    test_items = items[:n_test]
    val_items = items[n_test:n_test + n_val]
    train_items = items[n_test + n_val:]
    return {"train": train_items, "val": val_items, "test": test_items}


def _parse_annotation2d(ann_path: str, img_shape, exclude_classes: set, max_box_area_ratio: float):
    """Parses one SUN RGB-D annotation2Dfinal/index.json into class-agnostic
    YOLO label lines ("0 cx cy w h", normalized), using the schema
    frames[0]["polygon"][i] = {"x": [...], "y": [...], "object": idx} and
    objects[idx]["name"] to drop room-surface classes (wall/floor/ceiling
    by default) that would otherwise become near-full-frame "obstacle"
    boxes and poison training. --max_box_area_ratio is a second safety net
    against any remaining oversized polygon regardless of its class name.
    """
    with open(ann_path, "r") as f:
        ann = json.load(f)
    h, w = img_shape[:2]
    frame = ann["frames"][0]
    objects = ann.get("objects", [])
    max_area = max_box_area_ratio * w * h

    lines = []
    for poly in frame.get("polygon", []):
        xs, ys = poly.get("x", []), poly.get("y", [])
        if not xs or not ys:
            continue

        obj_idx = poly.get("object")
        if isinstance(obj_idx, int) and 0 <= obj_idx < len(objects) and objects[obj_idx]:
            name = str(objects[obj_idx].get("name", "")).strip().lower()
            if name in exclude_classes:
                continue

        x1, x2 = max(min(xs), 0), min(max(xs), w)
        y1, y2 = max(min(ys), 0), min(max(ys), h)
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) > max_area:
            continue

        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


def process_yolo_labels(args):
    """Converts SUN RGB-D's own 2D bounding-box annotations
    (annotation2Dfinal/index.json) into single-class ("obstacle") YOLO
    detection labels -- the only dataset scripts/train_yolo_obstacle.py
    trains on (no COCO, no other dataset, no pretrained weights). Scenes
    missing annotation2Dfinal/, or whose JSON doesn't parse (a handful of
    scenes in the official release have malformed JSON), are skipped and
    counted, not silently substituted with another data source.
    """
    exclude_classes = {c.strip().lower() for c in args.exclude_classes.split(",") if c.strip()}
    scenes = _find_scenes(args.sunrgbd_root)
    if not scenes:
        raise SystemExit(
            f"No SUN RGB-D scenes found under {args.sunrgbd_root}. "
            "Expected kv1/kv2/realsense/xtion subfolders, each scene "
            "containing image/ and annotation2Dfinal/."
        )

    items = []  # (scene_id, rgb_path, ext, yolo_lines)
    n_skip = 0
    for scene_dir in tqdm(scenes, desc="Converting SUN RGB-D 2D boxes to YOLO"):
        ann_path = os.path.join(scene_dir, "annotation2Dfinal", "index.json")
        found = _scene_rgb_depth(scene_dir)
        if not os.path.isfile(ann_path) or found is None:
            n_skip += 1
            continue
        rgb_path, _ = found
        try:
            img = cv2.imread(rgb_path)
            if img is None:
                n_skip += 1
                continue
            lines = _parse_annotation2d(ann_path, img.shape, exclude_classes, args.max_box_area_ratio)
            if not lines:
                n_skip += 1
                continue
            scene_id = os.path.relpath(scene_dir, args.sunrgbd_root).replace(os.sep, "_")
            ext = os.path.splitext(rgb_path)[1]
            items.append((scene_id, rgb_path, ext, lines))
        except Exception:
            n_skip += 1
            continue

    print(f"YOLO obstacle labels: {len(items)} scenes converted, {n_skip} skipped "
          "(missing/unparseable annotation2Dfinal, or no non-structural objects).")
    if not items:
        raise SystemExit(
            "No SUN RGB-D scenes produced usable obstacle labels -- check that "
            "--sunrgbd_root contains annotation2Dfinal/index.json per scene, and "
            "that --exclude_classes isn't dropping everything."
        )

    splits = _split_items(items, args.val_ratio, args.test_ratio, args.seed)
    for split_name, split_items in splits.items():
        img_dir = os.path.join(args.yolo_out_dir, "images", split_name)
        lbl_dir = os.path.join(args.yolo_out_dir, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for scene_id, rgb_path, ext, lines in split_items:
            _link_or_copy(rgb_path, os.path.join(img_dir, f"{scene_id}{ext}"))
            with open(os.path.join(lbl_dir, f"{scene_id}.txt"), "w") as f:
                f.write("\n".join(lines))
        print(f"  {split_name}: {len(split_items)} images -> {img_dir}")

    yaml_path = os.path.join(args.yolo_out_dir, "obstacle.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {os.path.abspath(args.yolo_out_dir)}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 1\n"
            "names: ['obstacle']\n"
        )
    print(f"Wrote {yaml_path} (single class: obstacle)")


def process_yolo_depth_layout(args, splits: dict) -> None:
    """Mirrors the exact train/val/test split used for RT-MonoDepth/FastDepth
    into the images/{split} + depth/{split} layout Ultralytics' native depth
    task expects (ultralytics/data/dataset.py pairs images/train/x.jpg with
    depth/train/x.npy by swapping the last "images" path component), via
    symlinks so no data is duplicated on disk. This gives YOLO26n-depth and
    YOLO26s-depth literally the same images as RT-MonoDepth/FastDepth --
    the basis for the apples-to-apples comparison in
    evaluation/eval_depth_comparison.py.
    """
    out_dir = args.yolo_depth_out_dir
    total = 0
    for split_name, pairs in splits.items():
        img_dir = os.path.join(out_dir, "images", split_name)
        depth_dir = os.path.join(out_dir, "depth", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        for rgb_path, depth_path in pairs:
            stem = os.path.splitext(os.path.basename(depth_path))[0]
            ext = os.path.splitext(rgb_path)[1]
            _link_or_copy(rgb_path, os.path.join(img_dir, f"{stem}{ext}"))
            _link_or_copy(depth_path, os.path.join(depth_dir, f"{stem}.npy"))
        total += len(pairs)

    yaml_path = os.path.join(out_dir, "depth_comparison.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {os.path.abspath(out_dir)}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 1\n"
            "names:\n"
            "  0: depth\n"
            "channels: 3\n"
            f"max_depth: {args.max_depth}\n"
        )
    print(f"Wrote {yaml_path} ({total} images mirrored from {args.splits_dir})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sunrgbd_root", required=True,
                         help="Path to extracted SUNRGBD/ root (contains kv1/, kv2/, realsense/, xtion/)")
    parser.add_argument("--out_dir", default="./wheelchair_nav/data/sunrgbd_processed",
                         help="Where metric-depth .npy files are written")
    parser.add_argument("--splits_dir", default="./wheelchair_nav/splits_sunrgbd",
                         help="Where train/val/test.txt pair lists are written")
    parser.add_argument("--depth_encoding", choices=["bitshift", "raw_mm"], default="bitshift")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make_yolo_labels", action="store_true",
                         help="Convert SUN RGB-D's own annotation2Dfinal/ 2D boxes into YOLO obstacle "
                              "labels -- required before training YOLO26-nano (train_yolo_obstacle.py)")
    parser.add_argument("--yolo_out_dir", default="./wheelchair_nav/data/sunrgbd_yolo")
    parser.add_argument("--exclude_classes", default="wall,floor,ceiling",
                         help="Comma-separated SUN RGB-D object names (case-insensitive) to drop -- "
                              "room surfaces are not physical obstacles and would otherwise become "
                              "near-full-frame boxes")
    parser.add_argument("--max_box_area_ratio", type=float, default=0.9,
                         help="Drop any box covering more than this fraction of the image area, "
                              "regardless of class name (second safety net against mislabeled polygons)")
    parser.add_argument("--make_yolo_depth_layout", action="store_true",
                         help="Also mirror the train/val/test split into the images/+depth/ layout "
                              "used by YOLO26n-depth/YOLO26s-depth (baselines/yolo_depth_estimator.py)")
    parser.add_argument("--yolo_depth_out_dir", default="./wheelchair_nav/data/sunrgbd_yolo_depth")
    parser.add_argument("--max_depth", type=float, default=MAX_DEPTH_M,
                         help="Written into depth_comparison.yaml so DepthValidator's metric range matches")
    return parser.parse_args()


def main():
    args = parse_args()
    splits = process_depth(args)
    if args.make_yolo_labels:
        process_yolo_labels(args)
    if args.make_yolo_depth_layout:
        process_yolo_depth_layout(args, splits)


if __name__ == "__main__":
    main()
