"""Trains YOLO26n-depth or YOLO26s-depth -- Ultralytics' UNMODIFIED native
monocular depth architecture (cfg/models/26/yolo26-depth.yaml, 'n'/'s'
scale) -- on the exact same SUN RGB-D train/val/test split as RT-MonoDepth
and FastDepth (see scripts/prepare_sunrgbd.py --make_yolo_depth_layout),
for an apples-to-apples comparison in evaluation/eval_depth_comparison.py.

Training, the loss (SILog + depth-gradient), and post-training metric-scale
calibration are entirely native Ultralytics
(ultralytics/models/yolo/depth/: DepthTrainer/DepthValidator); this script
only wraps model.train() with this project's CLI conventions -- the same
pattern as scripts/train_yolo_obstacle.py.

Usage:
    python -m wheelchair_nav.scripts.train_yolo_depth \
        --variant n --data ./wheelchair_nav/data/sunrgbd_yolo_depth/depth_comparison.yaml --epochs 60

Hyperparameters found by scripts/tune_yolo_depth.py (Optuna, TPE sampler)
can be applied directly with --hparams_json:

    python -m wheelchair_nav.scripts.train_yolo_depth \
        --variant n --data ./wheelchair_nav/data/sunrgbd_yolo_depth/depth_comparison.yaml --epochs 60 \
        --hparams_json ./wheelchair_nav/log_yolo_depth/optuna_best_yolo26n_depth_hparams.json
"""
from __future__ import annotations

import argparse
import json

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["n", "s"], default="n", help="YOLO26 scale: n (nano) or s (small)")
    p.add_argument("--data", default="./wheelchair_nav/data/sunrgbd_yolo_depth/depth_comparison.yaml")
    p.add_argument("--pretrained", default="",
                    help="Empty (default) trains from random init (yolo26{variant}-depth.yaml) -- "
                         "no pretrained checkpoint is used unless a path is passed explicitly")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="./wheelchair_nav/log_yolo_depth")
    p.add_argument("--name", default=None, help="default: yolo26{variant}_depth_sunrgbd")
    p.add_argument("--hparams_json", default=None,
                    help="JSON from tune_yolo_depth.py ({'best_params': {...}}); its keys "
                         "(lr0, momentum, weight_decay, dlog, dgrad, dlam, ...) are passed straight "
                         "through to Ultralytics' model.train()")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.pretrained) if args.pretrained else YOLO(f"yolo26{args.variant}-depth.yaml")
    name = args.name or f"yolo26{args.variant}_depth_sunrgbd"

    extra_hparams = {}
    if args.hparams_json:
        with open(args.hparams_json, "r") as f:
            extra_hparams = json.load(f).get("best_params", {})
        print(f"Loaded tuned hyperparameters from {args.hparams_json}: {extra_hparams}")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=name,
        **extra_hparams,
    )


if __name__ == "__main__":
    main()
