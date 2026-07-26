"""Fine-tunes YOLO26-nano as a single-class, class-agnostic "obstacle"
detector. RT-MonoDepth + the navigation logic only need bounding boxes --
object identity is discarded downstream -- so every source class collapses
into class 0 ("obstacle") both here (single_cls=True) and at inference
time (perception/obstacle_detector.py just drops the class id).

Two ways to get a usable detector:
  1. (recommended) start from COCO-pretrained yolo26n.pt and fine-tune on
     data/sunrgbd_yolo/obstacle.yaml (built by
     scripts/prepare_sunrgbd.py --make_yolo_labels). Fast, robust, and
     works even if SUN RGB-D's 2D box annotations only cover part of the
     dataset on your particular download.
  2. train yolo26n from random weights (--pretrained "") if you'd rather
     not use any COCO weights; expect to need substantially more epochs
     and data to converge.

Usage:
    python -m wheelchair_nav.scripts.train_yolo_obstacle \
        --data ./data/sunrgbd_yolo/obstacle.yaml --epochs 60

Hyperparameters found by scripts/tune_yolo_obstacle.py (Optuna, TPE
sampler) can be applied directly with --hparams_json, which passes the
tuned optimizer/loss/augmentation values straight through to Ultralytics'
train():

    python -m wheelchair_nav.scripts.train_yolo_obstacle \
        --data ./data/sunrgbd_yolo/obstacle.yaml --epochs 60 \
        --hparams_json ./log_yolo/optuna_best_yolo_hparams.json
"""
from __future__ import annotations

import argparse
import json

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="./data/sunrgbd_yolo/obstacle.yaml")
    p.add_argument("--pretrained", default="yolo26n.pt",
                    help="Pretrained weights to start from, or '' to train from random init")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="./log_yolo")
    p.add_argument("--name", default="obstacle_yolo26n")
    p.add_argument("--hparams_json", default=None,
                    help="JSON from tune_yolo_obstacle.py ({'best_params': {...}}); its keys "
                         "(lr0, momentum, weight_decay, box, cls, hsv_h, ...) are passed straight "
                         "through to Ultralytics' model.train()")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.pretrained) if args.pretrained else YOLO("yolo26n.yaml")

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
        name=args.name,
        single_cls=True,  # collapse every class into one generic "obstacle"
        **extra_hparams,
    )


if __name__ == "__main__":
    main()
