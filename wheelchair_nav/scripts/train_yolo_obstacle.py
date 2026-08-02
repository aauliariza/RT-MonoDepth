"""Trains YOLO26-nano from scratch as a single-class, class-agnostic
"obstacle" detector, on SUN RGB-D's own 2D bounding-box annotations
converted by scripts/prepare_sunrgbd.py --make_yolo_labels
(data/sunrgbd_yolo/obstacle.yaml) -- no other dataset and no pretrained
checkpoint (COCO or otherwise) is used, matching the "from scratch" policy
already used for RT-MonoDepth and the FastDepth/YOLO-depth baselines.
RT-MonoDepth + the navigation logic only need bounding boxes -- object
identity is discarded downstream -- so every SUN RGB-D class collapses
into class 0 ("obstacle") both here (single_cls=True) and at inference
time (perception/obstacle_detector.py just drops the class id).

Usage:
    python -m wheelchair_nav.scripts.train_yolo_obstacle \
        --data ./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml --epochs 200

Hyperparameters found by scripts/tune_yolo_obstacle.py (Optuna, TPE
sampler) can be applied directly with --hparams_json, which passes the
tuned optimizer/loss/augmentation values straight through to Ultralytics'
train():

    python -m wheelchair_nav.scripts.train_yolo_obstacle \
        --data ./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml --epochs 200 \
        --hparams_json ./wheelchair_nav/log_yolo/optuna_best_yolo_hparams.json
"""
from __future__ import annotations

import argparse
import json

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml")
    p.add_argument("--pretrained", default="",
                    help="Empty (default) trains from random init (yolo26n.yaml) -- no pretrained "
                         "checkpoint is used. Only set this if you explicitly want to start from an "
                         "existing checkpoint instead.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="./wheelchair_nav/log_yolo")
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
