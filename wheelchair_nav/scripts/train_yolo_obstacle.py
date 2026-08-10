"""Fine-tunes YOLO26-nano as a single-class, class-agnostic "obstacle"
detector on SUN RGB-D's own 2D bounding-box annotations, converted by
scripts/prepare_sunrgbd.py --make_yolo_labels
(data/sunrgbd_yolo/obstacle.yaml). RT-MonoDepth + the navigation logic
only need bounding boxes -- object identity is discarded downstream -- so
every SUN RGB-D class collapses into class 0 ("obstacle") both here
(single_cls=True) and at inference time (perception/obstacle_detector.py
just drops the class id).

WHY THIS ONE STARTS FROM COCO WEIGHTS WHILE EVERY DEPTH MODEL DOES NOT
----------------------------------------------------------------------
The five depth models (RT-MonoDepth, Ghost-Depth, FastDepth, YOLO26n/s-
depth) are the OBJECT of comparison, so they must all get identical
treatment; pretrained checkpoints are not available symmetrically across
them (Ghost-Depth has none at all, RT-MonoDepth has no indoor one), so
random init is the only setup under which their comparison is valid.

This detector is not compared against anything -- it is a FIXED component
of the perception stage. No fairness constraint applies to it, and one
concrete failure mode argues strongly for pretraining it: depth is only
ever read INSIDE a detected box (obstacle_list.py takes the median over
the inner 60% of each bbox), so an object the detector misses is invisible
to every depth model alike. A weak detector therefore flattens the very
differences eval_navigation_metrics.py is supposed to measure. COCO also
overlaps heavily with SUN RGB-D's indoor inventory (chair, couch, dining
table, tv, bed, toilet, sink, refrigerator, person, potted plant), so the
transfer is close to ideal here.

State this asymmetry explicitly when reporting results: the depth models
are trained on SUN RGB-D only, the detector is fine-tuned from COCO.

Usage:
    python -m wheelchair_nav.scripts.train_yolo_obstacle \
        --data ./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml --epochs 100

Hyperparameters found by scripts/tune_yolo_obstacle.py (Optuna, TPE
sampler) can be applied directly with --hparams_json, which passes the
tuned optimizer/loss/augmentation values straight through to Ultralytics'
train():

    python -m wheelchair_nav.scripts.train_yolo_obstacle \
        --data ./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml --epochs 100 \
        --hparams_json ./wheelchair_nav/log_yolo/optuna_best_yolo_hparams.json

To reproduce the from-scratch ablation instead, pass --pretrained "".
"""
from __future__ import annotations

import argparse
import json

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml")
    p.add_argument("--pretrained", default="yolo26n.pt",
                    help="Checkpoint to fine-tune from (default: COCO-pretrained yolo26n.pt). "
                         "Pass '' to train from random init (yolo26n.yaml) for the from-scratch "
                         "ablation -- expect substantially lower mAP at this dataset size.")
    p.add_argument("--epochs", type=int, default=100,
                    help="100 is enough when fine-tuning from COCO; raise to 300-500 if you set "
                         "--pretrained '' , since random init converges far more slowly")
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
    if args.pretrained:
        print(f"Initialising from pretrained checkpoint: {args.pretrained}")
        model = YOLO(args.pretrained)
    else:
        print("Initialising from random weights (yolo26n.yaml) -- from-scratch ablation")
        model = YOLO("yolo26n.yaml")

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
