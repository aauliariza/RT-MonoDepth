"""Detection evaluation for the fine-tuned YOLO26-nano obstacle detector:
mAP50 / mAP50-95 / precision / recall (via Ultralytics' own validator),
parameter count, and inference FPS.

Usage:
    python -m wheelchair_nav.evaluation.eval_detection_metrics \
        --weights ./log_yolo/obstacle_yolo26n/weights/best.pt \
        --data ./data/sunrgbd_yolo/obstacle.yaml
"""
from __future__ import annotations

import argparse
import time

import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="./data/sunrgbd_yolo/obstacle.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--fps_cycles", type=int, default=200)
    p.add_argument("--fps_warmup", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print("\nDetection accuracy:")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  precision: {metrics.box.mp:.4f}")
    print(f"  recall:    {metrics.box.mr:.4f}")

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"\nParameters: {n_params / 1e6:.3f} M ({n_params} total)")

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    for _ in range(args.fps_warmup):
        model.predict(dummy, device=args.device, verbose=False)
    t0 = time.time()
    for _ in range(args.fps_cycles):
        model.predict(dummy, device=args.device, verbose=False)
    dt = time.time() - t0
    print(f"Inference FPS ({args.device}, {args.imgsz}x{args.imgsz}): {args.fps_cycles / dt:.1f}")


if __name__ == "__main__":
    main()
