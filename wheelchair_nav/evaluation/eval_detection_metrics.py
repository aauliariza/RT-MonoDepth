"""Detection evaluation for the fine-tuned YOLO26-nano obstacle detector:
mAP50 / mAP50-95 / precision / recall (via Ultralytics' own validator),
parameter count, and the full per-inference latency distribution
(mean/std/min/p50/p90/p95/p99/max, plus FPS derived from the mean).

Recall deserves more weight than precision here: an obstacle the detector
misses never enters the obstacle list at all, so no depth model can
recover it, whereas a false positive costs at most an unnecessary stop.

Usage:
    python -m wheelchair_nav.evaluation.eval_detection_metrics \
        --weights ./wheelchair_nav/log_yolo/obstacle_yolo26n/weights/best.pt \
        --data ./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml
"""
from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO

from wheelchair_nav.evaluation.latency import format_latency_table, measure_latency


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--latency_cycles", "--fps_cycles", dest="fps_cycles", type=int, default=200)
    p.add_argument("--latency_warmup", "--fps_warmup", dest="fps_warmup", type=int, default=20)
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
    lat = measure_latency(
        lambda: model.predict(dummy, device=args.device, verbose=False),
        warmup=args.fps_warmup,
        cycles=args.fps_cycles,
        device=next(model.model.parameters()).device,
    )
    lat["model"] = f"YOLO26-nano @ {args.imgsz}x{args.imgsz}"
    print()
    print(format_latency_table([lat], title=f"Per-inference latency (device={args.device}):"))


if __name__ == "__main__":
    main()
