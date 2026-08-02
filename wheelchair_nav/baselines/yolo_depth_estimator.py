"""Thin inference wrapper around the UNMODIFIED Ultralytics YOLO26-depth
architecture (yolo26n-depth / yolo26s-depth), used only as an
apples-to-apples comparison baseline against RT-MonoDepth -- see
scripts/train_yolo_depth.py and evaluation/eval_depth_comparison.py.

Training, the SILog + gradient loss, and post-training metric-scale
calibration are entirely native Ultralytics
(ultralytics/models/yolo/depth/: DepthTrainer, DepthValidator, and the
Depth head's built-in exp() + log-affine calibration). This file only
wraps inference into the same interface (.infer(frame_bgr) -> metric depth
in meters) used by DepthEstimator (RT-MonoDepth) and FastDepthEstimator,
so all three can be driven identically from
evaluation/eval_depth_comparison.py.
"""
from __future__ import annotations

import numpy as np


class YoloDepthEstimator:
    def __init__(self, weights: str, device: str = "cuda", imgsz: int = 640):
        from ultralytics import YOLO  # lazy import, mirrors perception/obstacle_detector.py

        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """frame_bgr: HxWx3 uint8, as read by cv2.VideoCapture. Returns a
        float32 (H, W) metric depth map in meters, already rescaled by
        Ultralytics' own DepthPredictor to the input frame's resolution.
        """
        results = self.model.predict(source=frame_bgr, device=self.device, imgsz=self.imgsz, verbose=False)
        depth = results[0].depth.data
        return depth.detach().cpu().numpy().astype(np.float32)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.model.parameters())
