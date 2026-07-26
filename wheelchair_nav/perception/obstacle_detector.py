"""YOLO26-nano wrapper used purely as a class-agnostic bounding-box
proposer. RT-MonoDepth needs to know *where* an obstacle is, not what it
is, so every detected class is collapsed into a single generic "obstacle"
downstream -- the class id/name returned by YOLO is discarded entirely.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


class ObstacleDetector:
    def __init__(
        self,
        weights: str = "yolo26n.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "cuda",
        imgsz: int = 640,
        min_box_area: int = 400,
    ):
        from ultralytics import YOLO  # lazy import: the rest of the package
        # works fine without ultralytics installed (e.g. depth-only usage).

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.min_box_area = min_box_area

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Returns a list of (x1, y1, x2, y2, confidence) boxes, class
        identity intentionally discarded.
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        boxes: List[Tuple[float, float, float, float, float]] = []
        if not results:
            return boxes

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                continue
            boxes.append((x1, y1, x2, y2, conf))
        return boxes

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.model.parameters())
