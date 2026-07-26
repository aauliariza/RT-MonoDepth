"""Fuses YOLO26-nano bounding boxes with the RT-MonoDepth depth map into a
per-frame Obstacle List, following the perception diagram:

    Obstacle List: class * bbox * depth_m
    depth_m = median(depth_map[bbox_inner_ROI])

Objects are never classified by name -- every detection collapses into the
generic "obstacle" class, since the navigation logic only cares about where
something is and how far away it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from wheelchair_nav.config import BBOX_INNER_RATIO, MAX_DEPTH_M, MIN_DEPTH_M


@dataclass
class Obstacle:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords, clipped to frame
    depth_m: float                    # median depth over the bbox's inner ROI
    cx: float                         # bbox center x, used for sector assignment
    cy: float
    cls: str = field(default="obstacle")  # class-agnostic by design


def _inner_roi(x1: int, y1: int, x2: int, y2: int, ratio: float) -> Tuple[int, int, int, int]:
    """Shrinks a bbox toward its center so background pixels leaking in at
    the box edges don't bias the median depth estimate.
    """
    w, h = x2 - x1, y2 - y1
    mx, my = w * (1 - ratio) / 2, h * (1 - ratio) / 2
    ix1, iy1 = int(round(x1 + mx)), int(round(y1 + my))
    ix2, iy2 = int(round(x2 - mx)), int(round(y2 - my))
    if ix2 <= ix1:
        ix1, ix2 = x1, x2
    if iy2 <= iy1:
        iy1, iy2 = y1, y2
    return ix1, iy1, ix2, iy2


def build_obstacle_list(
    depth_map: np.ndarray,
    boxes: Sequence[Tuple[float, float, float, float, float]],
    inner_ratio: float = BBOX_INNER_RATIO,
) -> List[Obstacle]:
    """boxes: (x1, y1, x2, y2, confidence) tuples, e.g. from ObstacleDetector.detect().
    depth_map: (H, W) float32 metric depth in meters, same resolution as the
    frame the boxes were detected on.
    """
    h, w = depth_map.shape[:2]
    obstacles: List[Obstacle] = []

    for (x1, y1, x2, y2, _conf) in boxes:
        x1c, y1c = max(0, int(x1)), max(0, int(y1))
        x2c, y2c = min(w, int(x2)), min(h, int(y2))
        if x2c <= x1c or y2c <= y1c:
            continue

        ix1, iy1, ix2, iy2 = _inner_roi(x1c, y1c, x2c, y2c, inner_ratio)
        roi = depth_map[iy1:iy2, ix1:ix2]
        roi = roi[np.isfinite(roi)]
        if roi.size == 0:
            roi = depth_map[y1c:y2c, x1c:x2c]
            roi = roi[np.isfinite(roi)]
        if roi.size == 0:
            continue

        depth_m = float(np.clip(np.median(roi), MIN_DEPTH_M, MAX_DEPTH_M))
        cx, cy = (x1c + x2c) / 2.0, (y1c + y2c) / 2.0
        obstacles.append(Obstacle(bbox=(x1c, y1c, x2c, y2c), depth_m=depth_m, cx=cx, cy=cy))

    return obstacles
