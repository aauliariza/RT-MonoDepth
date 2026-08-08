"""Thin inference wrapper around Ghost-Depth
(networks/GhostDepth/ghost_depth.py, reimplemented from Quan et al.,
CNIOT '23 -- see that module's docstring for the reproduction notes).

Used as a comparison baseline alongside RT-MonoDepth and FastDepth --
see scripts/train_ghostdepth_sunrgbd.py and
evaluation/eval_depth_comparison.py.

Ghost-Depth's depth head is a plain 3x3 convolution with no output
activation, and it predicts at HALF the fed resolution (the paper trains
304x228 inputs against 152x114 depth maps). Both are handled here: the
raw output is clamped into the project's indoor metric range -- exactly
as done at training time -- and then upsampled to the caller's frame
size, so this class exposes the same interface as DepthEstimator and
FastDepthEstimator.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from networks.GhostDepth.ghost_depth import GhostDepth  # noqa: E402

from wheelchair_nav.config import MAX_DEPTH_M, MIN_DEPTH_M  # noqa: E402


def _resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LANCZOS4)


class GhostDepthEstimator:
    """Loads a Ghost-Depth checkpoint saved by train_ghostdepth_sunrgbd.py
    and produces a metric depth map in meters for a single BGR frame.
    """

    def __init__(
        self,
        weights_dir: str,
        device: str = "cuda",
        min_depth_m: float = MIN_DEPTH_M,
        max_depth_m: float = MAX_DEPTH_M,
    ):
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

        weights_path = os.path.join(weights_dir, "ghostdepth.pth")
        ckpt = torch.load(weights_path, map_location=self.device)
        self.feed_height = int(ckpt["height"])
        self.feed_width = int(ckpt["width"])

        # Architecture switches are stored with the weights so a checkpoint
        # always rebuilds the model it was trained as.
        self.model = GhostDepth(
            width=float(ckpt.get("width_mult", 1.0)),
            iaff_channels=int(ckpt.get("iaff_channels", 40)),
            keep_final_conv=bool(ckpt.get("keep_final_conv", True)),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """frame_bgr: HxWx3 uint8, as read by cv2.VideoCapture. Returns a
        float32 (H, W) metric depth map in meters, resized to match the
        input frame's resolution.
        """
        h0, w0 = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        resized = _resize_rgb(rgb, self.feed_height, self.feed_width)
        tensor = torch.from_numpy(np.ascontiguousarray(resized))
        tensor = tensor.permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)

        raw = self.model(tensor)
        depth = raw.clamp(self.min_depth_m, self.max_depth_m)
        depth = F.interpolate(depth, (h0, w0), mode="bilinear", align_corners=False)
        return depth.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
