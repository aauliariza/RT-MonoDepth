"""Thin inference wrapper around the UNMODIFIED RT-MonoDepth (full) network
(networks/RTMonoDepth/RTMonoDepth.py: DepthEncoder + DepthDecoder). This
file adds no new layers and does not touch the architecture -- it only
loads a checkpoint trained from scratch on SUN RGB-D
(wheelchair_nav/scripts/train_depth_sunrgbd.py) and converts the network's
sigmoid disparity output into a metric depth map in meters, reusing the
repo's own layers.disp_to_depth() with an indoor min/max range.
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

from layers import disp_to_depth  # noqa: E402  (repo root, unmodified)
from networks.RTMonoDepth.RTMonoDepth import DepthDecoder, DepthEncoder  # noqa: E402  (repo root, unmodified)

from wheelchair_nav.config import MAX_DEPTH_M, MIN_DEPTH_M  # noqa: E402


def _resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LANCZOS4)


class DepthEstimator:
    """Loads RT-MonoDepth (full) encoder/decoder checkpoints and produces a
    metric depth map in meters for a single BGR frame.
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

        encoder_path = os.path.join(weights_dir, "encoder.pth")
        decoder_path = os.path.join(weights_dir, "depth.pth")

        self.encoder = DepthEncoder()
        enc_ckpt = torch.load(encoder_path, map_location=self.device)
        self.feed_height = int(enc_ckpt["height"])
        self.feed_width = int(enc_ckpt["width"])
        enc_state = {k: v for k, v in enc_ckpt.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(enc_state)
        self.encoder.to(self.device).eval()

        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
        self.decoder.to(self.device).eval()

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

        features = self.encoder(tensor)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        _, depth = disp_to_depth(disp, self.min_depth_m, self.max_depth_m)

        depth = F.interpolate(depth, (h0, w0), mode="bilinear", align_corners=False)
        return depth.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters()) + \
            sum(p.numel() for p in self.decoder.parameters())
