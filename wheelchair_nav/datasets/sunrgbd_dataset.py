"""PyTorch Dataset over preprocessed SUN RGB-D RGB / metric-depth pairs
(see wheelchair_nav/scripts/prepare_sunrgbd.py), used to train RT-MonoDepth
(full model, unmodified architecture) from scratch with supervised depth
regression instead of the repo's original KITTI self-supervised
photometric-reprojection training (trainer.py) -- SUN RGB-D is a
collection of single RGB-D snapshots with sensor-measured ground-truth
depth rather than posed video sequences, so direct supervision is the
appropriate objective here.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from wheelchair_nav.config import INPUT_HEIGHT, INPUT_WIDTH, MAX_DEPTH_M, MIN_DEPTH_M


def _read_split(list_path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rgb, depth = line.split()
            pairs.append((rgb, depth))
    return pairs


class SUNRGBDDepthDataset(Dataset):
    """Each depth file is a float32 .npy holding metric depth in meters
    (written by prepare_sunrgbd.py); 0 marks invalid/missing depth.
    """

    def __init__(
        self,
        list_path: str,
        height: int = INPUT_HEIGHT,
        width: int = INPUT_WIDTH,
        min_depth: float = MIN_DEPTH_M,
        max_depth: float = MAX_DEPTH_M,
        is_train: bool = True,
    ):
        self.pairs = _read_split(list_path)
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.is_train = is_train

        # Same brightness/contrast/saturation/hue ranges trainer.py's
        # color_aug uses for KITTI -- only the depth objective changed
        # (see module docstring), photometric augmentation is unaffected
        # by that and is just as useful here for robustness to the
        # wheelchair camera's real-world lighting variation.
        self.color_jitter = transforms.ColorJitter(
            brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1),
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        rgb_path, depth_path = self.pairs[index]

        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(rgb_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        depth = np.load(depth_path).astype(np.float32)
        depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        if self.is_train and np.random.rand() > 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            depth = np.ascontiguousarray(depth[:, ::-1])

        if self.is_train:
            img = np.array(self.color_jitter(Image.fromarray(img)))

        valid = (depth > self.min_depth) & (depth < self.max_depth)
        depth_clipped = np.clip(depth, self.min_depth, self.max_depth)

        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(np.ascontiguousarray(depth_clipped)).unsqueeze(0).float()
        valid_t = torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0).float()

        return {"color": img_t, "depth_gt": depth_t, "valid_mask": valid_t}
