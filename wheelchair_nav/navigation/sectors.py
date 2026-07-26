"""Splits the camera field of view into the 5 horizontal sectors used by
Sector-Based Free-Path Selection (FL0, L1, CTR2, R3, FR4, left to right)
and aggregates obstacle depth per sector.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from wheelchair_nav.config import SECTOR_NAMES


def assign_sector(cx: float, image_width: int, num_sectors: int = len(SECTOR_NAMES)) -> str:
    """Maps a bbox center x-coordinate to one of the 5 equal-width vertical
    strips of the image, ordered left to right: FL0, L1, CTR2, R3, FR4.
    """
    idx = int(cx / max(image_width, 1) * num_sectors)
    idx = min(max(idx, 0), num_sectors - 1)
    return SECTOR_NAMES[idx]


def compute_sector_depths(
    obstacles: Sequence,
    image_width: int,
    center_depth_override: Optional[float] = None,
) -> Dict[str, float]:
    """obstacles: any sequence of objects exposing `.cx` and `.depth_m`
    (e.g. wheelchair_nav.perception.obstacle_list.Obstacle).

    Each sector's depth is the *minimum* depth among the obstacles that
    fall in it (closest obstacle wins), so a sector only counts as free
    once nothing inside it is nearer than the safety threshold. Sectors
    with no detected obstacle default to +inf (free).

    center_depth_override lets the caller fold in a depth-map-only check
    (e.g. the raw minimum depth in the central band) as a second, detector
    independent safety signal for the CTR2 sector.
    """
    sector_depths: Dict[str, float] = {name: float("inf") for name in SECTOR_NAMES}

    for obs in obstacles:
        sector = assign_sector(obs.cx, image_width)
        if obs.depth_m < sector_depths[sector]:
            sector_depths[sector] = obs.depth_m

    if center_depth_override is not None:
        sector_depths["CTR2"] = min(sector_depths["CTR2"], center_depth_override)

    return sector_depths
