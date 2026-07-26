"""Sector-Based Free-Path Selection, matching the decision diagram:

    Priority:  CTR > L > R > FL > FR > STOP
    Hysteresis: N=3 majority-vote window
    Decision:  FORWARD | TURN_LEFT | TURN_RIGHT | STOP
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Dict, Optional

from wheelchair_nav.config import (
    EMERGENCY_DISTANCE_M,
    HYSTERESIS_WINDOW,
    SAFE_DISTANCE_M,
    SECTOR_PRIORITY,
    SECTOR_TO_DECISION,
)


def select_raw_decision(sector_depths: Dict[str, float], safe_distance_m: float = SAFE_DISTANCE_M) -> str:
    """Picks the highest-priority free sector (CTR > L > R > FL > FR) or
    STOP if every sector has something closer than the safety threshold.
    """
    for sector in SECTOR_PRIORITY:
        if sector_depths.get(sector, 0.0) >= safe_distance_m:
            return SECTOR_TO_DECISION[sector]
    return "STOP"


class FreePathSelector:
    """Smooths the raw per-frame decision with an N-frame majority vote so
    a single flickering detection (e.g. someone briefly crossing a sector)
    doesn't cause the wheelchair to zig-zag. An imminent obstacle
    (< emergency_distance_m anywhere) always stops the chair immediately,
    bypassing the hysteresis window entirely -- safety is never delayed for
    smoothness.
    """

    def __init__(
        self,
        window: int = HYSTERESIS_WINDOW,
        safe_distance_m: float = SAFE_DISTANCE_M,
        emergency_distance_m: Optional[float] = None,
    ):
        self.window = window
        self.safe_distance_m = safe_distance_m
        self.emergency_distance_m = (
            EMERGENCY_DISTANCE_M if emergency_distance_m is None else emergency_distance_m
        )
        self._history: Deque[str] = deque(maxlen=window)
        self._last_decision = "STOP"

    def update(self, sector_depths: Dict[str, float]) -> str:
        min_depth = min(sector_depths.values()) if sector_depths else float("inf")
        if min_depth < self.emergency_distance_m:
            self._history.clear()
            self._last_decision = "STOP"
            return "STOP"

        raw = select_raw_decision(sector_depths, self.safe_distance_m)
        self._history.append(raw)

        counts = Counter(self._history)
        top_count = max(counts.values())
        candidates = {d for d, c in counts.items() if c == top_count}

        if self._last_decision in candidates:
            decision = self._last_decision
        else:
            decision = next(
                (SECTOR_TO_DECISION[s] for s in SECTOR_PRIORITY if SECTOR_TO_DECISION[s] in candidates),
                next(iter(candidates)),
            )

        self._last_decision = decision
        return decision

    def reset(self) -> None:
        self._history.clear()
        self._last_decision = "STOP"
