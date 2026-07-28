"""Maps a smoothed navigation decision to a simulated wheelchair drive
command. There is no physical wheelchair in this repo's test setup (video
file testing only), so this is a logging/visualization-facing stand-in for
whatever motor-driver interface a real chair would expose.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriveCommand:
    linear_mps: float
    angular_radps: float
    decision: str


class WheelchairController:
    def __init__(self, cruise_speed_mps: float = 0.6, turn_speed_radps: float = 0.5):
        self.cruise_speed_mps = cruise_speed_mps
        self.turn_speed_radps = turn_speed_radps

    def step(self, decision: str) -> DriveCommand:
        if decision == "FORWARD":
            return DriveCommand(self.cruise_speed_mps, 0.0, decision)
        if decision == "TURN_LEFT":
            return DriveCommand(self.cruise_speed_mps * 0.4, self.turn_speed_radps, decision)
        if decision == "TURN_RIGHT":
            return DriveCommand(self.cruise_speed_mps * 0.4, -self.turn_speed_radps, decision)
        if decision == "TURN_FAR_LEFT":
            # Sharper evasive turn (obstacle forces a wider detour): slower
            # forward speed, stronger angular rate than a plain TURN_LEFT.
            return DriveCommand(self.cruise_speed_mps * 0.2, self.turn_speed_radps * 1.5, decision)
        if decision == "TURN_FAR_RIGHT":
            return DriveCommand(self.cruise_speed_mps * 0.2, -self.turn_speed_radps * 1.5, decision)
        return DriveCommand(0.0, 0.0, "STOP")
