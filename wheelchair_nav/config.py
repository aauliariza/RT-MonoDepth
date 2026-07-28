"""Central configuration for the wheelchair navigation system built on top
of the (unmodified) RT-MonoDepth full model + YOLO26-nano.
"""

# ---------------------------------------------------------------------------
# Safety thresholds
# ---------------------------------------------------------------------------
SAFE_DISTANCE_M = 1.0        # a sector is only "free" if nothing in it is closer than this
EMERGENCY_DISTANCE_M = 0.5   # bypasses the hysteresis window and stops immediately

# ---------------------------------------------------------------------------
# Depth range RT-MonoDepth is trained / interpreted over (indoor, metres).
# Reuses the repo's own layers.disp_to_depth(disp, min_depth, max_depth)
# formula, just with an indoor-appropriate range instead of KITTI's 0.1-100m.
# ---------------------------------------------------------------------------
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 10.0

# RT-MonoDepth input resolution (matches options.py defaults for the repo)
INPUT_HEIGHT = 192
INPUT_WIDTH = 640

# ---------------------------------------------------------------------------
# Obstacle List (perception fusion): depth_m = median(depth_map[bbox_inner_ROI])
# ---------------------------------------------------------------------------
BBOX_INNER_RATIO = 0.6  # shrink each bbox to its central 60% before taking the median depth

# ---------------------------------------------------------------------------
# Sector-Based Free-Path Selection (left -> right across the image)
# Each of the 5 sectors maps to its own distinct decision (bijective), so the
# chosen sector can always be recovered from the final decision alone (see
# navigation/free_path.py's DECISION_TO_SECTOR).
# ---------------------------------------------------------------------------
SECTOR_NAMES = ["FL0", "L1", "CTR2", "R3", "FR4"]
SECTOR_PRIORITY = ["CTR2", "L1", "R3", "FL0", "FR4"]
SECTOR_TO_DECISION = {
    "FL0": "TURN_FAR_LEFT",
    "L1": "TURN_LEFT",
    "CTR2": "FORWARD",
    "R3": "TURN_RIGHT",
    "FR4": "TURN_FAR_RIGHT",
}
DECISIONS = ["FORWARD", "TURN_LEFT", "TURN_RIGHT", "TURN_FAR_LEFT", "TURN_FAR_RIGHT", "STOP"]
HYSTERESIS_WINDOW = 3  # N=3 majority-vote window
