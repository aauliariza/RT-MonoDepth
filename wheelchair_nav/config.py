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

# ---------------------------------------------------------------------------
# Input resolution shared by EVERY depth model in this project, so the
# apples-to-apples comparison in evaluation/eval_depth_comparison.py comes
# down to architecture rather than how much of the image each model saw.
#
# Why 288x384 rather than the repo's original 192x640: SUN RGB-D is 640x480
# (4:3), and 288x384 is the resolution that
#   - preserves that 4:3 aspect exactly -- a uniform 0.6x scale in BOTH axes,
#     so nothing is distorted. The repo's 192x640 default is inherited from
#     KITTI (~10:3), and applied to 4:3 indoor frames it SQUASHES them 2.5x
#     vertically, which handicapped every depth model here;
#   - stays divisible by 32, as the 5-stage encoders require;
#   - costs ~10% FEWER MACs than 192x640 (110,592 px vs 122,880), so fixing
#     the distortion is free.
#
# The Ghost-Depth paper independently uses 228x304, also exactly 4:3.
# ---------------------------------------------------------------------------
INPUT_HEIGHT = 288
INPUT_WIDTH = 384

# Ultralytics takes one square imgsz and letterboxes into it, preserving
# aspect ratio. At imgsz=384 a 640x480 frame lands on exactly 288x384 of
# real content (scale 0.6, verified against ultralytics.data.augment.LetterBox)
# with the remainder grey padding -- i.e. the SAME effective resolution the
# other depth models get from INPUT_HEIGHT x INPUT_WIDTH. The padding makes
# YOLO's canvas 384x384, so it pays ~1.33x the MACs of its own useful content;
# that is padding overhead, not extra image information.
YOLO_DEPTH_IMGSZ = 384

# ---------------------------------------------------------------------------
# Training budget shared by every depth model, for the same reason the input
# resolution is shared: a comparison across models trained for different
# numbers of epochs confounds architecture with budget.
#
# The per-model source recipes disagree (RT-MonoDepth/FastDepth 40 epochs,
# Ghost-Depth 55 per its paper sec. 4.2, YOLO26-depth 60), so this picks one
# budget generous enough that all of them have plateaued well before the end
# -- verified on the 100-epoch runs, where every model's validation curve is
# flat over the final third.
#
# The LR step size is NOT fixed here: scaling the budget without scaling the
# schedule wastes it. StepLR(gamma=0.1) at the old step=25 spends the last 25
# of 100 epochs at lr=1e-7, i.e. frozen. The trainers therefore default
# scheduler_step_size to num_epochs // 3, which keeps two meaningful decays
# regardless of the budget.
# ---------------------------------------------------------------------------
TRAIN_EPOCHS = 100

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
