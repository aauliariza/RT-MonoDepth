"""Autonomous indoor navigation system for a smart wheelchair, built on top
of the unmodified RT-MonoDepth (full) architecture from this repository and
YOLO26-nano for class-agnostic obstacle bounding boxes.

Nothing under networks/ is changed by this package -- it only adds the
training/inference/navigation/evaluation code needed to go from RGB video
to a FORWARD / TURN_LEFT / TURN_RIGHT / STOP decision.
"""
