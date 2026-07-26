"""Comparison-only depth baselines trained on the SAME SUN RGB-D
train/val/test split as RT-MonoDepth (splits_sunrgbd/, see
scripts/prepare_sunrgbd.py), for an apples-to-apples comparison against
this project's chosen depth model. Nothing here is used by the actual
navigation pipeline in wheelchair_nav/perception/ or run_navigation.py --
it exists purely to produce evaluation/eval_depth_comparison.py's table.

Models compared:
  - FastDepth  (networks/FastDepth/model.py, unmodified, MobileNet
    encoder + skip-add decoder)
  - YOLO26n-depth / YOLO26s-depth (Ultralytics' native monocular depth
    task, unmodified architecture)
  - RT-MonoDepth full (wheelchair_nav/perception/depth_estimator.py)
"""
