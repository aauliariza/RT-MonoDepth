"""Shared train/val loss curve logging for the custom PyTorch training loops
(train_depth_sunrgbd.py, train_fastdepth_sunrgbd.py) -- lets overfitting
(val loss rising while train loss keeps falling) or underfitting (both
staying high) be read off a plot instead of scrollback logs.
"""
from __future__ import annotations

import csv
import os


def save_training_curve(
    log_dir: str,
    epochs: list,
    train_losses: list,
    val_losses: list,
    train_label: str = "train_loss",
    val_label: str = "val_loss",
    title: str = "Training curve",
) -> None:
    """Overwrites training_log.csv and training_curve.png under log_dir from
    the full epoch history given. Called once per epoch so both files stay
    current if training is interrupted. train_loss and val_loss are plotted
    on separate y-axes since here they are different loss compositions/units,
    not directly comparable in magnitude -- only their trends matter.
    """
    os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", train_label, val_label])
        writer.writerows(zip(epochs, train_losses, val_losses))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_losses, color="tab:blue", marker=".", label=train_label)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel(train_label, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_losses, color="tab:orange", marker=".", label=val_label)
    ax2.set_ylabel(val_label, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "training_curve.png"), dpi=150)
    plt.close(fig)
