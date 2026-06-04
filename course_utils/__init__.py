"""Self-contained, CPU-runnable teaching code for the nnU-Net fine-tuning course.

These are *simplified teaching reimplementations* of the schedulers and Dice
helpers used by the lab's nnU-Net trainers — kept here so the course runs on a
laptop with no nnU-Net install, no cluster, and no private repo. See NOTICE.md.
"""

from course_utils.dsc import dice_for_label, dice_per_label, nanmean_dice
from course_utils.lr_schedulers import (
    Lin_incr_LRScheduler,
    PolyLRScheduler_offset,
    two_phase_lr,
)

__all__ = [
    "Lin_incr_LRScheduler",
    "PolyLRScheduler_offset",
    "two_phase_lr",
    "dice_for_label",
    "dice_per_label",
    "nanmean_dice",
]
