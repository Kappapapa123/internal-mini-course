"""Dice (DSC) scoring helpers for integer label-map segmentations.

Teaching copy of the lab's scoring helper. Dice for a label is
``2 |A ∩ B| / (|A| + |B|)``. When a label is absent from *both* prediction and
reference the score is ``NaN`` -- :func:`nanmean_dice` then drops it, which is a
*policy choice* (ignore classes that aren't present) rather than a detail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np


def dice_for_label(prediction: np.ndarray, reference: np.ndarray, label: int) -> float:
    """Dice for one integer label; NaN when the label is in neither array."""
    pred_mask = prediction == label
    ref_mask = reference == label
    denom = int(pred_mask.sum() + ref_mask.sum())
    if denom == 0:
        return float("nan")
    return float(2 * np.logical_and(pred_mask, ref_mask).sum() / denom)


def dice_per_label(
    prediction: np.ndarray,
    reference: np.ndarray,
    labels: list[int] | tuple[int, ...],
) -> dict[int, float]:
    """Dice for each requested integer label."""
    if prediction.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: prediction {prediction.shape}, reference {reference.shape}"
        )
    return {
        int(label): dice_for_label(prediction, reference, int(label)) for label in labels
    }


def nanmean_dice(scores: dict[int, float], labels: list[int] | tuple[int, ...]) -> float:
    """Average selected Dice scores with ``np.nanmean`` (drops absent-class NaNs)."""
    if not labels:
        raise ValueError("labels must not be empty")
    values = np.asarray([scores[int(label)] for label in labels], dtype=float)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def load_nifti_labels(path: Path) -> np.ndarray:
    """Load a NIfTI label map as an integer NumPy array (lazy import of nibabel)."""
    import nibabel as nib

    image = cast(Any, nib.load(str(path)))
    data = np.asanyarray(image.dataobj)
    return np.asarray(np.rint(data).astype(np.int64, copy=False))
