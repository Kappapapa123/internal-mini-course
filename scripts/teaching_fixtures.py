"""Deterministic teaching fixtures for the CPU-only notebooks.

Real nnU-Net inference needs a GPU, so the inference / post-processing / evaluation
notebooks cannot produce a genuine model prediction on a laptop. This builds a clearly
labelled *illustrative* prediction from the ground truth so those notebooks have something
to visualize and score. It is NOT a model output.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def illustrative_prediction(gt: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return a fake 'prediction' for label 1: GT eroded by one voxel, plus a couple of
    small spurious blobs (false positives) so the post-processing demo has something to
    remove. Deterministic given ``seed``. Illustrative only."""
    pred = ndimage.binary_erosion(gt == 1, iterations=1).astype(int)
    rng = np.random.default_rng(seed)
    for _ in range(2):
        x = int(rng.integers(5, gt.shape[0] - 5))
        y = int(rng.integers(5, gt.shape[1] - 5))
        z = int(rng.integers(3, gt.shape[2] - 3))
        pred[x : x + 2, y : y + 2, z : z + 1] = 1
    return pred
