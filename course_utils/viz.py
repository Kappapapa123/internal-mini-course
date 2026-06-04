"""Small matplotlib helpers for the course notebooks (CPU-only, no GUI backend).

Two things the notebooks need repeatedly:
* overlay a segmentation mask on a CT slice, and
* plot the two-phase LR curve with the warm-up -> decay switch marked.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from course_utils.lr_schedulers import two_phase_lr


def overlay_mask_on_slice(
    ct_volume: np.ndarray,
    mask_volume: np.ndarray,
    z: int | None = None,
    ax=None,
    alpha: float = 0.4,
):
    """Show one axial slice of ``ct_volume`` with ``mask_volume`` overlaid.

    Volumes are indexed ``[x, y, z]``. If ``z`` is None the middle slice is used.
    Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if ct_volume.shape != mask_volume.shape:
        raise ValueError(
            f"shape mismatch: ct {ct_volume.shape}, mask {mask_volume.shape}"
        )
    if z is None:
        z = ct_volume.shape[2] // 2
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(ct_volume[:, :, z].T, cmap="gray", origin="lower")
    masked = np.ma.masked_where(mask_volume[:, :, z].T == 0, mask_volume[:, :, z].T)
    ax.imshow(masked, cmap="autumn", alpha=alpha, origin="lower")
    ax.set_title(f"axial slice z={z}")
    ax.axis("off")
    return ax


def plot_lr_curve(
    *,
    max_lr: float = 1e-3,
    warmup_epochs: int = 50,
    total_epochs: int = 1000,
    exponent: float = 0.9,
    ax=None,
):
    """Plot the two-phase LR schedule and mark the warm-up -> decay switch.

    Returns ``(ax, lrs)`` where ``lrs`` is the per-epoch LR array.
    """
    import matplotlib.pyplot as plt

    epochs = np.arange(total_epochs)
    lrs = np.array(
        [
            two_phase_lr(
                int(e),
                max_lr=max_lr,
                warmup_epochs=warmup_epochs,
                total_epochs=total_epochs,
                exponent=exponent,
            )
            for e in epochs
        ]
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(epochs, lrs, lw=2)
    ax.axvline(warmup_epochs, color="crimson", ls="--", lw=1)
    ax.annotate(
        f"switch @ epoch {warmup_epochs}\n(peak {max_lr:g})",
        xy=(warmup_epochs, max_lr),
        xytext=(warmup_epochs + total_epochs * 0.08, max_lr * 0.7),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="crimson"),
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("Two-phase fine-tuning LR: warm-up -> poly decay")
    return ax, lrs


def lr_values(epochs: Sequence[int], **kwargs) -> np.ndarray:
    """Convenience: LR values for an iterable of epochs (passes kwargs through)."""
    return np.array([two_phase_lr(int(e), **kwargs) for e in epochs])
