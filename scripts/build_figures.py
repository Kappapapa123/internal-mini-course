"""Build the two deck figures into ``assets/figures/``.

Source of truth for the slide visuals (same philosophy as ``build_notebooks.py``).
Run with::

    uv run python scripts/build_figures.py

Produces:

* ``lr_schedule.png`` -- the two-phase fine-tuning LR curve, computed from the
  real :func:`course_utils.lr_schedulers.two_phase_lr` (a genuine artifact, not a
  hand-drawn approximation).
* ``architecture_transfer.png`` -- a ResEnc-L U-Net schematic split into the three
  regions the talk hinges on: encoder + decoder body (transferred) and all
  deep-supervision ``seg_layers`` (re-initialized).
* ``ct_slice.png`` / ``ct_overlay.png`` -- one axial slice of the real synthetic
  phantom (the committed ``Dataset999_Phantom`` case), shown plain and with its
  lesion label overlaid in the deck accent colour. Used as the clinical-task and
  look-at-your-data visuals on the slides (a genuine artifact, not a stock image).

Both follow Tufte's rules: high data-ink ratio, a single accent colour reserved
for the focal element, direct labels rather than framed legends, range frames,
and the smallest effective visual difference. Labels avoid em-dashes / AI vocab.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from course_utils.lr_schedulers import two_phase_lr

FIG_DIR = Path(__file__).resolve().parent.parent / "assets" / "figures"

# Quiet palette. The data/structure carries the ink; one warm accent (orange /
# crimson) is reserved for the single focal element in each figure.
C_DATA = "#1f2933"     # the LR curve itself (near-black)
C_SWITCH = "#c0392b"   # focal accent: the epoch-50 switch
C_TEXT = "#1f2933"
C_MUTE = "#6b7280"     # axis / secondary text

C_ENC = "#9aa9bd"      # encoder: light slate (transferred)
C_DEC = "#5d7290"      # decoder body + bottleneck: dark slate (transferred)
C_SEG = "#e07b39"      # seg_layers: the one accent (re-initialized)
C_FLOW = "#9aa3b0"     # data-flow arrows (subtle, uniform)
C_SKIP = "#cdd2da"     # skip connections (faintest layer)


def build_lr_schedule(
    *,
    max_lr: float = 1e-3,
    warmup_epochs: int = 50,
    total_epochs: int = 1000,
) -> Path:
    """Plot the real two-phase LR schedule and save it for the deck."""
    epochs = np.arange(total_epochs)
    lrs = np.array(
        [two_phase_lr(int(e), max_lr=max_lr, warmup_epochs=warmup_epochs,
                      total_epochs=total_epochs) for e in epochs]
    )

    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=200)
    fig.patch.set_facecolor("white")

    # the data
    ax.plot(epochs, lrs, color=C_DATA, lw=2.3, solid_capstyle="round")

    # focal element: the epoch-50 switch, the only accent colour
    ax.plot([warmup_epochs, warmup_epochs], [0, max_lr],
            color=C_SWITCH, ls=(0, (3, 3)), lw=1.1)
    ax.scatter([warmup_epochs], [max_lr], color=C_SWITCH, s=26, zorder=5)

    # direct labels in place of a legend
    ax.text(warmup_epochs + 22, max_lr, "peak 0.001 at epoch 50",
            color=C_SWITCH, fontsize=10.5, va="center", ha="left")
    ax.text(560, 6.4e-4, "Phase 2: polynomial decay",
            color=C_TEXT, fontsize=11, va="bottom", ha="center")
    ax.annotate("Phase 1: linear warm-up\nstarts at 2e-5, not 0",
                xy=(0, lrs[0]), xytext=(95, 2.7e-4),
                fontsize=10.5, color=C_TEXT, va="center",
                arrowprops=dict(arrowstyle="->", color=C_MUTE, lw=1))

    # range frame: spines span only where the data lives
    ax.set_xlim(-12, 1012)
    ax.set_ylim(-2.5e-5, 1.05e-3)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_yticks([0, 5e-4, 1e-3])
    ax.set_yticklabels(["0", "0.0005", "0.0010"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_bounds(0, max_lr)
    ax.spines["bottom"].set_bounds(0, total_epochs)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(C_MUTE)
    ax.tick_params(colors=C_MUTE, labelsize=9.5)
    ax.set_xlabel("epoch", fontsize=11, color=C_TEXT)
    ax.set_ylabel("learning rate", fontsize=11, color=C_TEXT)
    ax.set_title("Two-phase fine-tuning learning-rate schedule",
                 loc="left", fontsize=13, color=C_TEXT, pad=10)

    out = FIG_DIR / "lr_schedule.png"
    fig.tight_layout()
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def _block(ax, x, y, w, h, color, label, *, text_color="white", fs=10):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0, facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", color=text_color,
            fontsize=fs, fontweight="bold")


def _arrow(ax, p0, p1, color, *, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11,
        color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=2,
    ))


def build_architecture() -> Path:
    """ResEnc-L U-Net transfer schematic. Two quiet slate tints mark the
    transferred encoder / decoder body; the single warm accent marks the
    re-initialized deep-supervision seg_layers."""
    fig, ax = plt.subplots(figsize=(11.0, 6.6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ys = [7.0, 5.9, 4.8, 3.7]          # levels 0..3
    y_bottle = 2.4
    x_enc, x_dec = 2.6, 7.4
    x_bottle = 5.0
    bw, bh = 1.7, 0.62

    enc_labels = ["stage 0", "stage 1", "stage 2", "stage 3"]
    dec_labels = ["dec 0", "dec 1", "dec 2", "dec 3"]

    enc_pts, dec_pts = [], []
    for y, lab in zip(ys, enc_labels):
        _block(ax, x_enc, y, bw, bh, C_ENC, lab, text_color=C_TEXT)
        enc_pts.append((x_enc, y))
    for y, lab in zip(ys, dec_labels):
        _block(ax, x_dec, y, bw, bh, C_DEC, lab)
        dec_pts.append((x_dec, y))
    _block(ax, x_bottle, y_bottle, 2.3, bh, C_DEC, "bottleneck")

    # data flow: one subtle uniform grey, down the encoder and up the decoder
    for i in range(len(ys) - 1):
        _arrow(ax, (x_enc, ys[i] - bh / 2), (x_enc, ys[i + 1] + bh / 2), C_FLOW)
    _arrow(ax, (x_enc, ys[-1] - bh / 2), (x_bottle - 1.0, y_bottle), C_FLOW)
    _arrow(ax, (x_bottle + 1.0, y_bottle), (x_dec, ys[-1] - bh / 2), C_FLOW)
    for i in range(len(ys) - 1, 0, -1):
        _arrow(ax, (x_dec, ys[i] + bh / 2), (x_dec, ys[i - 1] - bh / 2), C_FLOW)
    # skip connections: faintest layer
    for (xe, ye), (xd, yd) in zip(enc_pts, dec_pts):
        _arrow(ax, (xe + bw / 2, ye), (xd - bw / 2, yd),
               C_SKIP, lw=1.1, ls=(0, (4, 3)))
    ax.text((x_enc + x_dec) / 2, ys[0] + 0.55, "skip connections",
            ha="center", color=C_MUTE, fontsize=9)

    # deep-supervision seg_layers: heads off the top three decoder scales
    seg_x = x_dec + 1.9
    for y in ys[:3]:
        _block(ax, seg_x, y, 1.15, 0.5, C_SEG, "seg", fs=9)
        _arrow(ax, (x_dec + bw / 2, y), (seg_x - 1.15 / 2, y), C_SEG, lw=1.2)
        _arrow(ax, (seg_x + 1.15 / 2, y), (seg_x + 1.15, y), C_SEG, lw=1.2)
    ax.text(seg_x + 0.05, y_bottle + 0.35,
            "deep supervision:\nseg_layers at several scales",
            ha="center", va="center", color=C_SEG, fontsize=9.5)

    # direct key (unframed): two slate tints = transferred, accent = re-init
    ky1, ky2 = 1.45, 0.95
    ax.scatter([0.55, 0.83], [ky1, ky1], marker="s", s=150,
               color=[C_ENC, C_DEC])
    ax.scatter([0.55], [ky2], marker="s", s=150, color=C_SEG)
    ax.text(1.2, ky1, "encoder + decoder body + bottleneck: transferred",
            va="center", fontsize=10.5, color=C_TEXT)
    ax.text(1.2, ky2, "all seg_layers: randomly re-initialized",
            va="center", fontsize=10.5, color=C_TEXT)

    # the code line that drives the re-init
    ax.text(8.7, 0.7, 'skip_strings_in_pretrained = (".seg_layers.",)',
            ha="center", va="center", family="monospace", fontsize=9.5,
            color=C_MUTE)

    ax.set_title("ResEnc-L U-Net: what transfers and what re-initializes",
                 loc="left", x=0.02, fontsize=13, color=C_TEXT, y=0.99)

    out = FIG_DIR / "architecture_transfer.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "assets" / "data" / "Dataset999_Phantom"
)


def _load_phantom():
    """Load the committed synthetic phantom CT + label as ``[x, y, z]`` arrays."""
    import nibabel as nib

    img = nib.load(DATA_DIR / "imagesTr" / "PHANTOM_001_0000.nii.gz")
    lab = nib.load(DATA_DIR / "labelsTr" / "PHANTOM_001.nii.gz")
    return np.asarray(img.dataobj), np.asarray(lab.dataobj)


def _best_lesion_slice(mask: np.ndarray) -> int:
    """Axial index whose lesion cross-section is largest (so the overlay reads)."""
    areas = mask.reshape(-1, mask.shape[2]).sum(axis=0)
    return int(np.argmax(areas)) if areas.any() else mask.shape[2] // 2


def _render_slice(name: str, sl, lo, hi, overlay_mask=None, color=C_SEG) -> Path:
    """Save one axial slice on a black 'scan viewer' background, with an optional
    binary ``overlay_mask`` painted in ``color``."""
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=200)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, origin="lower")
    if overlay_mask is not None:
        masked = np.ma.masked_where(overlay_mask == 0, overlay_mask)
        ax.imshow(masked, cmap=ListedColormap([color]), alpha=0.55, origin="lower")
    ax.axis("off")
    out = FIG_DIR / name
    fig.savefig(out, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out


def build_ct_slices() -> list[Path]:
    """Render real phantom slices for the deck on a black scan-viewer background.

    Overlays use the same warm accent as the architecture figure's ``seg_layers``
    so the deck reads as one consistent visual language. The "prediction" overlays
    are the repo's *illustrative* fixture (eroded GT plus two spurious blobs), not a
    model output, matching how notebooks 04 to 06 frame it.
    """
    from scipy import ndimage

    from teaching_fixtures import illustrative_prediction

    ct, mask = _load_phantom()
    z = _best_lesion_slice(mask)
    sl = np.rot90(ct[:, :, z])
    gt = np.rot90((mask[:, :, z] == 1).astype(int))
    lo, hi = np.percentile(sl, [1, 99])  # robust window for the grayscale

    pred_vol = illustrative_prediction(mask)
    pred = np.rot90(pred_vol[:, :, z])
    # connected-component cleanup: keep only the largest 2D component
    lbl, n = ndimage.label(pred)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
        pred_clean = (lbl == (int(np.argmax(sizes)) + 1)).astype(int)
    else:
        pred_clean = pred

    return [
        _render_slice("ct_slice.png", sl, lo, hi),
        _render_slice("ct_overlay.png", sl, lo, hi, gt),
        _render_slice("ct_pred_overlay.png", sl, lo, hi, pred),
        _render_slice("ct_pred_clean_overlay.png", sl, lo, hi, pred_clean),
    ]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a = build_lr_schedule()
    b = build_architecture()
    cts = build_ct_slices()
    for p in (a, b, *cts):
        print(f"wrote {p.relative_to(FIG_DIR.parent.parent)} "
              f"({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
