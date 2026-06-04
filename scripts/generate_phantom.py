"""Generate a deterministic synthetic CT phantom in nnU-Net dataset format.

Produces a tiny, license-safe sample (no patient data) the notebooks can load:

    <out>/Dataset999_Phantom/
        imagesTr/PHANTOM_001_0000.nii.gz   # CT-like volume (channel 0)
        labelsTr/PHANTOM_001.nii.gz        # integer label map (0 bg, 1 lesion)
        dataset.json                       # nnU-Net v2 dataset descriptor

The "_0000" suffix is nnU-Net's channel convention; the label file shares the
case id with no channel suffix. Run:

    uv run python scripts/generate_phantom.py --out assets/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 20260604
SHAPE = (64, 64, 32)  # x, y, z
SPACING = (1.0, 1.0, 1.0)  # mm, isotropic (matches the teaching preprocessing story)


def _ellipsoid(shape: tuple[int, int, int], center, radii) -> np.ndarray:
    xx, yy, zz = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cx, cy, cz = center
    rx, ry, rz = radii
    return (
        ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2
    ) <= 1.0


def make_phantom(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return (ct_volume float32 HU-ish, label_map int16) deterministically."""
    rng = np.random.default_rng(seed)
    cx, cy, cz = (s / 2 for s in SHAPE)

    # Air background around a soft-tissue "body" ellipsoid.
    ct = np.full(SHAPE, -1000.0, dtype=np.float32)
    body = _ellipsoid(SHAPE, (cx, cy, cz), (26, 26, 13))
    ct[body] = 40.0 + rng.normal(0, 8, size=ct.shape).astype(np.float32)[body]

    # A higher-intensity ellipsoidal "lesion" offset from centre.
    lesion = _ellipsoid(SHAPE, (cx + 8, cy - 5, cz + 2), (7, 6, 5))
    ct[lesion] = 130.0 + rng.normal(0, 6, size=ct.shape).astype(np.float32)[lesion]

    label = np.zeros(SHAPE, dtype=np.int16)
    label[lesion] = 1
    return ct, label


def _save_nifti(array: np.ndarray, path: Path) -> None:
    import nibabel as nib  # lazy import so --help works without nibabel

    affine = np.diag([*SPACING, 1.0])
    nib.save(nib.Nifti1Image(array, affine), str(path))


def generate(out: Path = Path("assets/data"), seed: int = SEED) -> Path:
    """Write the synthetic phantom dataset under ``out`` and return its root dir."""
    root = Path(out) / "Dataset999_Phantom"
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    ct, label = make_phantom(seed)
    _save_nifti(ct, root / "imagesTr" / "PHANTOM_001_0000.nii.gz")
    _save_nifti(label, root / "labelsTr" / "PHANTOM_001.nii.gz")

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": 1,
        "file_ending": ".nii.gz",
        "description": "Synthetic CT phantom (no patient data). See DATA_PROVENANCE.md.",
    }
    (root / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("assets/data"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root = generate(args.out, args.seed)
    print(f"wrote synthetic phantom to {root}")


if __name__ == "__main__":
    main()
