# Data provenance

All sample data committed to this repository is **synthetic** and generated
deterministically. It contains **no patient data** and is safe to redistribute.

```yaml
synthetic: true
generator: scripts/generate_phantom.py
seed: 20260604
shape: [64, 64, 32]      # x, y, z voxels
spacing_mm: [1.0, 1.0, 1.0]
modality: CT-like (Hounsfield-ish intensities, then Z-score normalized in notebooks)
labels:
  0: background
  1: synthetic "lesion" (ellipsoidal blob)
contains_patient_data: false
license: CC0-1.0   # synthetic, no rights reserved on the data itself
```

Regenerate with:

```bash
uv run python scripts/generate_phantom.py --out assets/data
```

If an openly-licensed *real* sample is ever added (e.g. a Medical Segmentation
Decathlon case), note that MSD carries its own terms and a cropped/downsampled
case is a licensed derivative, so add a dedicated `DATA_LICENSE.md` and cite the
source at that time. The default for this course remains synthetic.
