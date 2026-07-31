# Data provenance

All sample data **committed to this repository** is synthetic and generated
deterministically. It contains **no patient data** and is safe to redistribute.
Notebooks 01–06 use only this synthetic phantom.

Notebook 07 is the one exception, and it downloads at runtime rather than committing
anything — see "Real data in notebook 07" below.

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

## Real data in notebook 07

`notebooks/07_real_checkpoints_colab.ipynb` (optional, Colab GPU) runs inference on a
couple of **real** held-out CT cases from `Dataset591_liver_lesions` — "Training dataset
for TotalSegmentator task liver_lesions", Jakob Wasserthal, University Hospital Basel,
licensed **CC BY 4.0**, version DOI `10.5281/zenodo.20272572` (concept DOI
`10.5281/zenodo.20272571`).

```yaml
synthetic: false
used_by: notebooks/07_real_checkpoints_colab.ipynb   # only
committed_to_this_repo: false      # downloaded at runtime, never checked in
license: CC-BY-4.0
attribution: Jakob Wasserthal, University Hospital Basel (TotalSegmentator)
doi_version: 10.5281/zenodo.20272572
contains_patient_data: false       # public, de-identified, openly licensed release
```

CC BY 4.0 permits redistribution with attribution, so the mirrored cases live alongside the
NB07 checkpoints (see [`MODEL_CARD.md`](MODEL_CARD.md)) rather than in git — the upstream
release is a single 38.9 GB zip, which a Colab session cannot download to get two cases.
Attribution and both DOIs appear in the model card, in NB07's own cells, and in
[`NOTICE.md`](NOTICE.md).

If any other openly-licensed real sample is added later (e.g. a Medical Segmentation
Decathlon case), note that MSD carries its own terms and a cropped/downsampled case is a
licensed derivative, so add a dedicated `DATA_LICENSE.md` and cite the source at that time.
The default for committed data in this course remains synthetic.
