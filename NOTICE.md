# Third-party attributions

This course teaches and adapts ideas from the following work. Code in
`course_utils/` is a **simplified teaching reimplementation**; it is not a drop-in
copy of any upstream package.

## nnU-Net
- **Project:** nnU-Net (MIC-DKFZ), Apache License 2.0.
- **Used for:** dataset format, preprocessing/plans concepts, deep-supervision
  segmentation heads, sliding-window inference, connected-component post-processing.
- The `nnUNetv2_*` commands shown in the notebooks are illustrative and **read-only**;
  the course does not bundle or run nnU-Net.

## Learning-rate schedulers (`course_utils/lr_schedulers.py`)
- Behavior (linear warm-up → offset polynomial decay) is reimplemented for teaching,
  matching the schedule used by the PANTHER / TBI fine-tuning trainers. The exact
  pretrained checkpoint, its plans file, source URL, and license must be named in the
  README before any public release (do **not** redistribute the checkpoint here).

## Reference papers (cited, not redistributed)
- **AIMS-TBI** (`arXiv:2504.06741`): "Large Scale Supervised Pretraining For Traumatic
  Brain Injury Segmentation" (Ulrich, Wald, Isensee, Maier-Hein, DKFZ). MultiTalent-inspired
  ResEnc-L pretraining; the warm-up to 1e-3 fine-tuning ablation (Table 2).
- **PANTHER** (`arXiv:2508.21775`): "A Multi-Stage Fine-Tuning and Ensembling Strategy for
  Pancreatic Tumor Segmentation in Diagnostic and Therapeutic MRI" (Team MIC-DKFZ).
  Multi-stage cascaded transfer; used here as a contrast (the warm-up benefit is task dependent).
- **MultiTalent** (`arXiv:2303.14444`): multi-dataset CT segmentation; a shared encoder
  with per-task heads.

> Note: the lab's `MULTITALENT_FT_HANDOFF.md` cites the TBI paper as `2508.21775`, which is
> actually PANTHER's ID. The IDs above are taken from the papers' own title pages.

Paper PDFs are **not** committed to this repository (copyright). Figures derived from
papers are **redrawn** from reported values and cited, not screenshotted.

## Sample data
- All committed sample data is **synthetic** (see `DATA_PROVENANCE.md`). No patient data.
