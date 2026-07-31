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
  matching the schedule used by the PANTHER / TBI fine-tuning trainers.

## Pretrained base checkpoint
- **MultiTalentV2 Challenge Edition**, Constantin Ulrich (DKFZ) — a multi-dataset CT
  segmentation ResEnc-L. **CC BY 4.0**, https://zenodo.org/records/13753413, related
  publication `10.1007/978-3-031-43898-1_6`. The native-CT Z-score variant
  (`Dataset617_nativect`, `fold_all`, `checkpoint_final.pth`) is what the course's
  fine-tunes start from; details in [`MODEL_CARD.md`](MODEL_CARD.md).
- No checkpoint — base or fine-tuned — is committed to **this** repository. Notebook 07
  optionally downloads fine-tuned weights at runtime from a separate Hugging Face model
  repo whose card is `MODEL_CARD.md`. CC BY 4.0 permits that redistribution with
  attribution; publication is pending lab sign-off, and until then NB07 runs in
  bring-your-own-checkpoint mode.

## Dataset591_liver_lesions (used by notebook 07 only)
- **"Training dataset for TotalSegmentator task liver_lesions"**, Jakob Wasserthal,
  University Hospital Basel. **CC BY 4.0.**
- Version DOI `10.5281/zenodo.20272572` (record: https://zenodo.org/records/20272572);
  concept DOI `10.5281/zenodo.20272571`.
- Not committed here. A few held-out cases are mirrored alongside the NB07 checkpoints,
  which CC BY 4.0 permits with attribution; attribution is carried in the model card, in
  NB07 itself, and here.

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
- Notebook 07 downloads real, openly-licensed (CC BY 4.0) CT cases at runtime; still
  nothing licensed or patient-identifiable is committed to this repository.
