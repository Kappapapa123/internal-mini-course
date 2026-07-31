---
license: cc-by-4.0
pipeline_tag: image-segmentation
tags:
  - medical-imaging
  - segmentation
  - nnunet
  - computed-tomography
---

# Liver-lesion nnU-Net fine-tunes: an LR-schedule ablation

Two nnU-Net v2 checkpoints that differ **only** in the fine-tuning learning-rate schedule.
They exist to make one teaching claim reproducible: that a short linear warm-up to 1e-3
beats jumping straight to a constant 1e-3 when fine-tuning a pretrained segmentation
encoder. Companion to the mini-course at
https://github.com/Kappapapa123/internal-mini-course (notebook 07).

## The two variants

| Variant | Trainer | Schedule | Liver-lesion Dice |
|---|---|---|---|
| `plain1e3` | `nnUNetTrainer_plain1e3_wandb` | LR 1e-3 with default PolyLR (PANTHER-style default) | **0.7889520** |
| `warmup1e3` | `nnUNetTrainer_warmup1e3_wandb` | 50-epoch linear warm-up (2e-5 → 1e-3 over 50 epochs), then offset PolyLR (TBI-style) | **0.8026699** |

Difference: **+0.0137 Dice** (+1.37 points) for warm-up.

This is the liver-lesion **class** Dice from nnU-Net's own end-of-training validation
(`fold_0/validation/summary.json`), computed on the **169 fold-0 validation cases** of the
842 — cases neither checkpoint trained on, but *not* an independent test set held out from
the whole 5-fold design. It is not the mean-foreground pseudo-Dice logged during training;
see Limitations.

## Base model

**MultiTalentV2 Challenge Edition**, Constantin Ulrich (DKFZ) — a multi-dataset CT
segmentation ResEnc-L in the MultiTalent (`arXiv:2303.14444`) / AIMS-TBI
(`arXiv:2504.06741`) line of work from MIC-DKFZ.

- Zenodo record: https://zenodo.org/records/13753413 (published 2024-09-12, open access)
- License: **CC BY 4.0** — attribution only, no further restrictions, so redistributing a
  fine-tuned derivative is permitted provided the base model is credited
- Related publication: `10.1007/978-3-031-43898-1_6`
- Specific weights used: the **native-CT Z-score variant**,
  `Dataset617_nativect/...nnUNetResEncUNetL1x1x1_Plans_znorm_bs24.../fold_all/checkpoint_final.pth`.
  `checkpoint_final.pth` rather than `checkpoint_best.pth`, following PANTHER's finding that
  final works better for this foundation model.
- The fine-tuning plans (`..._znorm_bs24_mig_bs1`) are the base model's own plans with only
  `batch_size` changed 24 → 1 to fit a MIG slice. This is what makes the preprocessing match
  — the point NB01 makes.
- Transfer check: 946 / 946 transferable non-segmentation keys loaded.

The whole chain is CC BY 4.0: base checkpoint (Zenodo 13753413) and fine-tuning data
(Zenodo 20272572).

## Fine-tuning dataset

`Dataset591_liver_lesions` — "Training dataset for TotalSegmentator task liver_lesions",
Jakob Wasserthal, University Hospital Basel. **CC BY 4.0.**

- Zenodo record: https://zenodo.org/records/20272572
- Version DOI: `10.5281/zenodo.20272572` (pinned; a new Zenodo version cannot silently
  change what was used here)
- Concept DOI: `10.5281/zenodo.20272571`
- 842 CT cases with manual liver-lesion segmentations, already in nnU-Net v2 format,
  published 2026-05-18, distributed as a single 38.9 GB zip
- Labels: `{background: 0, liver_lesion: 1}`, single CT channel
- Split: nnU-Net's own 5-fold split, fold 0 = **673 train / 169 validation**

The license stated here is the one on the Zenodo deposit, which is the authoritative record.
(The `dataset.json` inside the archive carries a leftover `"licence": "Apache 2.0"` from the
TotalSegmentator template; both licenses permit redistribution with attribution, so nothing
about the mirroring depends on which one applies.)

A small number of fold-0 **validation** cases (image + ground-truth label) are mirrored in
this repo under `cases/` in nnU-Net raw layout — cases neither checkpoint trained on. This
is because a Colab session cannot download 38.9 GB to obtain two cases. Please cite
Wasserthal if you use them.

## Training configuration

- Configuration: `3d_fullres`, fold 0, 1000 epochs
- Plans: `nnUNetResEncUNetL1x1x1_Plans_znorm_bs24_mig_bs1` — 1 mm isotropic spacing,
  Z-score normalization, patch size 192×192×192, batch size 1, `batch_dice: True`, matched
  to the pretrained checkpoint's preprocessing
- Architecture: `dynamic_network_architectures...ResidualEncoderUNet`, 6 stages,
  features per stage `[32, 64, 128, 256, 320, 320]`, `n_blocks_per_stage [1, 3, 4, 6, 6, 6]`,
  InstanceNorm3d, LeakyReLU
- Optimizer: SGD, momentum 0.99, Nesterov
- Peak LR 1e-3 for both variants (the peak is identical; only the ramp differs). Verified in
  the training logs: plain starts at `Current learning rate: 0.001` at epoch 0, warm-up
  starts at `2e-05`.
- nnU-Net version: **2.6.4** (`/home/kcuoft/nnunet_env_fir`, Python 3.11). The two LR
  variants are installed as
  `nnunetv2/training/nnUNetTrainer/variants/lr_schedule/{plain1e3,warmup1e3}_wandb.py`, so a
  reader loading these checkpoints needs the trainer files too, not just a matching pip
  version.
- Hardware: one H100 MIG `2g.20gb` slice (20 GB), 3 CPUs, 48 GB host RAM, Alliance Canada
  (Fir). Elapsed 23:17:51 (plain) and 1-04:51:18 (warm-up).
- Provenance: Phase-1 / S5 array job `42949211`, tasks `_0` (plain) and `_1` (warm-up); both
  `COMPLETED`, exit `0:0`, per `sacct`. This array resumed a partially OOM-killed earlier
  array (`42820974`). These runs predate the every-100-epoch snapshot mixin, so there are no
  `checkpoint_ep*.pth` files — only `best`, `final`, and `latest`.

## Files

Laid out exactly as `$nnUNet_results` expects, so `nnUNetv2_predict` finds them by path:

```
Dataset591_liver_lesions/
  nnUNetTrainer_plain1e3_wandb__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24_mig_bs1__3d_fullres/
    plans.json
    dataset.json
    fold_0/checkpoint_final.pth
  nnUNetTrainer_warmup1e3_wandb__.../fold_0/checkpoint_final.pth
  ...__3d_fullres/fold_0/validation_summary.json   # nnU-Net's own scores, as provenance
cases/images/<CASE>_0000.nii.gz     # 2 mirrored fold-0 validation cases
cases/labels/<CASE>.nii.gz
README.md         # this card
```

The two mirrored cases are `QTbgDDsctkRUgSTt` (207×199×141, 10,195 lesion voxels) and
`eZKVPnX7guzPW2Ph` (342×316×101, 19,942 lesion voxels) — both with substantial lesions, so
Dice on them is meaningful rather than dominated by a handful of voxels.

These are the unmodified nnU-Net checkpoints, byte for byte as trained — the sha256 sums
below match the files on Fir, so you can verify what you downloaded. That means each file
also carries the optimizer state nnU-Net saves for resuming training, which is most of the
819 MB and is not needed for inference. Stripping it would cut the download by roughly two
thirds; it is not done here, in favour of hashes that match the originals.

| File | Size | sha256 |
|---|---|---|
| `plain1e3/fold_0/checkpoint_final.pth` | 819,491,031 B | `91dd81f0d738885bcb8bf0cf9bb6d839b4012a975902614b256bc9ff42539168` |
| `warmup1e3/fold_0/checkpoint_final.pth` | 819,491,287 B | `7ae5546dd1c95cb9fed93bc4719c2a6e0643ff5bbdba119c3ac67101841abf1e` |

The training facts in this card were read off the original run directories on the Alliance
Canada *Fir* cluster on 2026-07-30, and both checkpoints' sha256 sums were re-verified
against the values above at upload time.

## Intended use

**Teaching and methods research.** Specifically: letting a reader of the mini-course verify
the plain-versus-warm-up comparison themselves instead of trusting a precomputed CSV.

**Not for clinical use.** These are single-fold research checkpoints trained on one public
dataset from one institution, with no prospective validation, no regulatory clearance, and
no evaluation of subgroup performance. Do not use them to inform patient care.

## Limitations

- **The gain is task dependent.** +1.37 Dice points on liver lesions here, about +0.4 on
  TBI segmentation, and *no* benefit on the PANTHER pancreas-MRI task. Warm-up is not
  universally better; run the ablation for your own task.
- **The training curves nearly overlap.** For these same two runs, nnU-Net's
  mean-foreground pseudo-Dice peaks at 0.899 (plain) versus 0.902 (warm-up) and ends at
  0.882 versus 0.881 — i.e. the training curve does not predict the final gap. The
  separation appears in *per-class* evaluation. Same runs, different metric.
- **Fold 0 only**, single seed, and the reported Dice is that fold's *validation* split
  rather than a set held out from the whole 5-fold design. No cross-fold variance estimate,
  so the +0.0137 gap has no error bar.
- **The same ablation on a second dataset agreed but by a different margin**: on
  `Dataset315_pleural_pericard_effusion`, warm-up won by +0.0205 on the decision metric
  (nanmean of pleural and pericardial effusion Dice). Two datasets, both favouring warm-up,
  is still two datasets.
- **Domain**: portal-venous-ish abdominal CT at 1 mm isotropic, one institution. Expect
  degradation on other scanners, protocols, and contrast phases.
- **Untested on a 16 GB GPU.** These were trained and validated on a 20 GB H100 MIG slice.
  Inference needs less memory than training, so a 16 GB Colab T4 should be enough, but that
  has not been measured. Notebook 07 carries a fallback ladder if you hit out-of-memory.

## Citation

Please cite the dataset and the underlying methods, not this teaching artifact:

- Wasserthal, J. *Training dataset for TotalSegmentator task liver_lesions.* Zenodo.
  https://doi.org/10.5281/zenodo.20272572
- Ulrich, C. *MultiTalentV2 Challenge Edition.* Zenodo (2024).
  https://zenodo.org/records/13753413
- Isensee, F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical
  image segmentation.* Nature Methods 18, 203–211 (2021).
- Ulrich, C., Wald, T., Isensee, F., Maier-Hein, K. *Large Scale Supervised Pretraining For
  Traumatic Brain Injury Segmentation.* `arXiv:2504.06741`.
- Ulrich, C. et al. *MultiTalent: A Multi-Dataset Approach to Medical Image Segmentation.*
  `arXiv:2303.14444`.

## License

Course code and this card: MIT. Course content: CC-BY-4.0. The mirrored `cases/` data:
CC BY 4.0, © Jakob Wasserthal / University Hospital Basel. The checkpoint weights are
**CC BY 4.0**, inherited from the MultiTalentV2 Challenge Edition base model
(© Constantin Ulrich / DKFZ) — attribution to both the base model and the fine-tuning
dataset is required if you redistribute or build on them.
