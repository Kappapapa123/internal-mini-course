# Fine-tuning nnU-Net with Pre-trained Models: a two-phase LR schedule

A short, laptop-runnable course on fine-tuning nnU-Net from a pretrained
MultiTalent-style foundation model using a two-phase learning-rate schedule
(linear warm-up, then polynomial decay). It pairs a 50-minute slide talk with one
Jupyter notebook per section.

## Course materials

- **Slides** (read-only): https://docs.google.com/presentation/d/1Wl9y9EIAsGPOuoC2OpM2PAfnT9VTAYRGLwaIw3b3vhs/preview
- **Quiz/NotebookLM**: https://notebooklm.google.com/notebook/3ccfa389-cc14-4755-bc97-37663138e12d?authuser=1

## Learning objectives

By the end you will be able to:
1. Lay out a dataset in nnU-Net format and explain why fine-tuning must match the pretrained
   checkpoint's preprocessing (for example 1 mm isotropic with Z-score, set by moving the plans).
2. Identify which weights transfer (encoder and decoder body) and which are re-initialized
   (all deep-supervision segmentation layers).
3. Implement and plot the two-phase LR schedule, and explain why a lower learning rate with
   warm-up helped on the TBI task and why it is not universally better (PANTHER).
4. Run sliding-window inference, simple post-processing, and Dice evaluation.

## Run the notebooks (no GPU, no cluster, no private repo needed)

```bash
uv sync                          # installs CPU-only PyTorch and the other deps
uv run python scripts/generate_phantom.py --out assets/data   # synthetic sample
uv run pytest                    # course_utils correctness checks
uv run jupyter lab               # open notebooks/01..06 and run top to bottom
```

The notebooks teach the logic of nnU-Net on a tiny synthetic phantom. Real `nnUNetv2_*`
commands appear as read-only cells. The committed notebooks already contain their executed
outputs (figures and printouts), so you can read them without running anything.

Kernel note: `uv run jupyter lab` uses the project's `.venv` automatically. If you open the
notebooks in another editor (for example VS Code), select the `.venv` interpreter or kernel,
or the `course_utils` and `torch` imports will fail.

To regenerate and re-execute the notebooks after editing `scripts/build_notebooks.py`:

```bash
uv run python scripts/build_notebooks.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/0[1-6]*.ipynb
```

Notebook 07 is deliberately outside that glob: it needs a GPU and real weights, so its
outputs come from a manual Colab run (see below).

## Optional: run the actual model (Colab GPU)

[`notebooks/07_real_checkpoints_colab.ipynb`](notebooks/07_real_checkpoints_colab.ipynb)
closes the loop. Notebooks 01–06 teach the *logic* on a phantom, and NB03's
plain-versus-warm-up comparison is read from a precomputed CSV. NB07 loads two **real**
fold-0 fine-tunes that differ only in LR schedule, runs `nnUNetv2_predict` with each on real
held-out liver CT, and computes the Dice difference in front of you. Inference only — no
training, no cluster.

It needs a Colab **GPU** runtime. The weights and two sample cases are downloaded from
[`KS987/multitalentv2-finetune-liver`](https://huggingface.co/KS987/multitalentv2-finetune-liver);
[`MODEL_CARD.md`](MODEL_CARD.md) describes exactly what they are, including the CC BY 4.0
chain from the MultiTalentV2 base model through the liver-lesion dataset. To use your own
fine-tunes instead, point `CKPT_SOURCE` or `LOCAL_CKPT_DIR` at them and every other cell
works unchanged.

This notebook is **additive and optional**. 

## What this course does not cover

- Running real nnU-Net **training** (needs CUDA, large patches, and about 24 GB VRAM).
  Real *inference* is optional and covered by notebook 07 on a Colab GPU.
- Cluster and scheduler specifics, which stay in the lab's private materials.
- No pretrained checkpoint and no patient data are committed to this repository. NB07
  downloads openly-licensed (CC BY 4.0) CT cases and fine-tuned weights at runtime; see
  [`NOTICE.md`](NOTICE.md) and [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

## Repository layout

```
notebooks/     01 data, 02 architecture, 03 training (main), 04 inference, 05 post, 06 eval,
               07 real checkpoints on a Colab GPU (optional, needs GPU + weights)
course_utils/  vendored, simplified, CPU-runnable teaching code (lr_schedulers, dsc, viz)
scripts/       generate_phantom.py, build_notebooks.py, teaching_fixtures.py
tests/         correctness checks for the schedulers and Dice
assets/        figures/, data/ (synthetic, committed), precomputed/
MODEL_CARD.md  card for the fine-tuned checkpoints notebook 07 uses
```

## Limitations and honesty notes

- Schedulers and behavior drift across nnU-Net versions. The concepts here are verified
  against a specific version (state it before release); the code is a teaching reimplementation.
- The warm-up gain is small (about +0.4 Dice over plain 1e-3 on TBI) and task dependent
  (PANTHER saw no benefit). Validate the schedule for your own task.
