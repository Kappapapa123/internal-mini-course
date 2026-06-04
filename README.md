# Fine-tuning nnU-Net with Pre-trained Models: a two-phase LR schedule

A short, laptop-runnable course on fine-tuning nnU-Net from a pretrained
MultiTalent-style foundation model using a two-phase learning-rate schedule
(linear warm-up, then polynomial decay). It pairs a 50-minute slide talk with one
Jupyter notebook per section.

> What "two-phase" means here: a learning-rate schedule for the fine-tuning run. The
> learning rate ramps linearly from about 2e-5 to 1e-3 over 50 epochs, then decays
> polynomially toward 0. The whole network trains the entire time. 

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
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## What this course does not cover

- Running real nnU-Net training or inference (needs CUDA, large patches, and about 24 GB VRAM).
- Installing nnU-Net itself, or distributing any pretrained checkpoint or patient data.
- Cluster and scheduler specifics, which stay in the lab's private materials.

## Repository layout

```
slides/        OUTLINE.md (per-slide spec, speaker notes, timing) and deck.pptx
notebooks/     01 data, 02 architecture, 03 training (main), 04 inference, 05 post, 06 eval
course_utils/  vendored, simplified, CPU-runnable teaching code (lr_schedulers, dsc, viz)
scripts/       generate_phantom.py, build_notebooks.py, teaching_fixtures.py
tests/         correctness checks for the schedulers and Dice
assets/        figures/, data/ (synthetic, committed), precomputed/
```

## Citations and provenance

See [`NOTICE.md`](NOTICE.md) for third-party attribution (nnU-Net, the TBI, PANTHER, and
MultiTalent papers) and [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) for the synthetic data.
Course code is MIT; course content is additionally CC-BY-4.0.

## Limitations and honesty notes

- Schedulers and behavior drift across nnU-Net versions. The concepts here are verified
  against a specific version (state it before release); the code is a teaching reimplementation.
- The warm-up gain is small (about +0.4 Dice over plain 1e-3 on TBI) and task dependent
  (PANTHER saw no benefit). Validate the schedule for your own task.
