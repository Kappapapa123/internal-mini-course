"""Author the course notebooks deterministically with nbformat.

The ``.ipynb`` files under ``notebooks/`` are the committed deliverables; this script
is their source of truth. Edit here, then regenerate and execute:

    uv run python scripts/build_notebooks.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

Every notebook is laptop/CPU runnable: it teaches the logic of nnU-Net on the tiny
synthetic phantom, and shows real ``nnUNetv2_*`` commands as read-only markdown. A shared
bootstrap cell makes each notebook runnable from any working directory (it adds the repo
to ``sys.path`` and regenerates the synthetic data if missing).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
REPO_ROOT = NB_DIR.parent

BOOTSTRAP = '''\
# bootstrap: make course_utils + scripts importable from any working directory,
# use the inline backend so figures render, and regenerate the phantom if missing.
%matplotlib inline
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


REPO = _find_repo_root(Path.cwd())
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = REPO / "assets" / "data" / "Dataset999_Phantom"
PRE = REPO / "assets" / "precomputed"

if not (DATA / "imagesTr" / "PHANTOM_001_0000.nii.gz").exists():
    import generate_phantom
    generate_phantom.generate(REPO / "assets" / "data")

print("repo root:", REPO.name)
'''

LOAD_PHANTOM = (
    "import nibabel as nib\n"
    "ct = np.asarray(nib.load(str(DATA / 'imagesTr' / 'PHANTOM_001_0000.nii.gz')).dataobj, dtype=np.float32)\n"
    "gt = np.rint(np.asarray(nib.load(str(DATA / 'labelsTr' / 'PHANTOM_001.nii.gz')).dataobj)).astype(int)\n"
)


def _save(nb, name: str) -> None:
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, str(NB_DIR / name))
    print("wrote", (NB_DIR / name).relative_to(REPO_ROOT))


# --------------------------------------------------------------------------- NB01
def nb01() -> None:
    c = [
        new_markdown_cell(
            "# 01. Data preparation\n\n"
            "**Goal:** see the nnU-Net dataset format, look at the data, and understand the most "
            "important step for fine-tuning: matching the pretrained checkpoint's preprocessing.\n\n"
            "The sample is a synthetic CT phantom (no patient data; see `DATA_PROVENANCE.md`)."
        ),
        new_code_cell(BOOTSTRAP),
        new_markdown_cell(
            "## nnU-Net dataset layout\n\n"
            "```\n"
            "Dataset999_Phantom/\n"
            "  imagesTr/PHANTOM_001_0000.nii.gz   # channel 0 (the '_0000' suffix marks the modality)\n"
            "  labelsTr/PHANTOM_001.nii.gz        # label map, same case id, no channel suffix\n"
            "  dataset.json                       # declares channel_names and labels\n"
            "```"
        ),
        new_code_cell(
            "import json\n"
            "meta = json.loads((DATA / 'dataset.json').read_text())\n"
            "print(json.dumps(meta, indent=2))"
        ),
        new_markdown_cell(
            "## Load the volume and look at it\n\n"
            "Load the CT and its label, check the shape and intensity range, and list the label "
            "values. Then view a few axial slices so you can see the phantom and its lesion.\n\n"
            "We render the slices with a **fixed intensity window** so brightness is comparable "
            "across them. The lesion sits near the middle of the volume, so it appears only on "
            "the central slice (z=16); the other slices show body tissue and air only."
        ),
        new_code_cell(
            LOAD_PHANTOM
            + "print('CT shape:', ct.shape)\n"
            "print('intensity range:', (round(float(ct.min())), round(float(ct.max()))))\n"
            "print('label values (np.unique):', np.unique(gt))"
        ),
        new_code_cell(
            "fig, axes = plt.subplots(1, 3, figsize=(9, 3))\n"
            "# Fixed window (vmin/vmax) so the slices are directly comparable. Without it,\n"
            "# matplotlib auto-scales each slice to its own min/max, so the body renders white\n"
            "# on the lesion-free slices (it is the brightest thing there) and looks identical\n"
            "# to the lesion on z=16. The lesion is only on z=16.\n"
            "for ax, z in zip(axes, [8, 16, 24]):\n"
            "    ax.imshow(ct[:, :, z].T, cmap='gray', origin='lower', vmin=-200, vmax=200)\n"
            "    ax.set_title(f'CT, z={z}')\n"
            "    ax.axis('off')\n"
            "fig.suptitle('Synthetic CT phantom (axial slices)')\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
        new_code_cell(
            "from course_utils.viz import overlay_mask_on_slice\n"
            "ax = overlay_mask_on_slice(ct, gt)\n"
            "ax.set_title('lesion label overlaid on the central slice')\n"
            "ax.figure.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## Match the pretraining preprocessing\n\n"
            "The checkpoint was trained on volumes resampled to 1 mm isotropic and Z-score "
            "normalized. Fine-tuning data has to be preprocessed the same way, or the pretrained "
            "features do not line up with the inputs. This is a property of the chosen checkpoint, "
            "not a default nnU-Net rule: nnU-Net normally derives spacing and normalization from "
            "your own dataset.\n\n"
            "In practice you copy the checkpoint's `plans.json` onto your dataset and preprocess "
            "with it (read-only here; needs the real toolchain and the checkpoint):\n\n"
            "```bash\n"
            "nnUNetv2_move_plans_between_datasets -s <PRETRAIN_ID> -t 999 \\\n"
            "    -sp nnUNetResEncUNetLPlans -tp nnUNetResEncUNetLPlans\n"
            "nnUNetv2_preprocess -d 999 -plans_name nnUNetResEncUNetLPlans -c 3d_fullres\n"
            "```"
        ),
        new_code_cell(
            "# The Z-score step the pretraining used, shown here on the phantom body (ignoring air).\n"
            "body = ct > -500\n"
            "z = ct.copy()\n"
            "z[body] = (ct[body] - ct[body].mean()) / (ct[body].std() + 1e-8)\n"
            "print('after Z-score on the body: mean=%.3f std=%.3f' % (z[body].mean(), z[body].std()))"
        ),
        new_markdown_cell(
            "## Recap\n"
            "1. nnU-Net uses a fixed `imagesTr / labelsTr / dataset.json` layout with a `_0000` channel suffix.\n"
            "2. Always check spacing, intensities, and `np.unique(label)`.\n"
            "3. Match the checkpoint's preprocessing (1 mm isotropic, Z-score) by moving its plans onto your data."
        ),
    ]
    _save(new_notebook(cells=c), "01_data_preparation.ipynb")


# --------------------------------------------------------------------------- NB02
def nb02() -> None:
    c = [
        new_markdown_cell(
            "# 02. Model architecture and weight transfer\n\n"
            "**Goal:** understand which weights are copied from the pretrained checkpoint and which "
            "are randomly initialized when you fine-tune on a new task."
        ),
        new_code_cell(BOOTSTRAP),
        new_markdown_cell(
            "## What is in a trained network\n\n"
            "A trained PyTorch network is stored as a *state dict*: a dictionary that maps each "
            "layer's parameter name (a string such as `encoder.stages.0.convs.0.conv.weight`) to "
            "its weight tensor.\n\n"
            "Fine-tuning splits these parameters into two groups:\n"
            "- **transfer**: copy the tensor from the pretrained checkpoint (the encoder and the "
            "decoder body, which carry general anatomy);\n"
            "- **re-initialize**: start from random weights (the segmentation outputs, which depend "
            "on the new task's label set).\n\n"
            "nnU-Net decides this **by parameter name**: it loads every tensor whose name does not "
            "contain `.seg_layers.`, and randomly initializes the rest. Because nnU-Net uses **deep "
            "supervision**, segmentation outputs exist at several decoder depths, so there is not a "
            "single final layer; all of them are re-initialized."
        ),
        new_markdown_cell(
            "## A small concrete network\n\n"
            "A miniature module with the same structure: a shared body plus one segmentation head "
            "per decoder scale (deep supervision). We inspect its real `state_dict` keys."
        ),
        new_code_cell(
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class TinyDeepSupUNet(nn.Module):\n"
            "    def __init__(self, ch=4, n_classes=2, n_scales=3):\n"
            "        super().__init__()\n"
            "        self.encoder = nn.ModuleList([nn.Conv3d(1 if i == 0 else ch, ch, 3, padding=1) for i in range(n_scales)])\n"
            "        self.decoder = nn.ModuleList([nn.Conv3d(ch, ch, 3, padding=1) for _ in range(n_scales)])\n"
            "        self.seg_layers = nn.ModuleList([nn.Conv3d(ch, n_classes, 1) for _ in range(n_scales)])\n\n"
            "net = TinyDeepSupUNet()\n"
            "print('this toy net has', len(net.state_dict()), 'parameter tensors')"
        ),
        new_markdown_cell(
            "## Apply nnU-Net's rule: load everything except `.seg_layers.`\n\n"
            "For each parameter we show its shape, which part of the network it belongs to, and "
            "whether fine-tuning transfers it or re-initializes it."
        ),
        new_code_cell(
            "SKIP = '.seg_layers.'\n"
            "print(f\"{'parameter name':40s} {'shape':16s} {'part':10s} action\")\n"
            "print('-' * 80)\n"
            "for name, tensor in net.state_dict().items():\n"
            "    part = 'encoder' if name.startswith('encoder') else ('seg head' if 'seg_layers' in name else 'decoder')\n"
            "    action = 're-init (random)' if SKIP in ('.' + name) else 'transfer (load)'\n"
            "    print(f'{name:40s} {str(tuple(tensor.shape)):16s} {part:10s} {action}')"
        ),
        new_markdown_cell(
            "## Why re-initialize the heads?\n\n"
            "The pretraining heads predict the pretraining datasets' classes, which mean nothing for "
            "your lesion label. You drop them, attach fresh heads, and let fine-tuning learn the new "
            "mapping on top of the transferred encoder and decoder body.\n\n"
            "Caveat: if your class count matches the pretraining, nnU-Net may load the old heads "
            "instead. Re-initialization is not automatic; it follows from skipping the `.seg_layers.` "
            "keys.\n\n"
            "Real fine-tuning command (read-only):\n"
            "```bash\n"
            "nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainer_warmup1e3 \\\n"
            "    -pretrained_weights /path/to/checkpoint_final.pth\n"
            "```"
        ),
        new_markdown_cell(
            "## Recap\n"
            "1. A network is a dictionary of named weight tensors.\n"
            "2. Fine-tuning transfers the encoder and decoder body and re-initializes the segmentation outputs.\n"
            "3. The choice is made by name (`.seg_layers.`), and deep supervision means there are several seg layers."
        ),
    ]
    _save(new_notebook(cells=c), "02_model_architecture.ipynb")


# --------------------------------------------------------------------------- NB03
def nb03() -> None:
    c = [
        new_markdown_cell(
            "# 03. Training protocol: the two-phase LR schedule (centerpiece)\n\n"
            "**Goal:** implement and plot the schedule. The learning rate ramps up over the first 50 "
            "epochs (linear warm-up), then decays polynomially. The whole network trains the whole "
            "time; no part is frozen."
        ),
        new_code_cell(BOOTSTRAP),
        new_markdown_cell(
            "## Drive the real schedulers\n\n"
            "`course_utils` ships the same two schedulers the lab's fine-tuning trainer uses. We "
            "step a dummy optimizer through 1000 epochs and record the learning rate."
        ),
        new_code_cell(
            "import torch\n"
            "from course_utils.lr_schedulers import Lin_incr_LRScheduler, PolyLRScheduler_offset\n\n"
            "MAX_LR, WARMUP, TOTAL = 1e-3, 50, 1000\n"
            "param = torch.nn.Parameter(torch.zeros(1))\n"
            "opt = torch.optim.SGD([param], lr=MAX_LR, momentum=0.99, nesterov=True)\n\n"
            "warm = Lin_incr_LRScheduler(opt, max_lr=MAX_LR, max_steps=WARMUP)\n"
            "poly = PolyLRScheduler_offset(opt, initial_lr=MAX_LR, max_steps=TOTAL, start_step=WARMUP)\n"
            "lrs = []\n"
            "for epoch in range(TOTAL):\n"
            "    sched = warm if epoch < WARMUP else poly   # the trainer swaps schedulers at epoch 50\n"
            "    sched.step(epoch)\n"
            "    lrs.append(opt.param_groups[0]['lr'])\n\n"
            "print('epoch   0:', round(lrs[0], 8), '(about 2e-5, not zero)')\n"
            "print('epoch  49:', round(lrs[49], 8), '(peak)')\n"
            "print('epoch  50:', round(lrs[50], 8), '(decay restarts at the peak)')\n"
            "print('epoch 999: %.3e (heading to zero, still positive)' % lrs[999])\n"
            "assert abs(lrs[0] - 2e-5) < 1e-9 and abs(lrs[49] - 1e-3) < 1e-9 and abs(lrs[50] - 1e-3) < 1e-9"
        ),
        new_markdown_cell("## Plot the two-phase curve"),
        new_code_cell(
            "from course_utils.viz import plot_lr_curve\n"
            "ax, _ = plot_lr_curve(max_lr=MAX_LR, warmup_epochs=WARMUP, total_epochs=TOTAL)\n"
            "ax.figure.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## How the trainer wires it (read-only)\n\n"
            "A custom trainer swaps the scheduler at the right epoch and reuses the optimizer, so "
            "momentum carries across the switch. Both stages train the whole network; `'train'` is "
            "just the name of the post-warm-up decay phase.\n\n"
            "```python\n"
            "def on_train_epoch_start(self):\n"
            "    if self.current_epoch == 0:\n"
            "        self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_all')\n"
            "    elif self.current_epoch == self.warmup_duration_whole_net:   # == 50\n"
            "        self.optimizer, self.lr_scheduler = self.configure_optimizers('train')\n"
            "    super().on_train_epoch_start()\n"
            "```"
        ),
        new_markdown_cell(
            "## Compare on a real run\n\n"
            "The schedule only matters once you actually train. The comparison that isolates the "
            "warm-up is **plain 1e-3 (no warm-up) versus warm-up to 1e-3** (same 1e-3 peak, the only "
            "difference is the 50-epoch ramp). Run both fine-tuning jobs, export the validation Dice "
            "(`val/ema_fg_dice`, the smoothed foreground Dice nnU-Net uses to pick the best "
            "checkpoint) from W&B, save it to `assets/precomputed/real_training_curves.csv` (columns "
            "`epoch,plain,warmup`), and the cell below plots the comparison. Higher is better; no "
            "curve is fabricated. The curve below is an in-house reproduction on a separate dataset, "
            "distinct from the published TBI numbers.\n\n"
            "For the evidence table, paste the screenshot of Table 2 from the TBI paper "
            "(`arXiv:2504.06741`) onto the slide. For reference, the 5-fold average Dice is: "
            "from scratch 53.44, fine-tune at plain 1e-3 53.80, warm-up to 1e-2 53.28, warm-up to "
            "1e-3 54.21. So warm-up to 1e-3 beats plain 1e-3 by 0.41 Dice (modest but real), while "
            "warm-up to 1e-2 is worse than not warming up at all."
        ),
        new_code_cell(
            "import csv\n"
            "p = PRE / 'real_training_curves.csv'\n"
            "if p.exists():\n"
            "    rows = list(csv.DictReader(p.open()))\n"
            "    ep = [int(r['epoch']) for r in rows]\n"
            "    fig, ax = plt.subplots(figsize=(6, 3.2))\n"
            "    ax.plot(ep, [float(r['plain']) for r in rows], label='plain 1e-3 (no warm-up)')\n"
            "    ax.plot(ep, [float(r['warmup']) for r in rows], label='warm-up to 1e-3')\n"
            "    ax.axvline(WARMUP, color='crimson', ls='--', lw=1, label='warm-up ends')\n"
            "    ax.set_xlabel('epoch'); ax.set_ylabel('validation Dice (EMA)'); ax.legend()\n"
            "    ax.set_title('Fine-tuning: plain 1e-3 vs warm-up to 1e-3'); fig.tight_layout()\n"
            "    plt.show()\n"
            "else:\n"
            "    print('No real_training_curves.csv yet. Add it after the two fine-tuning runs.')"
        ),
        new_markdown_cell(
            "## A note on generality\n\n"
            "The warm-up benefit is task dependent. PANTHER (`arXiv:2508.21775`) tried warm-up and "
            "cosine schedules for a pancreas-MRI task and found they did not beat the default poly "
            "schedule. Run the small ablation for your own task before assuming warm-up helps.\n\n"
            "## Recap\n"
            "1. Two phases means warm-up then decay of the learning rate, with one peak at epoch 50.\n"
            "2. The peak is 1e-3, not the default 1e-2.\n"
            "3. The whole network trains throughout."
        ),
    ]
    _save(new_notebook(cells=c), "03_training_protocol.ipynb")


# --------------------------------------------------------------------------- NB04
def nb04() -> None:
    c = [
        new_markdown_cell(
            "# 04. Inference\n\n"
            "**Goal:** the ideas behind nnU-Net inference (sliding window and test-time mirroring), "
            "then visualize a prediction. Real inference needs a GPU, so we use an illustrative "
            "prediction built from the ground truth."
        ),
        new_code_cell(BOOTSTRAP),
        new_markdown_cell(
            "## Sliding window and test-time augmentation\n\n"
            "The model trained on fixed-size patches (for example 192 cubed), but scans are larger, "
            "so inference slides a window across the volume with overlap and blends the patches "
            "(Gaussian weighting). Test-time augmentation mirrors the input along axes and averages "
            "the predictions, which adds a small accuracy gain for extra compute.\n\n"
            "Real command (read-only):\n"
            "```bash\n"
            "nnUNetv2_predict -i imagesTs -o preds -d 999 -c 3d_fullres -tr nnUNetTrainer_warmup1e3\n"
            "```"
        ),
        new_code_cell(
            LOAD_PHANTOM
            + "from teaching_fixtures import illustrative_prediction\n"
            "pred = illustrative_prediction(gt)   # illustrative, not a real model output\n"
            "print('prediction voxels:', int((pred == 1).sum()), ' ground-truth voxels:', int((gt == 1).sum()))"
        ),
        new_markdown_cell("## Visualize the prediction against the CT"),
        new_code_cell(
            "from course_utils.viz import overlay_mask_on_slice\n"
            "z = gt.shape[2] // 2\n"
            "fig, axes = plt.subplots(1, 2, figsize=(8, 4))\n"
            "overlay_mask_on_slice(ct, gt, z=z, ax=axes[0]); axes[0].set_title('ground truth')\n"
            "overlay_mask_on_slice(ct, pred, z=z, ax=axes[1]); axes[1].set_title('prediction (illustrative)')\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## Recap\n"
            "1. Inference slides a window over the volume and blends overlapping patches.\n"
            "2. Mirroring at test time averages a few flips for a small gain.\n"
            "3. The next notebooks reuse this illustrative prediction."
        ),
    ]
    _save(new_notebook(cells=c), "04_inference.ipynb")


# --------------------------------------------------------------------------- NB05
def nb05() -> None:
    c = [
        new_markdown_cell(
            "# 05. Post-processing\n\n"
            "**Goal:** connected-component filtering, and why it is a dataset-specific heuristic "
            "rather than a default cleanup step."
        ),
        new_code_cell(BOOTSTRAP),
        new_markdown_cell(
            "## Keep large components, drop tiny ones\n\n"
            "nnU-Net's real post-processing is selected per dataset from cross-validation. Here we "
            "show the mechanic on the illustrative prediction, which has a couple of spurious blobs."
        ),
        new_code_cell(
            LOAD_PHANTOM
            + "from teaching_fixtures import illustrative_prediction\n"
            "from scipy import ndimage\n"
            "pred = illustrative_prediction(gt)\n"
            "labels, n = ndimage.label(pred == 1)\n"
            "sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, n + 1))\n"
            "print('connected components:', n, ' sizes:', sorted(map(int, sizes), reverse=True))\n\n"
            "keep = {i + 1 for i, s in enumerate(sizes) if s >= 20}   # keep components with >= 20 voxels\n"
            "clean = np.where(np.isin(labels, list(keep)), 1, 0)\n"
            "print('voxels before:', int((pred == 1).sum()), ' after filter:', int((clean == 1).sum()))"
        ),
        new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(8, 4))\n"
            "for ax, m, t in zip(axes, [pred, clean], ['before', 'after CC filter']):\n"
            "    ax.imshow(m.max(axis=2).T, origin='lower', cmap='magma')   # max projection so small blobs show\n"
            "    ax.set_title(t); ax.axis('off')\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## When it hurts\n\n"
            "Aggressive component filtering can delete real multifocal lesions, metastases, or "
            "bilateral structures. Treat it as a validated, dataset-specific heuristic: measure its "
            "effect on held-out data before trusting it.\n\n"
            "## Recap\n"
            "1. Connected-component filtering removes speckle but can erase real small lesions.\n"
            "2. It is chosen from cross-validation, not applied blindly."
        ),
    ]
    _save(new_notebook(cells=c), "05_postprocessing.ipynb")


# --------------------------------------------------------------------------- NB06
def nb06() -> None:
    c = [
        new_markdown_cell(
            "# 06. Evaluation and visualization\n\n"
            "**Goal:** Dice scoring per class with `nanmean`, absent-class handling, and "
            "prediction-versus-ground-truth overlays."
        ),
        new_code_cell(BOOTSTRAP),
        new_markdown_cell(
            "## Dice = 2 |A and B| / (|A| + |B|)\n\n"
            "A hand-built half-overlap case scores exactly 0.5."
        ),
        new_code_cell(
            "from course_utils.dsc import dice_for_label, dice_per_label, nanmean_dice\n"
            "a = np.array([1, 1, 0, 0]); b = np.array([1, 0, 1, 0])\n"
            "print('half overlap :', dice_for_label(a, b, 1))\n"
            "print('perfect      :', dice_for_label(a, a, 1))\n"
            "print('absent class :', dice_for_label(np.zeros(4, int), np.zeros(4, int), 1))"
        ),
        new_markdown_cell(
            "## Absent classes and `nanmean`\n\n"
            "If a class is in neither the prediction nor the reference, Dice is undefined (NaN). "
            "`nanmean` drops it. Ignoring absent classes is a decision that changes the reported "
            "number, so state it."
        ),
        new_code_cell(
            "scores = dice_per_label(a, b, labels=[1, 2])   # label 2 is absent\n"
            "print('per label:', scores)\n"
            "print('nanmean over {1, 2}:', nanmean_dice(scores, labels=[1, 2]), '(equals the label-1 score)')"
        ),
        new_markdown_cell("## Score the illustrative prediction on the phantom"),
        new_code_cell(
            "import json\n"
            + LOAD_PHANTOM
            + "from teaching_fixtures import illustrative_prediction\n"
            "pred = illustrative_prediction(gt)\n"
            "scores = dice_per_label(pred, gt, labels=[1])\n"
            "summary = {'metric_per_label': {str(k): v for k, v in scores.items()},\n"
            "           'foreground_mean_dice': nanmean_dice(scores, labels=[1])}\n"
            "print(json.dumps(summary, indent=2))   # same shape as nnU-Net's summary.json"
        ),
        new_code_cell(
            "from course_utils.viz import overlay_mask_on_slice\n"
            "z = gt.shape[2] // 2\n"
            "fig, axes = plt.subplots(1, 2, figsize=(8, 4))\n"
            "overlay_mask_on_slice(ct, gt, z=z, ax=axes[0]); axes[0].set_title('ground truth')\n"
            "overlay_mask_on_slice(ct, pred, z=z, ax=axes[1]); axes[1].set_title('prediction')\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## Recap\n"
            "1. Score Dice per label and aggregate with `nanmean`, dropping absent-class NaNs (a stated policy).\n"
            "2. Look at overlays, not just the number. nnU-Net writes these metrics to `summary.json`."
        ),
    ]
    _save(new_notebook(cells=c), "06_evaluation_and_visualization.ipynb")


def main() -> None:
    NB_DIR.mkdir(exist_ok=True)
    nb01(); nb02(); nb03(); nb04(); nb05(); nb06()


if __name__ == "__main__":
    main()
