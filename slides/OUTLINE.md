# Slide outline: Fine-tuning nnU-Net with Pre-trained Models (a two-phase LR schedule)

About 50 minutes of talk plus 10 minutes of Q&A, around 37 slides, in the clean
DeepLearning.AI style.

How to read this file:
- Each slide has **ON SLIDE** (the minimal text plus the one visual; keep it sparse),
  **NOTES** (what you say), and a **[m:ss]** running-time cue. Demo cues say when to switch
  to a notebook.
- This file also works as a prompt for a design tool. Paste a slide block into Plus AI or
  Canva to generate a styled version, or hand over the whole file for a full deck draft.
- Visual style: one idea per slide, about 8 words of title text, a single diagram or plot.
  Relax the word limit only for command lines, tables, and short caveats.

Accuracy guardrails (keep the slides consistent with these):
- "Two-phase" is a learning-rate schedule (linear warm-up, then polynomial decay). The
  whole network trains the entire time. No part is frozen anywhere in this talk.
- The warm-up learning rate starts near 2e-5 (not 0), reaches 1e-3 at epoch 49, switches to
  polynomial decay at epoch 50, and heads toward 0 (without reaching it) by epoch 999. Total
  epochs: 1000.
- Weight transfer: the encoder and decoder body transfer; all deep-supervision seg_layers
  are randomly re-initialized.
- The warm-up gain is small and task dependent (best on TBI, no benefit on PANTHER).
- Citations: TBI = arXiv:2504.06741; PANTHER = arXiv:2508.21775; MultiTalent = arXiv:2303.14444.

---

## Section 0: title and agenda (1 slide, about 1 min)

### Slide 1: title
- **ON SLIDE:** "Fine-tuning nnU-Net with Pre-trained Models", subtitle "A two-phase
  learning-rate schedule", your name, lab, date. Background: a CT slice with a faint lesion overlay.
- **NOTES:** Hook in one line. By the end you will know why continuing nnU-Net training on a
  pretrained model at the default settings underperforms, and how a 50-epoch learning-rate
  ramp improves it, on a real medical-imaging foundation model. Mention it is hands-on: six
  notebooks, all runnable on a laptop.
- **[0:00]**

---

## Section 1: background (6 slides, about 8 min)

### Slide 2: agenda
- **ON SLIDE:** Four items: why fine-tune, prepare data, transfer weights, the LR schedule
  that makes it work. Small icons.
- **NOTES:** Set expectations. The training-protocol section is the main one; everything else
  supports it. Running example: FLARE pan-cancer lesion segmentation from a MultiTalent-style
  CT foundation model.
- **[1:00]**

### Slide 3: the clinical task
- **ON SLIDE:** "Segment tumors in whole-body CT" plus a CT with a lesion mask. One line:
  small, variable, scarce labels.
- **NOTES:** FLARE pan-cancer Task-1: whole CT in, lesion mask out. Lesions are small and
  varied, and labeled data is scarce. This is the setting where you reach for a pretrained
  model instead of training from scratch.
- **[1:45]**

### Slide 4: why a foundation model
- **ON SLIDE:** "Reuse anatomy learned from many datasets" plus a schematic: many CT datasets
  into one pretrained encoder.
- **NOTES:** Instead of learning anatomy from your small tumor set, start from a network that
  already learned broad anatomy and pathology across many datasets. We use a MultiTalent-style
  ResEnc-L checkpoint (Zenodo 13753413): pretrained for 4000 epochs, 192-cubed patches, 1 mm
  isotropic with Z-score normalization, with a separate segmentation head per pretraining dataset.
- **[2:45]**

### Slide 5: what MultiTalent-style means
- **ON SLIDE:** Diagram: shared encoder and decoder body into N task-specific heads. Caption:
  one body, many heads. (MultiTalent: arXiv:2303.14444.)
- **NOTES:** MultiTalent trains one shared U-Net body with a separate output head per dataset,
  so the body sees many tasks. When we fine-tune on a new task we keep the body and attach a
  fresh head. That structural idea drives the architecture section.
- **[3:45]**

### Slide 6: default fine-tuning underperforms
- **ON SLIDE:** "Default fine-tuning can be sub-optimal" plus a small contrast mark. No numbers yet.
- **NOTES:** Two failure modes, both easy to miss. First, preprocessing mismatch: if you let
  nnU-Net re-plan your data with its own spacing and normalization, the pretrained encoder
  sees inputs unlike anything it trained on. Second, the default learning rate of 0.01 with no
  warm-up moves the good pretrained weights too far, too early. Neither means fine-tuning is
  broken; both are fixable recipe mistakes. We fix both.
- **[5:00]**

### Slide 7: the fix in one picture
- **ON SLIDE:** Three aligned choices: (1) match preprocessing, (2) transfer body and
  re-initialize heads, (3) two-phase LR. A simple three-row stack.
- **NOTES:** This is the whole talk on one slide. Match the pretraining preprocessing, load
  the body and re-initialize the segmentation heads, and use a learning-rate schedule that
  eases in before decaying. Each gets its own section.
- **[6:30]**

### Slide 8: what "two-stage" is not
- **ON SLIDE:** Two boxes: pretrain then fine-tune (two training jobs) versus warm-up then
  decay (the LR schedule, our topic). Arrow to the second.
- **NOTES:** "Two-stage" is overloaded. People use it for the pipeline (pretrain, then
  fine-tune, two separate jobs). In this talk the two phases are the learning-rate schedule
  inside one fine-tuning job. The whole network trains the entire time; we never freeze any
  part. Keep this distinction in mind; we return to it at the end.
- **[8:00]**

---

## Section 2: data preparation (5 slides, about 7 min). Notebook 01

### Slide 9: nnU-Net dataset format
- **ON SLIDE:** Folder tree: `imagesTr/CASE_0000.nii.gz`, `labelsTr/CASE.nii.gz`,
  `dataset.json`. Highlight the `_0000` channel suffix.
- **NOTES:** nnU-Net expects a fixed layout. Image files carry a four-digit channel suffix
  (`_0000` is the first modality); the label shares the case id with no suffix. `dataset.json`
  declares `channel_names` and `labels`. Our synthetic phantom follows this exactly.
- **DEMO CUE:** switch to NB01, cell that loads the phantom and prints `dataset.json`.
- **[8:45]**

### Slide 10: look at the data
- **ON SLIDE:** A phantom axial slice and its lesion overlay; caption `np.unique(label) = [0, 1]`.
- **NOTES:** Always look at your data. Here: one CT channel, one foreground label. In the
  notebook we view a few slices and confirm the label set. The same habit catches real
  surprises like extra labels or swapped intensities.
- **DEMO CUE:** NB01 slice view and overlay cells.
- **[9:45]**

### Slide 11: match the preprocessing
- **ON SLIDE:** "Preprocess like the pretraining did": 1 mm isotropic plus Z-score. A
  before-and-after spacing sketch.
- **NOTES:** This is the most important data step. The checkpoint was trained on 1 mm
  isotropic, Z-score normalized volumes. If you preprocess differently, the pretrained
  features do not line up, which is the usual reason default fine-tuning fails. This is
  specific to the chosen checkpoint, not a general nnU-Net rule: nnU-Net normally derives
  spacing and normalization from your dataset; here you override to match the model.
- **[11:00]**

### Slide 12: move the plans across datasets
- **ON SLIDE:** `nnUNetv2_move_plans_between_datasets -s <pretrain> -t <yours>` (read-only).
  Caption: copy the checkpoint's plans onto your data.
- **NOTES:** Mechanically, take the pretrained model's `plans.json` (it encodes architecture,
  spacing, and normalization), move it onto your dataset, and preprocess with that plans. Now
  your data and the checkpoint match. There is a cost: forced 1 mm isotropic resampling uses
  more compute and memory and can be wrong for very anisotropic scans. Match the checkpoint,
  but know why.
- **[12:30]**

### Slide 13: sanity checks
- **ON SLIDE:** Checklist: spacing is [1,1,1]? normalization is Z-score? channels match?
- **NOTES:** After preprocessing, confirm the generated plans say 1 mm isotropic and Z-score,
  not the nnU-Net defaults. A 30-second check that prevents a multi-day mis-trained run.
- **DEMO CUE:** NB01 cell that inspects the dataset and Z-score step.
- **[14:00]**

---

## Section 3: model architecture (6 slides, about 7 min). Notebook 02

### Slide 14: U-Net recap
- **ON SLIDE:** Classic U-Net encoder and decoder with skip connections.
- **NOTES:** A short refresher: the encoder compresses to semantic features, the decoder
  upsamples back to a full-resolution map, and skip connections carry spatial detail across.
  Segmentation is per-voxel classification at the output.
- **[14:45]**

### Slide 15: ResEnc-L, the backbone
- **ON SLIDE:** "Residual encoder, large" plus a note: nnU-Net's ResEncL preset.
- **NOTES:** The checkpoint is an nnU-Net ResEnc-L U-Net, a residual-encoder variant nnU-Net
  selects for large-data settings. We do not redesign it; we inherit it from the plans we
  moved over. Patch size is 192-cubed at 1 mm, which is why real training needs a large GPU,
  and why the notebooks teach the logic on a tiny model instead.
- **[16:00]**

### Slide 16: three parts, encoder, decoder body, seg heads
- **ON SLIDE:** The U-Net colored in three regions: encoder, decoder body, and the seg layers
  at the outputs.
- **NOTES:** Split the network into three parts. The encoder and decoder body are general;
  they carry transferable anatomy. The segmentation layers are task specific; they map
  features to your class set. This split decides what transfers.
- **[17:15]**

### Slide 17: deep supervision means several heads
- **ON SLIDE:** Decoder with several seg outputs at different scales, all labeled `seg_layers`.
  Caption: not one final conv, several.
- **NOTES:** A correction to a common simplification: nnU-Net uses deep supervision, so there
  are segmentation layers at several decoder resolutions, not a single final 1x1x1 conv. When
  we re-initialize the head, we mean all of them (every `.seg_layers.` parameter).
- **[18:30]**

### Slide 18: what transfers versus what re-initializes
- **ON SLIDE:** Two columns. Transfer: encoder and decoder body. Re-init: all seg_layers. A
  code stripe: `skip_strings_in_pretrained = (".seg_layers.",)`.
- **NOTES:** nnU-Net loads pretrained weights for everything except keys containing
  `.seg_layers.`, which it skips and randomly initializes. The official guidance is to load
  all layers except the segmentation layers, not only when class counts differ. One caveat:
  if your class count matches the pretraining, nnU-Net may load the heads instead;
  re-initialization follows from skipping those keys, it is not automatic.
- **DEMO CUE:** NB02 builds a small network with deep-supervision heads and prints, per
  parameter, whether it transfers or re-initializes.
- **[20:00]**

### Slide 19: why re-initialize the heads
- **ON SLIDE:** "New task, new label meaning" plus an arrow from old heads (N datasets) to one
  fresh head.
- **NOTES:** The pretraining heads predict the pretraining datasets' classes, which mean
  nothing for your lesion label. Drop them, attach a fresh head, and let fine-tuning learn the
  new mapping on top of the transferred body. This is the MultiTalent swap-the-head idea from
  Slide 5.
- **[21:00]**

---

## Section 4: training protocol, the main section (10 slides, about 15 min). Notebook 03

### Slide 20: the schedule that makes it work
- **ON SLIDE:** Title plus a faded preview of the LR curve.
- **NOTES:** This is the heart of the talk. We matched preprocessing and set up weight
  transfer. The question now is how to fine-tune so we improve the pretrained weights instead
  of damaging them. The answer is to control the learning rate over time.
- **[21:30]**

### Slide 21: the default, poly decay from 0.01
- **ON SLIDE:** A single decaying curve from 0.01 to 0. Caption: strong from scratch.
- **NOTES:** From scratch, nnU-Net uses SGD with polynomial decay starting at a learning rate
  of 0.01. That is a strong default for random initialization. On a pretrained model, starting
  at 0.01 takes large steps immediately and washes out the features you wanted to keep.
- **[22:30]**

### Slide 22: phase 1, linear warm-up
- **ON SLIDE:** A rising line, epochs 0 to 50, from about 2e-5 up to 1e-3. Formula
  `lr = max_lr / 50 * (epoch + 1)`.
- **NOTES:** Phase 1 ramps the learning rate linearly over 50 epochs, from about 2e-5 (which
  is `max_lr / 50` at epoch 0, not zero) up to the peak of 1e-3. Small early steps let the
  fresh heads and the transferred body settle together. The whole network trains; nothing is
  frozen.
- **[24:00]**

### Slide 23: phase 2, offset polynomial decay
- **ON SLIDE:** A falling curve, epochs 50 to 1000, 1e-3 to about 0. Formula
  `lr = init * (1 - t / (T - 50)) ** 0.9`.
- **NOTES:** At epoch 50 we switch to polynomial decay that restarts at the 1e-3 peak and
  decays over the remaining 950 epochs toward zero (without reaching it). "Offset" means the
  decay clock starts at epoch 50, not 0. The same SGD optimizer continues, so momentum is
  preserved across the switch.
- **[25:30]**

### Slide 24: the full two-phase curve
- **ON SLIDE:** The complete curve with a dashed line at epoch 50 marked "switch".
- **NOTES:** Put them together: ramp up, then decay, one peak at epoch 50. That is the entire
  two-phase idea, a learning-rate trajectory.
- **DEMO CUE:** NB03 drives the `course_utils` schedulers and plots this curve. It also checks
  epoch 0 is about 2e-5, epoch 49 is 1e-3, and epoch 50 is 1e-3.
- **[27:00]**

### Slide 25: how the trainer wires it
- **ON SLIDE:** Read-only snippet of `on_train_epoch_start`: epoch 0 builds the warm-up
  scheduler; epoch 50 builds the poly scheduler.
- **NOTES:** A custom trainer swaps the scheduler at the right epoch. At epoch 0 it builds the
  linear warm-up scheduler; at epoch `warmup_duration_whole_net` (50) it builds the
  offset-poly scheduler, reusing the same optimizer. There are two stage labels, `warmup_all`
  and `train`, but both train the whole network. `train` just names the post-warm-up decay
  phase; it does not mean training starts there.
- **[28:30]**

### Slide 26: the evidence, TBI Table 2
- **ON SLIDE:** A screenshot of Table 2 from the TBI paper (arXiv:2504.06741). Do not redraw
  it; paste the figure and cite it.
- **NOTES:** Read the table carefully. Pretraining with correct fine-tuning beats from-scratch
  (5-fold average Dice 53.44 to 54.21; the paper reports an improvement of up to about 2 Dice
  points in its headline setting). Among schedules, warm-up to 1e-3 (54.21) beats plain 1e-3
  (53.80), while warm-up to 1e-2 is worse (53.28). Two lessons: the schedule matters, and the
  peak learning rate matters; use 1e-3, not 1e-2.
- **[30:30]**

### Slide 27: read the table honestly
- **ON SLIDE:** Callout: warm-up to 1e-3 versus plain 1e-3 is +0.41 Dice. Small but real.
- **NOTES:** Do not oversell it. The warm-up gain over already-lowered plain 1e-3 is about
  0.4 Dice, small but real. The larger gains come from getting off the default 1e-2 and
  matching preprocessing. Being honest here is what keeps the talk credible.
- **[31:30]**

### Slide 28: contrast, PANTHER says not always
- **ON SLIDE:** Two papers side by side. TBI: warm-up helped. PANTHER (arXiv:2508.21775):
  warm-up did not beat the default. Caption: validate per task.
- **NOTES:** A related DKFZ paper, PANTHER (pancreas tumor, MRI), tried warm-up and cosine
  schedules and found they did not beat the default polynomial schedule for that task; its
  wins came from multi-stage transfer, augmentation, and ensembling. So the warm-up trick is
  task dependent, not a general law. Run the small ablation yourself. (Speaker: confirm
  PANTHER's exact schedule-ablation wording before presenting it as fact.)
- **[33:00]**

### Slide 29: see it on a real run
- **ON SLIDE:** The actual fine-tuning loss curves for the default schedule (1e-2) versus
  warm-up to 1e-3, with the epoch-50 switch marked. This plot comes from your own two runs.
- **NOTES:** The schedule only matters once you train. Run the two fine-tuning jobs, export
  the loss curves, and show the comparison. The notebook has a cell that plots this once you
  drop the exported CSV into `assets/precomputed/`. We do not fabricate a curve; the
  comparison is the point and it needs the real runs.
- **DEMO CUE:** NB03 cell that plots the real curves if the CSV is present.
- **[34:30]**

---

## Section 5: inference (3 slides, about 4 min). Notebook 04

### Slide 30: sliding-window inference
- **ON SLIDE:** A patch window tiling a large volume with a Gaussian-weighted overlap sketch.
- **NOTES:** The model trained on 192-cubed patches, but scans are larger, so inference slides
  a window across the volume and blends overlaps with Gaussian weighting into a full prediction.
- **[35:30]**

### Slide 31: test-time augmentation (mirroring)
- **ON SLIDE:** "Average predictions over mirrored flips." Small flip icons.
- **NOTES:** nnU-Net also mirrors the input along axes, predicts each, and averages, for a
  small accuracy gain. It costs compute, so it is optional at inference time.
- **[36:30]**

### Slide 32: the real command and our demo
- **ON SLIDE:** `nnUNetv2_predict -i ... -o ... -d ... -c 3d_fullres` (read-only). Caption: we
  visualize an illustrative prediction.
- **NOTES:** That is the real call on a cluster. In the notebook we do not run it; we build an
  illustrative prediction from the ground truth and visualize it against the image. Same
  learning, no GPU.
- **DEMO CUE:** NB04 builds the illustrative prediction and overlays it.
- **[37:30]**

---

## Section 6: post-processing (2 slides, about 3 min). Notebook 05

### Slide 33: connected-component filtering
- **ON SLIDE:** Before and after: speckle removed, main object kept. Caption: a heuristic, not
  a default.
- **NOTES:** A common cleanup: keep large connected components, drop tiny ones. nnU-Net's real
  post-processing is selected per dataset from cross-validation; it is not automatic.
- **[38:30]**

### Slide 34: when it hurts
- **ON SLIDE:** Warning: deletes real small or multifocal lesions.
- **NOTES:** The risk in oncology: aggressive component filtering can erase real multifocal
  lesions or bilateral structures. Treat it as a validated, dataset-specific heuristic;
  measure its effect, do not apply it blindly.
- **DEMO CUE:** NB05 runs a CC filter on the illustrative prediction and shows before and after.
- **[39:30]**

---

## Section 7: evaluation and visualization (4 slides, about 5 min). Notebook 06

### Slide 35: Dice coefficient
- **ON SLIDE:** `DSC = 2 |A and B| / (|A| + |B|)` plus an overlap diagram. Caption: 1 is
  perfect, 0 is none.
- **NOTES:** Dice measures overlap between prediction and reference. `course_utils/dsc.py`
  computes it per label; the notebook checks a hand-built case equals 0.5.
- **DEMO CUE:** NB06 runs `dice_for_label` on small arrays.
- **[40:30]**

### Slide 36: absent classes and nanmean
- **ON SLIDE:** A label table with one class absent, giving NaN, then dropped. Caption: a
  policy choice.
- **NOTES:** If a class is in neither prediction nor reference, Dice is undefined. We return
  NaN and aggregate with `nanmean`, that is, ignore classes that are not present. That is a
  policy decision and it changes your headline number, so state it explicitly in results.
- **[42:00]**

### Slide 37: prediction versus ground truth
- **ON SLIDE:** Side-by-side overlays: ground truth and prediction on a slice.
- **NOTES:** Numbers hide where the model fails. Always look. Are the misses at boundaries?
  Are whole lesions missed? The overlay helper in `course_utils/viz.py` is the first debugging tool.
- **DEMO CUE:** NB06 overlays ground truth and prediction.
- **[43:00]**

### Slide 38: reading summary.json
- **ON SLIDE:** A trimmed `summary.json` with per-case and mean Dice highlighted.
- **NOTES:** nnU-Net writes a `summary.json` after evaluation, with per-case and aggregate
  metrics. Know where the mean foreground Dice lives; that is usually your reported number.
- **[44:00]**

---

## Section 8: conclusion (3 slides, about 3 min)

### Slide 39: recap, three aligned choices
- **ON SLIDE:** The Slide 7 triad again: match preprocessing, transfer body and re-initialize
  heads, two-phase LR.
- **NOTES:** If you remember one slide, this is it. These three choices must agree for
  fine-tuning a medical foundation model to pay off.
- **[44:45]**

### Slide 40: clearing up "two-stage"
- **ON SLIDE:** "Two-phase is the LR schedule. The whole network trains throughout." Plus the
  pipeline-versus-schedule boxes from Slide 8.
- **NOTES:** Close the loop on the naming. The two phases are warm-up and decay of the learning
  rate, inside one fine-tuning job, distinct from the pretrain-then-fine-tune pipeline. No part
  of the network is ever frozen.
- **[45:30]**

### Slide 41: pitfalls and when to use this
- **ON SLIDE:** Checklist: peak LR 1e-3 not 1e-2, match preprocessing first, ablate warm-up
  per task, remember the deep-supervision heads.
- **NOTES:** When to use this: adapting a strong pretrained medical model to a new, small-data
  task. What to watch: the peak learning rate, the preprocessing match, and a quick per-task
  ablation, since the gain is not guaranteed (PANTHER). Pointers: the TBI, PANTHER, and
  MultiTalent papers, the nnU-Net fine-tuning docs, and this repo's notebooks.
- **[46:30]**

### Slide 42: thanks and Q&A
- **ON SLIDE:** "Questions?" plus the repo URL and the LR curve as a closing visual.
- **NOTES:** Point people at the GitHub repo (runnable on a laptop) and the quiz. Open the floor.
- **[47:30, with buffer to about 50:00]**

---

## Timing summary

| Section | Slides | Target |
|---|---|---|
| Title | 1 | 1 min |
| Background | 2 to 8 | 8 min |
| Data prep | 9 to 13 | 7 min |
| Architecture | 14 to 19 | 7 min |
| Training (main section) | 20 to 29 | 15 min |
| Inference | 30 to 32 | 4 min |
| Post-processing | 33 to 34 | 3 min |
| Evaluation | 35 to 38 | 5 min |
| Conclusion | 39 to 42 | 3 min |
| Total | 42 slides | about 50 min |

If the dry run runs long, trim the thinnest background and inference slides; the target band
is 35 to 42 slides.
