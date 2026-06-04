/*
 * build_deck.js -- source of truth for slides/deck.pptx.
 *
 * Generates the course deck from slides/OUTLINE.md using pptxgenjs, in the clean
 * DeepLearning.AI style (light content slides, dark title / dividers / closing).
 * Embeds the real figures built by scripts/build_figures.py and carries speaker
 * notes (with timing cues) on every slide.
 *
 * Structure (36 slides, 5 sections, with a table-of-contents and section dividers):
 *   Background | Data preparation | Model architecture | Training protocol | Conclusion
 * Inference / post-processing / evaluation are condensed into one "standard nnU-Net"
 * summary slide (they are generic nnU-Net mechanics, not the two-phase contribution).
 *
 * Palette is taken verbatim from build_figures.py so figures and drawn elements match.
 *
 * Run (pptxgenjs is installed globally):
 *   NODE_PATH="$(npm root -g)" node scripts/build_deck.js
 */

const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.join(__dirname, "..");
const FIG = (f) => path.join(ROOT, "assets", "figures", f);
const OUT = path.join(ROOT, "slides", "deck.pptx");

// ---- palette (matches scripts/build_figures.py) -------------------------------
const INK = "1F2933";   // primary text / dark backgrounds / decoder body
const INK2 = "5D7290";  // decoder slate
const ENC = "9AA9BD";   // encoder light slate (transferred)
const SEG = "E07B39";   // the one warm accent (re-init / focal)
const RED = "C0392B";   // signal: a warning
const MUTE = "6B7280";  // secondary text
const SKIP = "CDD2DA";  // faint connectors
const PAPER = "FFFFFF";
const PANEL = "F4F6F8"; // soft card fill on light slides
const LINEC = "DDE3EA"; // hairline borders
const ICE = "CADCFC";   // light text on dark slides
const DIM = "9AA6B2";   // dim text on dark slides

const FONT = "Segoe UI";
const MONO = "Consolas";

const W = 13.333, H = 7.5, M = 0.7;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Keishi Suzuki";
pres.company = "Internal mini-course";
pres.title = "Fine-tuning nnU-Net with Pre-trained Models";

// ---- helpers ------------------------------------------------------------------
const shadow = () => ({ type: "outer", color: "8A97A6", blur: 7, offset: 3, angle: 90, opacity: 0.22 });

function txt(slide, text, opts) {
  slide.addText(text, Object.assign({ fontFace: FONT, color: INK, margin: 0 }, opts));
}

function notes(slide, time, body, demo) {
  let n = `[${time}]  ${body}`;
  if (demo) n += `\n\nDEMO: ${demo}`;
  slide.addNotes(n);
}

function footer(slide, n) {
  txt(slide, "Fine-tuning nnU-Net  ·  two-phase LR schedule", {
    x: M, y: H - 0.5, w: 7, h: 0.3, fontSize: 9, color: MUTE, valign: "middle",
  });
  txt(slide, String(n), {
    x: W - M - 0.6, y: H - 0.5, w: 0.6, h: 0.3, fontSize: 9, color: MUTE,
    align: "right", valign: "middle",
  });
}

// Standard light content slide: section tag + title + footer. Returns body top y.
function scaffold(slide, tag, title, n) {
  slide.background = { color: PAPER };
  txt(slide, tag.toUpperCase(), {
    x: M, y: 0.46, w: W - 2 * M, h: 0.3, fontSize: 11, bold: true, color: SEG,
    charSpacing: 3, valign: "middle",
  });
  txt(slide, title, {
    x: M, y: 0.82, w: W - 2 * M, h: 0.85, fontSize: 28, bold: true, color: INK,
    valign: "middle",
  });
  footer(slide, n);
  return 1.95;
}

// Dark section divider slide.
function divider(slide, sec, name, subtitle, n) {
  slide.background = { color: INK };
  txt(slide, ("Section " + sec), {
    x: M, y: 2.55, w: 11.6, h: 0.4, fontSize: 13, bold: true, color: SEG, charSpacing: 4,
  });
  txt(slide, name, { x: M, y: 3.05, w: 11.6, h: 1.05, fontSize: 40, bold: true, color: "FFFFFF", valign: "top" });
  if (subtitle) txt(slide, subtitle, { x: M, y: 4.3, w: 11.6, h: 0.5, fontSize: 17, color: DIM });
  txt(slide, String(n), { x: W - M - 0.6, y: H - 0.5, w: 0.6, h: 0.3, fontSize: 9, color: DIM, align: "right", valign: "middle" });
}

function box(slide, x, y, w, h, o = {}) {
  const shape = o.radius ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE;
  const cfg = {
    x, y, w, h,
    fill: o.fill ? { color: o.fill } : { color: "FFFFFF" },
    line: o.line === null ? { type: "none" } : { color: o.line || LINEC, width: o.lineW || 1 },
  };
  if (o.radius) cfg.rectRadius = o.radius;
  if (o.shadow) cfg.shadow = shadow();
  slide.addShape(shape, cfg);
  if (o.text !== undefined) {
    txt(slide, o.text, {
      x, y, w, h, align: o.align || "center", valign: o.valign || "middle",
      fontSize: o.fontSize || 14, bold: o.bold || false,
      color: o.color || INK, fontFace: o.mono ? MONO : FONT,
      margin: o.pad !== undefined ? o.pad : 6,
    });
  }
}

function hArrow(slide, x, y, w, o = {}) {
  slide.addShape(pres.shapes.LINE, {
    x, y, w, h: 0,
    line: { color: o.color || MUTE, width: o.width || 1.5, endArrowType: "triangle", dashType: o.dash || "solid" },
  });
}
function vArrow(slide, x, y, h, o = {}) {
  slide.addShape(pres.shapes.LINE, {
    x, y, w: 0, h,
    line: { color: o.color || MUTE, width: o.width || 1.5, endArrowType: "triangle", dashType: o.dash || "solid" },
  });
}

function numCircle(slide, x, y, d, num, fill = SEG, color = "FFFFFF") {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" } });
  txt(slide, String(num), { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: d > 0.5 ? 18 : 13, bold: true, color });
}

// A triad row: number circle + bold head + body.
function triadRow(slide, x, y, w, num, head, body) {
  const d = 0.62;
  numCircle(slide, x, y, d, num);
  txt(slide, head, { x: x + d + 0.3, y: y - 0.06, w: w - d - 0.3, h: 0.42, fontSize: 19, bold: true, color: INK, valign: "middle" });
  txt(slide, body, { x: x + d + 0.3, y: y + 0.4, w: w - d - 0.3, h: 0.5, fontSize: 13.5, color: MUTE, valign: "top" });
}

// Code / command card. dark=true -> terminal look (read-only command).
function codeCard(slide, x, y, w, h, lines, o = {}) {
  const dark = o.dark !== false;
  box(slide, x, y, w, h, { fill: dark ? INK : PANEL, line: dark ? null : LINEC, radius: 0.06, shadow: !!o.shadow });
  if (o.label) {
    txt(slide, o.label.toUpperCase(), {
      x: x + 0.18, y: y - 0.34, w: w - 0.36, h: 0.3, fontSize: 9.5, bold: true,
      color: MUTE, charSpacing: 2,
    });
  }
  const runs = lines.map((ln) => {
    const t = typeof ln === "string" ? { text: ln } : ln;
    return {
      text: t.text,
      options: Object.assign(
        { fontFace: MONO, fontSize: o.fontSize || 13, color: t.color || (dark ? ICE : INK), breakLine: true },
        t.options || {}
      ),
    };
  });
  slide.addText(runs, { x: x + 0.25, y: y + 0.18, w: w - 0.5, h: h - 0.36, valign: "top", margin: 0, lineSpacingMultiple: 1.12 });
}

function bullets(slide, x, y, w, h, items, o = {}) {
  const runs = items.map((it) => ({
    text: typeof it === "string" ? it : it.text,
    options: Object.assign(
      { bullet: { code: "2022", indent: 16 }, color: o.color || INK, fontSize: o.fontSize || 15, breakLine: true, paraSpaceAfter: 10 },
      (typeof it === "object" && it.options) || {}
    ),
  }));
  slide.addText(runs, { x, y, w, h, fontFace: FONT, valign: "top", margin: 0 });
}

function lead(slide, x, y, w, text, o = {}) {
  txt(slide, text, Object.assign({ x, y, w, h: o.h || 0.9, fontSize: o.fontSize || 20, bold: true, color: INK, valign: "top" }, o));
}

function caption(slide, x, y, w, text, o = {}) {
  txt(slide, text, Object.assign({ x, y, w, h: 0.4, fontSize: 11, italic: true, color: MUTE, align: o.align || "center" }, o));
}

function scanViewer(slide, x, y, size, img, cap) {
  box(slide, x, y, size, size, { fill: "000000", line: INK2, lineW: 1, radius: 0.05, shadow: true });
  slide.addImage({ path: FIG(img), x: x + 0.12, y: y + 0.12, w: size - 0.24, h: size - 0.24 });
  if (cap) caption(slide, x, y + size + 0.07, size, cap);
}

function figFit(slide, img, dims, bx, by, bw, maxH, o = {}) {
  const [ow, oh] = dims;
  let w = bw, h = w * (oh / ow);
  if (h > maxH) { h = maxH; w = h * (ow / oh); }
  const x = bx + (bw - w) / 2;
  if (o.shadow || o.frame) {
    box(slide, x - 0.12, by - 0.12, w + 0.24, h + 0.24, { fill: "FFFFFF", line: o.frame ? LINEC : null, radius: 0.04, shadow: !!o.shadow });
  }
  slide.addImage({ path: FIG(img), x, y: by, w, h, transparency: o.transparency || 0 });
  return { x, y: by, w, h };
}

const D_LR = [1777, 899];
const D_ARCH = [1745, 1089];

// ============================================================================
// 1 -- title
// ============================================================================
{
  const s = pres.addSlide();
  s.background = { color: INK };
  txt(s, "INTERNAL MINI-COURSE  ·  DEEP LEARNING FOR MEDICAL IMAGING", {
    x: M, y: 1.2, w: 7.6, h: 0.35, fontSize: 12, bold: true, color: SEG, charSpacing: 2.5,
  });
  txt(s, "Fine-tuning nnU-Net with Pre-trained Models", {
    x: M, y: 1.75, w: 7.7, h: 1.9, fontSize: 40, bold: true, color: "FFFFFF", valign: "top", lineSpacingMultiple: 1.02,
  });
  txt(s, "A two-phase learning-rate schedule", { x: M, y: 3.95, w: 7.6, h: 0.6, fontSize: 22, color: ICE });
  txt(s, "Keishi Suzuki      ·      University of Toronto      ·      June 2026", { x: M, y: 4.95, w: 7.6, h: 0.4, fontSize: 15, color: DIM });
  txt(s, "Six laptop-runnable notebooks  ·  public GitHub repo", { x: M, y: 5.45, w: 7.6, h: 0.4, fontSize: 12.5, color: DIM });
  scanViewer(s, 9.0, 1.7, 3.5, "ct_overlay.png", "synthetic CT phantom  ·  lesion label");
  notes(s, "0:00",
    "Hook in one line. By the end you will know why continuing nnU-Net training on a pretrained model at the default settings underperforms, and how a 50-epoch learning-rate ramp improves it, on a real medical-imaging foundation model. Mention it is hands-on: six notebooks, all runnable on a laptop.");
}

// ============================================================================
// 2 -- table of contents
// ============================================================================
{
  const s = pres.addSlide();
  scaffold(s, "Overview", "What we'll cover", 2);
  const secs = [
    ["Background", "why fine-tune, and the fix in one picture"],
    ["Data preparation", "match the checkpoint's preprocessing"],
    ["Model architecture", "transfer the body, re-initialize the heads"],
    ["Training protocol", "the two-phase learning-rate schedule"],
    ["Conclusion", "recap, pitfalls, when to use it"],
  ];
  let y = 2.2;
  secs.forEach((it, i) => {
    const main = i === 3;
    numCircle(s, M, y, 0.58, i + 1, main ? SEG : INK);
    txt(s, it[0], { x: M + 0.9, y: y - 0.02, w: 8.5, h: 0.42, fontSize: 20, bold: true, color: main ? SEG : INK, valign: "middle" });
    txt(s, it[1] + (main ? "   (the main section)" : ""), { x: M + 0.9, y: y + 0.42, w: 10.0, h: 0.35, fontSize: 13.5, color: MUTE });
    y += 0.95;
  });
  notes(s, "0:45",
    "Five sections. Background sets up why and the fix in one picture; then the three choices get a section each: data preparation, model architecture, and the training protocol. Training is the main section, the two-phase learning-rate schedule. We close with a recap and pitfalls.");
}

// ============================================================================
// SECTION 1: BACKGROUND
// ============================================================================
{
  const s = pres.addSlide();
  divider(s, "1", "Background", "Why fine-tune a foundation model, and the fix in one picture", 3);
  notes(s, "1:15", "Set the running example: FLARE pan-cancer lesion segmentation from a MultiTalent-style CT foundation model. This section ends with the whole talk on one slide.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Background", "The clinical task", 4);
  lead(s, M, 2.2, 6.0, "Segment tumors in whole-body CT");
  bullets(s, M, 3.2, 6.0, 3.0, [
    "Lesions are small and variable in shape",
    "Labeled data is scarce",
    "So you reach for a pretrained model, not training from scratch",
  ]);
  scanViewer(s, 8.7, 2.1, 3.7, "ct_overlay.png", "FLARE pan-cancer Task-1  ·  lesion mask out");
  notes(s, "1:45",
    "FLARE pan-cancer Task-1: whole CT in, lesion mask out. Lesions are small and varied, and labeled data is scarce. This is the setting where you reach for a pretrained model instead of training from scratch.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Background", "Why a foundation model", 5);
  lead(s, M, 2.0, 11.9, "Reuse anatomy learned from many datasets");
  const dy = [2.95, 3.85, 4.75];
  const labels = ["CT dataset A", "CT dataset B", "CT dataset C"];
  dy.forEach((y, i) => box(s, M, y, 2.7, 0.7, { fill: PANEL, line: LINEC, radius: 0.08, text: labels[i], fontSize: 14, color: INK }));
  dy.forEach((y) => hArrow(s, M + 2.7, y + 0.35, 1.7, { color: ENC, width: 2 }));
  box(s, 5.6, 3.15, 3.1, 2.1, { fill: INK, line: null, radius: 0.08, text: "Pretrained\nResEnc-L encoder", fontSize: 16, bold: true, color: "FFFFFF" });
  hArrow(s, 8.7, 4.2, 1.0, { color: SEG, width: 2.5 });
  box(s, 9.8, 3.55, 2.8, 1.3, { fill: "FFFFFF", line: SEG, lineW: 1.5, radius: 0.08, text: "Your new task\n(small tumor set)", fontSize: 13.5, color: INK });
  caption(s, M, 5.75, 11.9, "MultiTalent-style ResEnc-L  ·  Zenodo 13753413  ·  4000 epochs  ·  192-cubed patches  ·  1 mm isotropic + Z-score", { align: "left" });
  notes(s, "2:45",
    "Instead of learning anatomy from your small tumor set, start from a network that already learned broad anatomy and pathology across many datasets. We use a MultiTalent-style ResEnc-L checkpoint (Zenodo 13753413): pretrained for 4000 epochs, 192-cubed patches, 1 mm isotropic with Z-score normalization, with a separate segmentation head per pretraining dataset.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Background", "What MultiTalent-style means", 6);
  lead(s, M, 2.0, 7.0, "One body, many segmentation heads");
  box(s, M, 3.0, 3.6, 1.6, { fill: INK, line: null, radius: 0.08, text: "Shared encoder\n+ decoder body", fontSize: 16, bold: true, color: "FFFFFF" });
  const hy = [2.5, 3.55, 4.6];
  const hl = ["seg head: dataset 1", "seg head: dataset 2", "seg head: dataset N"];
  hy.forEach((y, i) => box(s, M + 4.95, y, 2.55, 0.78, { fill: "FFFFFF", line: SEG, lineW: 1.5, radius: 0.08, text: hl[i], fontSize: 12.5, color: INK }));
  const jx = M + 4.5;
  hLine(s, M + 3.6, 3.8, jx, { color: SEG, width: 2 });
  vLine(s, jx, hy[0] + 0.39, hy[2] + 0.39, { color: SEG, width: 2 });
  hy.forEach((y) => hArrow(s, jx, y + 0.39, M + 4.95 - jx, { color: SEG, width: 2 }));
  bullets(s, 8.55, 2.95, 4.05, 2.4, [
    "Each dataset has its own label set, so each gets its own segmentation head",
    "The shared body must learn anatomy useful across all of them",
    "Fine-tune: drop the heads, keep the body, attach one fresh head",
  ], { fontSize: 13.5 });
  caption(s, M, 5.85, 7.5, "MultiTalent: arXiv:2303.14444", { align: "left" });
  notes(s, "3:45",
    "MultiTalent trains one shared encoder and decoder body, with a separate segmentation output head per pretraining dataset. Why a head each? Each dataset labels different structures, so they cannot share one output layer. Forcing one body to serve all of them pushes it to learn anatomy that generalizes. When we fine-tune on a new task we drop those heads, keep the body, and attach one fresh head. That swap-the-head idea drives the architecture section.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Background", "Default fine-tuning can be sub-optimal", 7);
  box(s, M, 2.5, 5.6, 2.5, { fill: PANEL, line: LINEC, radius: 0.08, shadow: true });
  txt(s, "1.  Preprocessing mismatch", { x: M + 0.4, y: 2.8, w: 4.8, h: 0.5, fontSize: 18, bold: true, color: INK });
  txt(s, "Let nnU-Net re-plan with its own spacing and the pretrained encoder sees inputs unlike its training data.", { x: M + 0.4, y: 3.35, w: 4.9, h: 1.4, fontSize: 14, color: MUTE, valign: "top" });
  box(s, 7.0, 2.5, 5.6, 2.5, { fill: PANEL, line: LINEC, radius: 0.08, shadow: true });
  txt(s, "2.  Learning rate 0.01, no warm-up", { x: 7.4, y: 2.8, w: 4.8, h: 0.5, fontSize: 18, bold: true, color: INK });
  txt(s, "The default LR moves the good pretrained weights too far, too early in fine-tuning.", { x: 7.4, y: 3.35, w: 4.9, h: 1.4, fontSize: 14, color: MUTE, valign: "top" });
  txt(s, "Neither means fine-tuning is broken. Both are fixable recipe mistakes, and we fix both.", { x: M, y: 5.3, w: 11.6, h: 0.6, fontSize: 16, italic: true, color: INK, align: "center" });
  notes(s, "5:00",
    "Two failure modes, both easy to miss. First, preprocessing mismatch: if you let nnU-Net re-plan your data with its own spacing and normalization, the pretrained encoder sees inputs unlike anything it trained on. Second, the default learning rate of 0.01 with no warm-up moves the good pretrained weights too far, too early. Neither means fine-tuning is broken; both are fixable recipe mistakes. We fix both.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Background", "The fix in one picture", 8);
  txt(s, "This is the whole talk on one slide.", { x: M, y: 1.95, w: 11.6, h: 0.4, fontSize: 15, italic: true, color: MUTE });
  const y = 2.7;
  triadRow(s, M + 0.4, y, 11.0, 1, "Match the preprocessing", "Resample and normalize the way the pretraining did (1 mm isotropic + Z-score).");
  triadRow(s, M + 0.4, y + 1.35, 11.0, 2, "Transfer the body, re-initialize the heads", "Load encoder + decoder body; randomly re-init all deep-supervision seg_layers.");
  triadRow(s, M + 0.4, y + 2.7, 11.0, 3, "Two-phase learning-rate schedule", "Ease in with a linear warm-up, then decay. The whole network trains throughout.");
  notes(s, "6:30",
    "This is the whole talk on one slide. Match the pretraining preprocessing, load the body and re-initialize the segmentation heads, and use a learning-rate schedule that eases in before decaying. Each gets its own section.");
}

// ============================================================================
// SECTION 2: DATA PREPARATION
// ============================================================================
{
  const s = pres.addSlide();
  divider(s, "2", "Data preparation", "Match the checkpoint's preprocessing  ·  Notebook 01", 9);
  notes(s, "7:15", "The first of the three choices. The load-bearing idea: preprocess your data the way the pretrained checkpoint was preprocessed, not the way nnU-Net would choose by default.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Data preparation  ·  Notebook 01", "nnU-Net dataset format", 10);
  codeCard(s, M, 2.3, 7.6, 3.0, [
    { text: "Dataset999_Phantom/" },
    { text: "├── imagesTr/" },
    { text: "│   └── PHANTOM_001", options: { breakLine: false } },
    { text: "_0000", options: { color: SEG, bold: true, breakLine: false } },
    { text: ".nii.gz" },
    { text: "├── labelsTr/" },
    { text: "│   └── PHANTOM_001.nii.gz" },
    { text: "└── dataset.json" },
  ], { dark: true, fontSize: 15, label: "nnU-Net raw layout" });
  bullets(s, 8.7, 2.4, 3.9, 3.0, [
    "_0000 is the channel/modality suffix (first modality)",
    "The label shares the case id, no suffix",
    "dataset.json declares channel_names and labels",
  ], { fontSize: 14 });
  caption(s, M, 5.5, 7.6, "Our synthetic phantom follows this layout exactly.", { align: "left" });
  notes(s, "7:30",
    "nnU-Net expects a fixed layout. Image files carry a four-digit channel suffix (_0000 is the first modality); the label shares the case id with no suffix. dataset.json declares channel_names and labels. Our synthetic phantom follows this exactly.",
    "switch to NB01, cell that loads the phantom and prints dataset.json.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Data preparation  ·  Notebook 01", "Match the preprocessing", 11);
  lead(s, M, 2.2, 6.2, "Preprocess like the pretraining did");
  txt(s, "1 mm isotropic   +   Z-score normalization", { x: M, y: 3.05, w: 6.2, h: 0.5, fontSize: 18, bold: true, color: SEG });
  bullets(s, M, 3.75, 6.2, 2.0, [
    "The checkpoint trained on 1 mm isotropic, Z-score volumes",
    "Preprocess differently and the pretrained features do not line up",
    "Checkpoint-specific, not a universal nnU-Net rule",
  ], { fontSize: 14 });
  txt(s, "your spacing", { x: 8.0, y: 2.35, w: 2.0, h: 0.3, fontSize: 12, color: MUTE, align: "center" });
  drawGrid(s, 8.0, 2.7, 2.0, 2.0, 3, ENC);
  hArrow(s, 10.15, 3.7, 0.75, { color: SEG, width: 2.5 });
  txt(s, "1 mm isotropic", { x: 10.6, y: 2.35, w: 2.0, h: 0.3, fontSize: 12, color: MUTE, align: "center" });
  drawGrid(s, 10.6, 2.7, 2.0, 2.0, 6, INK2);
  caption(s, 7.9, 5.0, 4.7, "resample so voxels match the checkpoint");
  txt(s, "Why these two? The next two slides.", { x: M, y: 5.95, w: 11.6, h: 0.4, fontSize: 14, italic: true, color: MUTE });
  notes(s, "8:30",
    "The most important data step. The checkpoint was trained on 1 mm isotropic, Z-score normalized volumes. If you preprocess differently, the pretrained features do not line up, which is the usual reason default fine-tuning fails. This is specific to the chosen checkpoint, not a general nnU-Net rule: nnU-Net normally derives spacing and normalization from your data; here you override to match the model. The next two slides say why each of the two matters.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Data preparation  ·  Notebook 01", "Why 1 mm isotropic", 12);
  lead(s, M, 2.2, 6.2, "Filters expect a fixed physical scale");
  bullets(s, M, 3.15, 6.2, 2.8, [
    "The encoder's 3D conv filters learned anatomy at 1 mm per voxel",
    "At a different spacing, the same structure spans a different number of voxels, so the filters fire at the wrong scale",
    "Isotropic means equal spacing on all three axes, matching how the pretraining sampled 3D context",
    { text: "Cost: forcing 1 mm can heavily upsample thick-slice scans (compute and memory) and is not ideal for very anisotropic data", options: { color: RED } },
  ], { fontSize: 14 });
  // same physical structure on a coarse vs a fine grid
  txt(s, "coarse spacing", { x: 7.7, y: 2.4, w: 2.1, h: 0.3, fontSize: 12, color: MUTE, align: "center" });
  drawGrid(s, 7.7, 2.75, 2.1, 2.1, 3, ENC);
  s.addShape(pres.shapes.OVAL, { x: 7.7 + 0.7, y: 2.75 + 0.7, w: 0.7, h: 0.7, fill: { color: SEG }, line: { type: "none" } });
  hArrow(s, 9.95, 3.8, 0.6, { color: SEG, width: 2.5 });
  txt(s, "1 mm isotropic", { x: 10.45, y: 2.4, w: 2.1, h: 0.3, fontSize: 12, color: MUTE, align: "center" });
  drawGrid(s, 10.45, 2.75, 2.1, 2.1, 6, INK2);
  s.addShape(pres.shapes.OVAL, { x: 10.45 + 0.7, y: 2.75 + 0.7, w: 0.7, h: 0.7, fill: { color: SEG }, line: { type: "none" } });
  caption(s, 7.6, 5.05, 5.0, "same structure, more voxels: the filters see the scale they trained on");
  notes(s, "9:45",
    "Convolutional filters are tied to a physical scale. The pretrained encoder learned its 3D filters on volumes sampled at 1 mm per voxel. If your data has a different spacing, the same anatomical structure covers a different number of voxels, so the learned filters respond at the wrong scale. Isotropic means the spacing is equal on all three axes, which is how the pretraining sampled 3D context. The cost is real: forcing 1 mm can heavily upsample thick-slice scans, using more compute and memory, and it is a poor fit for very anisotropic data. You match the checkpoint, but you know the trade-off.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Data preparation  ·  Notebook 01", "Why Z-score normalization", 13);
  lead(s, M, 2.2, 6.2, "Weights expect a fixed intensity distribution");
  bullets(s, M, 3.15, 6.2, 2.8, [
    "The network trained on inputs at roughly zero mean, unit variance",
    "Its early-layer weights assume that input distribution",
    "Feed raw HU or a different scaling and every activation shifts, pushing features outside the learned regime",
    "Z-score reproduces the pretraining's intensity statistics",
  ], { fontSize: 14 });
  box(s, 7.7, 2.75, 2.2, 1.3, { fill: PANEL, line: LINEC, radius: 0.08, text: "raw intensities\nmean and scale vary\n(scanner, protocol)", fontSize: 12, color: INK });
  hArrow(s, 9.95, 3.4, 0.6, { color: SEG, width: 2.5 });
  box(s, 10.45, 2.75, 2.2, 1.3, { fill: "FFFFFF", line: SEG, lineW: 1.5, radius: 0.08, text: "Z-scored\nmean 0, std 1\n(matches pretraining)", fontSize: 12, color: INK });
  codeCard(s, 7.7, 4.45, 4.95, 0.65, [{ text: "x = (x - mean) / std", options: { color: SEG } }], { dark: false, fontSize: 14 });
  notes(s, "11:00",
    "Normalization matters for the same reason. The network was trained on inputs normalized to roughly zero mean and unit variance, and its weights, especially in the early layers, are calibrated to that distribution. If you feed raw Hounsfield units or a different scaling, every downstream activation is shifted and scaled away from what the pretrained weights expect, and the features degrade. Z-score normalization, subtract the mean and divide by the standard deviation, reproduces the intensity statistics the pretraining used.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Data preparation  ·  Notebook 01", "Move the plans across datasets", 14);
  codeCard(s, M, 2.4, 11.6, 1.05, [
    { text: "nnUNetv2_move_plans_between_datasets", options: { color: ICE } },
    { text: "    -s <pretrain_dataset>  -t <your_dataset>", options: { color: SEG } },
  ], { dark: true, fontSize: 15, label: "real command (read-only)" });
  bullets(s, M, 3.9, 11.6, 2.4, [
    "The pretrained plans.json encodes architecture, spacing, and normalization",
    "Move it onto your dataset, then preprocess with that plans, and now data and checkpoint match",
    { text: "Cost: forced 1 mm isotropic resampling uses more compute and memory, and can be wrong for very anisotropic scans", options: { color: RED } },
  ], { fontSize: 15 });
  notes(s, "12:15",
    "Mechanically, take the pretrained model's plans.json, which encodes architecture, spacing, and normalization, move it onto your dataset, and preprocess with that plans. Now your data and the checkpoint match on both of the things we just discussed.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Data preparation  ·  Notebook 01", "Sanity checks", 15);
  txt(s, "After preprocessing, confirm the generated plans, not the nnU-Net defaults:", { x: M, y: 2.15, w: 11.6, h: 0.5, fontSize: 16, color: INK });
  const checks = ["spacing is [1, 1, 1] ?", "normalization is Z-score ?", "channels match the checkpoint ?"];
  let y = 2.95;
  checks.forEach((c) => {
    numCircle(s, M + 0.2, y, 0.5, "✓", SEG);
    txt(s, c, { x: M + 0.95, y, w: 10.0, h: 0.5, fontSize: 18, color: INK, valign: "middle", fontFace: MONO });
    y += 0.85;
  });
  txt(s, "A 30-second check that prevents a multi-day mis-trained run.", { x: M, y: 5.75, w: 11.6, h: 0.5, fontSize: 15, italic: true, color: MUTE });
  notes(s, "13:30",
    "After preprocessing, confirm the generated plans say 1 mm isotropic and Z-score, not the nnU-Net defaults. A 30-second check that prevents a multi-day mis-trained run.",
    "NB01 cell that inspects the dataset and Z-score step.");
}

// ============================================================================
// SECTION 3: MODEL ARCHITECTURE
// ============================================================================
{
  const s = pres.addSlide();
  divider(s, "3", "Model architecture", "Transfer the body, re-initialize the heads  ·  Notebook 02", 16);
  notes(s, "14:15", "The second choice. Which parts of the pretrained network do you keep, and which do you throw away and relearn?");
}

{
  const s = pres.addSlide();
  scaffold(s, "Model architecture  ·  Notebook 02", "U-Net recap", 17);
  drawUNet(s, M + 1.3, 2.4, 9.0);
  bullets(s, M, 5.35, 12.0, 0.9, [
    "Encoder compresses to semantic features; decoder upsamples back to full resolution; skips carry spatial detail across",
  ], { fontSize: 15 });
  caption(s, M, 6.1, 11.9, "Segmentation is per-voxel classification at the output.", { align: "left" });
  notes(s, "14:30",
    "A short refresher: the encoder compresses to semantic features, the decoder upsamples back to a full-resolution map, and skip connections carry spatial detail across. Segmentation is per-voxel classification at the output.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Model architecture  ·  Notebook 02", "ResEnc-L, the backbone", 18);
  lead(s, M, 2.15, 11.6, "Residual encoder, large preset");
  bullets(s, M, 3.05, 11.8, 2.9, [
    "Vanilla nnU-Net uses plain conv blocks; ResEnc adds residual blocks in the encoder, so it trains deeper and more stably",
    "The design is encoder-heavy: most of the capacity sits in the downsampling path",
    "\"L\" is the large preset: more features and blocks, sized for large GPUs and big datasets (M / L / XL scale with VRAM)",
    "We inherit it from the moved plans; patch size 192-cubed at 1 mm needs a large GPU, so the notebooks teach the logic on a tiny model",
  ], { fontSize: 15 });
  notes(s, "15:45",
    "The checkpoint is an nnU-Net ResEnc-L U-Net. What is different from the vanilla nnU-Net U-Net? Vanilla uses plain convolution blocks; ResEnc adds residual connections in the encoder, which let it go deeper and train more stably. The design is deliberately encoder-heavy, with most of the capacity in the downsampling path. And L is the large preset: the ResEnc family comes in M, L, and XL, scaling features and blocks to the available GPU memory; L is sized for large GPUs and large multi-dataset training, which is exactly the pretraining setting. We inherit all of this from the plans we moved over. The 192-cubed patch at 1 mm is why real training needs a large GPU, and why the notebooks teach the logic on a tiny model.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Model architecture  ·  Notebook 02", "Encoder, decoder body, seg heads", 19);
  txt(s, "Split the network into three parts. This split decides what transfers.", { x: M, y: 2.05, w: 11.6, h: 0.5, fontSize: 16, color: INK });
  const cards = [
    [ENC, "Encoder", "Compresses input to features. General anatomy.", "FFFFFF"],
    [INK2, "Decoder body", "Upsamples back to resolution. General.", "FFFFFF"],
    [SEG, "Seg layers (heads)", "Map features to your class set. Task-specific.", "FFFFFF"],
  ];
  let x = M;
  cards.forEach((c) => {
    box(s, x, 2.85, 3.73, 2.4, { fill: c[0], line: null, radius: 0.08, shadow: true });
    txt(s, c[1], { x: x + 0.3, y: 3.1, w: 3.1, h: 0.5, fontSize: 19, bold: true, color: c[3] });
    txt(s, c[2], { x: x + 0.3, y: 3.7, w: 3.2, h: 1.3, fontSize: 14, color: c[3], valign: "top" });
    x += 3.93;
  });
  txt(s, "Encoder and decoder body carry transferable anatomy. The seg layers are task-specific.", { x: M, y: 5.6, w: 11.6, h: 0.5, fontSize: 14.5, italic: true, color: MUTE, align: "center" });
  notes(s, "17:00",
    "Split the network into three parts. The encoder and decoder body are general; they carry transferable anatomy. The segmentation layers are task specific; they map features to your class set. This split decides what transfers.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Model architecture  ·  Notebook 02", "What transfers versus what re-initializes", 20);
  figFit(s, "architecture_transfer.png", D_ARCH, M, 1.95, 8.4, 4.6, { frame: true });
  bullets(s, 9.3, 2.05, 3.35, 3.4, [
    "Load pretrained weights for every key except those containing .seg_layers.",
    "Deep supervision puts seg_layers at several decoder scales; the substring match skips all of them, so every head re-inits",
    { text: "Caveat: if class counts match, nnU-Net may load the heads instead. Re-init follows from skipping the keys, it is not automatic", options: { color: RED } },
  ], { fontSize: 13 });
  notes(s, "18:15",
    "This is the figure to dwell on. nnU-Net loads pretrained weights for everything except keys whose name contains the substring .seg_layers. Because nnU-Net uses deep supervision, there are segmentation layers at several decoder scales, not a single final 1x1x1 conv, and the substring match skips all of them, so every head is randomly re-initialized. The official guidance is to load all layers except the segmentation layers. One caveat: if your class count matches the pretraining, nnU-Net may load the heads instead; the re-initialization follows from skipping those keys, it is not automatic.",
    "NB02 builds a small network with deep-supervision heads and prints, per parameter, whether it transfers or re-initializes.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Model architecture  ·  Notebook 02", "Why re-initialize the heads", 21);
  lead(s, M, 2.2, 11.6, "New task, new label meaning");
  box(s, M, 3.2, 4.2, 1.5, { fill: PANEL, line: LINEC, radius: 0.08, text: "Old heads\n(N pretraining datasets)", fontSize: 15, color: MUTE });
  hArrow(s, M + 4.4, 3.95, 1.5, { color: SEG, width: 3 });
  box(s, M + 6.1, 3.2, 4.2, 1.5, { fill: "FFFFFF", line: SEG, lineW: 2, radius: 0.08, text: "One fresh head\n(your lesion label)", fontSize: 15, bold: true, color: INK });
  txt(s, "Drop the pretraining heads, attach a fresh head, and let fine-tuning learn the new mapping on the transferred body. This is the MultiTalent swap-the-head idea from earlier.", { x: M, y: 5.1, w: 11.0, h: 1.0, fontSize: 15, italic: true, color: MUTE, valign: "top" });
  notes(s, "20:00",
    "The pretraining heads predict the pretraining datasets' classes, which mean nothing for your lesion label. Drop them, attach a fresh head, and let fine-tuning learn the new mapping on top of the transferred body. This is the swap-the-head idea from the MultiTalent slide.");
}

// ============================================================================
// SECTION 4: TRAINING PROTOCOL (main)
// ============================================================================
{
  const s = pres.addSlide();
  divider(s, "4", "Training protocol", "The two-phase learning-rate schedule  ·  the main section  ·  Notebook 03", 22);
  notes(s, "21:00", "The heart of the talk. We matched preprocessing and set up weight transfer. Now: how do we fine-tune so we improve the pretrained weights instead of damaging them?");
}

{
  const s = pres.addSlide();
  scaffold(s, "Training protocol  ·  Notebook 03", "The schedule that makes it work", 23);
  const f = figFit(s, "lr_schedule.png", D_LR, M, 2.2, 11.6, 3.6);
  s.addShape(pres.shapes.RECTANGLE, { x: f.x, y: f.y, w: f.w, h: f.h, fill: { color: "FFFFFF", transparency: 38 }, line: { type: "none" } });
  txt(s, "Control the learning rate over time.", { x: M, y: 5.9, w: 11.6, h: 0.5, fontSize: 18, bold: true, color: INK, align: "center" });
  notes(s, "21:15",
    "We matched preprocessing and set up weight transfer. The question now is how to fine-tune so we improve the pretrained weights instead of damaging them. The answer is to control the learning rate over time.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Training protocol  ·  Notebook 03", "The default: poly decay from 0.01", 24);
  const L = [], V = [];
  for (let e = 0; e <= 1000; e += 100) { L.push(String(e)); V.push(0.01 * Math.pow(1 - e / 1000, 0.9)); }
  lrChart(s, 6.5, 2.2, 6.1, 3.7, L, V, RED, "0.000");
  lead(s, M, 2.5, 5.4, "Strong from scratch");
  bullets(s, M, 3.4, 5.4, 2.6, [
    "From scratch, nnU-Net uses SGD with polynomial decay starting at 0.01",
    "A strong default for random initialization",
    { text: "On a pretrained model, starting at 0.01 takes large steps immediately and washes out the features you wanted to keep", options: { color: RED } },
  ], { fontSize: 14 });
  notes(s, "22:15",
    "From scratch, nnU-Net uses SGD with polynomial decay starting at a learning rate of 0.01. That is a strong default for random initialization. On a pretrained model, starting at 0.01 takes large steps immediately and washes out the features you wanted to keep.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Training protocol  ·  Notebook 03", "Phase 1: linear warm-up", 25);
  const L = [], V = [];
  for (let e = 0; e <= 50; e += 5) { L.push(String(e)); V.push((1e-3 / 50) * (e + 1)); }
  lrChart(s, 6.5, 2.2, 6.1, 3.7, L, V, INK, "0.0000");
  lead(s, M, 2.6, 5.4, "Ramp from ~2e-5 to 1e-3");
  codeCard(s, M, 3.55, 5.4, 0.62, [{ text: "lr = max_lr / 50 * (epoch + 1)", options: { color: SEG } }], { dark: false, fontSize: 13 });
  bullets(s, M, 4.45, 5.4, 1.6, [
    "Epoch 0 is about 2e-5 (max_lr / 50), not zero",
    "Small early steps let the fresh heads and the body settle together",
  ], { fontSize: 14 });
  notes(s, "23:30",
    "Phase 1 ramps the learning rate linearly over 50 epochs, from about 2e-5, which is max_lr over 50 at epoch 0, not zero, up to the peak of 1e-3. Small early steps let the fresh heads and the transferred body settle together.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Training protocol  ·  Notebook 03", "Phase 2: offset polynomial decay", 26);
  const L = [], V = [];
  for (let e = 50; e <= 950; e += 100) { L.push(String(e)); V.push(1e-3 * Math.pow(1 - (e - 50) / 950, 0.9)); }
  lrChart(s, 6.5, 2.2, 6.1, 3.7, L, V, INK, "0.0000");
  lead(s, M, 2.5, 5.4, "Decay 1e-3 toward 0");
  codeCard(s, M, 3.45, 5.4, 0.62, [{ text: "lr = init * (1 - t/(T-50))**0.9", options: { color: SEG } }], { dark: false, fontSize: 13 });
  bullets(s, M, 4.3, 5.4, 1.8, [
    "At epoch 50, switch to poly decay restarting at the 1e-3 peak",
    '"Offset" means the decay clock starts at epoch 50, not 0',
    "Same SGD optimizer continues, so momentum is preserved",
  ], { fontSize: 13.5 });
  notes(s, "25:00",
    'At epoch 50 we switch to polynomial decay that restarts at the 1e-3 peak and decays over the remaining 950 epochs toward zero, without reaching it. "Offset" means the decay clock starts at epoch 50, not 0. The same SGD optimizer continues, so momentum is preserved across the switch.');
}

{
  const s = pres.addSlide();
  scaffold(s, "Training protocol  ·  Notebook 03", "The full two-phase curve", 27);
  figFit(s, "lr_schedule.png", D_LR, M, 2.05, 11.6, 4.0, { frame: true });
  caption(s, M, 6.25, 11.6, "Ramp up, then decay, one peak at epoch 50. That is the entire two-phase idea.");
  notes(s, "26:30",
    "Put them together: ramp up, then decay, one peak at epoch 50. That is the entire two-phase idea, a learning-rate trajectory.",
    "NB03 drives the course_utils schedulers and plots this curve. It also checks epoch 0 is about 2e-5, epoch 49 is 1e-3, and epoch 50 is 1e-3.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Training protocol  ·  Notebook 03", "How the trainer wires it", 28);
  codeCard(s, M, 2.45, 7.7, 2.7, [
    { text: "def on_train_epoch_start(self):" },
    { text: "    if self.current_epoch == 0:" },
    { text: "        # build linear warm-up scheduler", options: { color: DIM } },
    { text: "        self.lr_scheduler = LinearWarmup(peak=1e-3, dur=50)", options: { color: SEG } },
    { text: "    elif self.current_epoch == warmup_duration_whole_net:  # 50", options: { color: ICE } },
    { text: "        # build offset-poly scheduler, same optimizer", options: { color: DIM } },
    { text: "        self.lr_scheduler = OffsetPolyLR(init=1e-3, total=1000)", options: { color: SEG } },
  ], { dark: true, fontSize: 12.5, label: "custom trainer (read-only)" });
  bullets(s, 8.9, 2.35, 3.7, 3.3, [
    "Swap the scheduler at the right epoch, reuse the same optimizer",
    'Two stage labels: "warmup_all" and "train"',
    '"train" just names the post-warm-up decay phase; both phases train the whole network',
  ], { fontSize: 13.5 });
  notes(s, "28:00",
    'A custom trainer swaps the scheduler at the right epoch. At epoch 0 it builds the linear warm-up scheduler; at epoch warmup_duration_whole_net, which is 50, it builds the offset-poly scheduler, reusing the same optimizer. There are two stage labels, warmup_all and train, but both train the whole network. train just names the post-warm-up decay phase; it does not mean training starts there.');
}

{
  const s = pres.addSlide();
  scaffold(s, "Training protocol  ·  Notebook 03", "The evidence: TBI Table 2", 29);
  const rows = [
    [{ text: "fine-tuning schedule", options: { bold: true, color: "FFFFFF", fill: { color: INK } } },
     { text: "5-fold avg Dice", options: { bold: true, color: "FFFFFF", fill: { color: INK }, align: "center" } }],
    ["from scratch", { text: "53.44", options: { align: "center" } }],
    ["plain 1e-3 (no warm-up)", { text: "53.80", options: { align: "center" } }],
    [{ text: "warm-up → 1e-3", options: { bold: true, color: INK } }, { text: "54.21   (best)", options: { align: "center", bold: true, color: SEG } }],
    [{ text: "warm-up → 1e-2", options: { color: RED } }, { text: "53.28   (worse)", options: { align: "center", color: RED } }],
  ];
  s.addTable(rows, {
    x: M, y: 2.35, w: 7.4, colW: [4.9, 2.5], rowH: 0.52,
    fontFace: FONT, fontSize: 15, color: INK, valign: "middle",
    border: { type: "solid", color: LINEC, pt: 1 }, align: "left", fill: { color: "FFFFFF" },
  });
  bullets(s, 8.7, 2.4, 3.9, 3.2, [
    "Pretraining + correct fine-tuning beats from-scratch",
    "Among schedules, warm-up to 1e-3 beats plain 1e-3",
    "Warm-up to 1e-2 is worse: the peak LR matters too",
    { text: "The warm-up gain over plain 1e-3 is +0.41 Dice: small but real", options: { color: SEG, bold: true } },
  ], { fontSize: 13 });
  caption(s, M, 5.75, 7.4, "Source: TBI paper, Table 2 (arXiv:2504.06741).", { align: "left" });
  notes(s, "30:00",
    "Read the table carefully. Pretraining with correct fine-tuning beats from-scratch, 5-fold average Dice 53.44 to 54.21; the paper reports an improvement of up to about 2 Dice points in its headline setting. Among schedules, warm-up to 1e-3, 54.21, beats plain 1e-3, 53.80, while warm-up to 1e-2 is worse, 53.28. Two lessons: the schedule matters, and the peak learning rate matters, use 1e-3 not 1e-2. And keep it honest: the warm-up gain over the already-lowered plain 1e-3 is about 0.4 Dice, small but real; the larger gains come from getting off the default 1e-2 and from matching preprocessing.\n\nDeck note: the four headline numbers are shown here as a clean cited table. If you would rather show the paper's exact Table 2, paste a cropped screenshot over the table during the Google Slides pass.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Training protocol  ·  Notebook 03", "PANTHER says: not always", 30);
  box(s, M, 2.6, 5.6, 2.7, { fill: "FFFFFF", line: SEG, lineW: 2, radius: 0.1, shadow: true });
  txt(s, "TBI", { x: M + 0.35, y: 2.85, w: 4.9, h: 0.45, fontSize: 18, bold: true, color: SEG });
  txt(s, "arXiv:2504.06741  ·  brain, T1 MRI", { x: M + 0.35, y: 3.3, w: 4.9, h: 0.35, fontSize: 12, color: MUTE });
  txt(s, "Warm-up to 1e-3 helped (+0.41 Dice).", { x: M + 0.35, y: 3.8, w: 4.9, h: 1.2, fontSize: 15.5, color: INK, valign: "top" });
  box(s, 7.0, 2.6, 5.6, 2.7, { fill: PANEL, line: LINEC, radius: 0.1, shadow: true });
  txt(s, "PANTHER", { x: 7.35, y: 2.85, w: 4.9, h: 0.45, fontSize: 18, bold: true, color: INK2 });
  txt(s, "arXiv:2508.21775  ·  pancreas, MRI", { x: 7.35, y: 3.3, w: 4.9, h: 0.35, fontSize: 12, color: MUTE });
  txt(s, "Warm-up and cosine did not beat the default poly. Wins came from multi-stage transfer, augmentation, ensembling.", { x: 7.35, y: 3.8, w: 4.9, h: 1.4, fontSize: 15, color: INK, valign: "top" });
  txt(s, "The warm-up trick is task dependent, not a general law. Run the small ablation yourself.", { x: M, y: 5.55, w: 11.6, h: 0.5, fontSize: 16, italic: true, color: INK, align: "center" });
  notes(s, "32:00",
    "A related DKFZ paper, PANTHER, pancreas tumor, MRI, tried warm-up and cosine schedules and found they did not beat the default polynomial schedule for that task; its wins came from multi-stage transfer, augmentation, and ensembling. So the warm-up trick is task dependent, not a general law. Run the small ablation yourself. (Speaker: confirm PANTHER's exact schedule-ablation wording before presenting it as fact.)");
}

{
  const s = pres.addSlide();
  scaffold(s, "Training protocol  ·  Notebook 03", "See it on a real run", 31);
  slide_placeholder(s, M, 2.2, 8.0, 3.6,
    "Real fine-tuning loss curves go here",
    "default (1e-2)  vs  warm-up → 1e-3, with the epoch-50 switch marked");
  bullets(s, 8.9, 2.35, 3.7, 3.3, [
    "The schedule only matters once you train",
    "NB03 plots this once you drop the exported CSV into assets/precomputed/",
    { text: "We do not fabricate a curve. The comparison is the point, and it needs the real runs", options: { color: RED } },
  ], { fontSize: 13.5 });
  notes(s, "33:30",
    "The schedule only matters once you train. Run the two fine-tuning jobs, export the loss curves, and show the comparison. The notebook has a cell that plots this once you drop the exported CSV into assets/precomputed/. We do not fabricate a curve; the comparison is the point and it needs the real runs.",
    "NB03 cell that plots the real curves if the CSV is present.");
}

// ============================================================================
// 32 -- the rest of the pipeline (condensed; standard nnU-Net)
// ============================================================================
{
  const s = pres.addSlide();
  scaffold(s, "After fine-tuning  ·  Notebooks 04 to 06", "The rest of the pipeline is standard nnU-Net", 32);
  txt(s, "The two fine-tuning choices are done. Everything downstream is the usual nnU-Net, unchanged.", { x: M, y: 2.0, w: 11.6, h: 0.5, fontSize: 16, color: INK });
  const cards = [
    ["Inference", "Sliding-window prediction + test-time mirroring.", "nnUNetv2_predict"],
    ["Post-processing", "Connected-component filtering, chosen per dataset from cross-validation.", "a validated heuristic"],
    ["Evaluation", "Dice per label, nanmean over present classes, summary.json.", "your reported number"],
  ];
  let x = M;
  cards.forEach((c) => {
    box(s, x, 2.75, 3.73, 2.45, { fill: PANEL, line: LINEC, radius: 0.08, shadow: true });
    txt(s, c[0], { x: x + 0.3, y: 3.0, w: 3.1, h: 0.5, fontSize: 18, bold: true, color: SEG });
    txt(s, c[1], { x: x + 0.3, y: 3.55, w: 3.2, h: 1.2, fontSize: 13.5, color: INK, valign: "top" });
    txt(s, c[2], { x: x + 0.3, y: 4.75, w: 3.2, h: 0.35, fontSize: 12, italic: true, color: MUTE });
    x += 3.93;
  });
  txt(s, "Notebooks 04 to 06 demo each on a laptop (CPU) using an illustrative prediction, so there is no GPU dependency.", { x: M, y: 5.55, w: 11.6, h: 0.5, fontSize: 14.5, italic: true, color: MUTE, align: "center" });
  notes(s, "35:00",
    "A deliberate fast-forward. Once you have matched preprocessing, transferred the weights, and trained with the two-phase schedule, everything downstream is standard nnU-Net, unchanged by this recipe. Inference is sliding-window prediction with optional test-time mirroring, via nnUNetv2_predict. Post-processing is connected-component filtering, which nnU-Net selects per dataset from cross-validation, so treat it as a validated heuristic, not an automatic step; it can delete real multifocal or bilateral lesions. Evaluation is Dice per label, aggregated with nanmean over the classes that are present, written to summary.json, where the mean foreground Dice is usually your reported number. Notebooks 04 to 06 demonstrate each of these on a laptop, on CPU, using an illustrative prediction built from the ground truth, so none of it needs a GPU. If anyone wants detail, we can open those notebooks in the Q&A.",
    "optional: open NB04 / NB05 / NB06 if the audience wants detail.");
}

// ============================================================================
// SECTION 5: CONCLUSION
// ============================================================================
{
  const s = pres.addSlide();
  divider(s, "5", "Conclusion", "Recap, pitfalls, and when to use this", 33);
  notes(s, "36:30", "Pull the three choices back together and give the practical guardrails.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Conclusion", "Recap: three aligned choices", 34);
  txt(s, "If you remember one slide, this is it. These three must agree.", { x: M, y: 1.9, w: 11.6, h: 0.4, fontSize: 15, italic: true, color: MUTE });
  const y = 2.55;
  triadRow(s, M + 0.4, y, 11.0, 1, "Match the preprocessing", "1 mm isotropic + Z-score, via the moved plans.");
  triadRow(s, M + 0.4, y + 1.25, 11.0, 2, "Transfer the body, re-initialize the heads", "Encoder + decoder body load; all deep-supervision seg_layers re-init.");
  triadRow(s, M + 0.4, y + 2.5, 11.0, 3, "Two-phase learning-rate schedule", "Linear warm-up to 1e-3, then poly decay.");
  txt(s, "Throughout, the whole network trains. No part is ever frozen.", { x: M, y: 5.95, w: 11.6, h: 0.45, fontSize: 16, bold: true, color: INK, align: "center" });
  notes(s, "36:45",
    "If you remember one slide, this is it. These three choices must agree for fine-tuning a medical foundation model to pay off. And one clarification that often trips people: this two-phase schedule is a learning-rate trajectory inside one fine-tuning job; the whole network trains the entire time, no part is ever frozen.");
}

{
  const s = pres.addSlide();
  scaffold(s, "Conclusion", "Pitfalls and when to use this", 35);
  txt(s, "When to use it: adapting a strong pretrained medical model to a new, small-data task.", { x: M, y: 2.0, w: 11.6, h: 0.5, fontSize: 16, color: INK });
  const checks = [
    "Peak LR 1e-3, not 1e-2",
    "Match preprocessing first",
    "Ablate warm-up per task (the gain is not guaranteed)",
    "Remember the deep-supervision heads (all seg_layers)",
  ];
  let y = 2.75;
  checks.forEach((c) => {
    numCircle(s, M + 0.2, y, 0.45, "✓", SEG);
    txt(s, c, { x: M + 0.85, y, w: 10.5, h: 0.45, fontSize: 16.5, color: INK, valign: "middle" });
    y += 0.68;
  });
  caption(s, M, 5.85, 11.6, "Pointers: TBI (arXiv:2504.06741), PANTHER (arXiv:2508.21775), MultiTalent (arXiv:2303.14444), the nnU-Net fine-tuning docs, and this repo's notebooks.", { align: "left" });
  notes(s, "38:00",
    "When to use this: adapting a strong pretrained medical model to a new, small-data task. What to watch: the peak learning rate, the preprocessing match, and a quick per-task ablation, since the gain is not guaranteed, as PANTHER showed. Pointers: the TBI, PANTHER, and MultiTalent papers, the nnU-Net fine-tuning docs, and this repo's notebooks.");
}

{
  const s = pres.addSlide();
  s.background = { color: INK };
  txt(s, "Questions?", { x: M, y: 2.0, w: 7.2, h: 1.2, fontSize: 48, bold: true, color: "FFFFFF" });
  txt(s, "github.com/Kappapapa123/internal-mini-course", { x: M, y: 3.5, w: 7.2, h: 0.5, fontSize: 17, bold: true, color: SEG, fontFace: MONO });
  txt(s, "Notebooks runnable on a laptop  ·  quiz in NotebookLM", { x: M, y: 4.15, w: 7.2, h: 0.4, fontSize: 14, color: DIM });
  txt(s, "Keishi Suzuki  ·  University of Toronto  ·  June 2026", { x: M, y: 5.6, w: 7.2, h: 0.4, fontSize: 13, color: DIM });
  box(s, 8.2, 2.4, 4.5, 2.85, { fill: "FFFFFF", line: null, radius: 0.05, shadow: true });
  figFit(s, "lr_schedule.png", D_LR, 8.35, 2.95, 4.2, 1.9);
  caption(s, 8.2, 4.95, 4.5, "the two-phase curve", { color: MUTE });
  notes(s, "39:30",
    "Point people at the GitHub repo, runnable on a laptop, and the quiz. Open the floor. Buffer to about 45 minutes, then 10 minutes of Q&A.");
}

// ---- shape helpers (hoisted) --------------------------------------------------
function hLine(slide, x1, y, x2, o = {}) {
  slide.addShape(pres.shapes.LINE, { x: x1, y, w: x2 - x1, h: 0, line: { color: o.color || MUTE, width: o.width || 1.5, dashType: o.dash || "solid" } });
}
function vLine(slide, x, y1, y2, o = {}) {
  slide.addShape(pres.shapes.LINE, { x, y: y1, w: 0, h: y2 - y1, line: { color: o.color || MUTE, width: o.width || 1.5 } });
}
function diag(slide, x1, y1, x2, y2, o = {}) {
  const opt = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: { color: o.color || SKIP, width: o.width || 1.5 } };
  if ((x2 - x1) * (y2 - y1) < 0) opt.flipV = true;
  slide.addShape(pres.shapes.LINE, opt);
}

function lrChart(slide, x, y, w, h, labels, values, color, fmt) {
  slide.addChart(pres.charts.LINE, [{ name: "lr", labels, values }], {
    x, y, w, h,
    chartColors: [color], lineSize: 3.5, lineSmooth: false,
    showLegend: false, showTitle: false,
    catAxisLabelColor: MUTE, valAxisLabelColor: MUTE,
    catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
    catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT,
    catAxisLabelRotate: 0,
    valGridLine: { color: "E8ECF1", size: 0.5 }, catGridLine: { style: "none" },
    valAxisLabelFormatCode: fmt, catAxisTitle: "epoch", showCatAxisTitle: true,
    catAxisTitleColor: MUTE, catAxisTitleFontSize: 11, catAxisTitleFontFace: FONT,
    chartArea: { fill: { color: "FFFFFF" } },
  });
}

function drawGrid(slide, x, y, w, h, n, color) {
  box(slide, x, y, w, h, { fill: "FFFFFF", line: color, lineW: 1.5, radius: 0.02 });
  const step = w / n;
  for (let i = 1; i < n; i++) {
    slide.addShape(pres.shapes.LINE, { x: x + i * step, y, w: 0, h, line: { color, width: 0.75 } });
    slide.addShape(pres.shapes.LINE, { x, y: y + i * step, w, h: 0, line: { color, width: 0.75 } });
  }
}

function drawUNet(slide, x, y, w) {
  // symmetric U: encoder (light slate) descending, bottleneck at the vertex,
  // decoder (dark slate) ascending; horizontal dashed skip connections.
  const encCx = x + 1.4, decCx = x + w - 1.4;
  const widths = [1.8, 1.35, 0.95];
  const boxH = 0.52, gap = 0.66;
  const lvlY = [y, y + gap, y + 2 * gap];
  const enc = [], dec = [];
  widths.forEach((wi, i) => {
    box(slide, encCx - wi / 2, lvlY[i], wi, boxH, { fill: ENC, line: null, radius: 0.05 });
    enc.push({ rx: encCx + wi / 2, cy: lvlY[i] + boxH / 2, by: lvlY[i] + boxH });
    box(slide, decCx - wi / 2, lvlY[i], wi, boxH, { fill: INK2, line: null, radius: 0.05 });
    dec.push({ lx: decCx - wi / 2, cy: lvlY[i] + boxH / 2, by: lvlY[i] + boxH });
  });
  const bnW = 1.5, bnY = y + 3 * gap, bnCx = (encCx + decCx) / 2;
  box(slide, bnCx - bnW / 2, bnY, bnW, boxH, { fill: INK, line: null, radius: 0.05, text: "bottleneck", fontSize: 11, bold: true, color: "FFFFFF" });
  diag(slide, encCx, enc[2].by, bnCx - bnW / 2, bnY + boxH / 2, { color: INK2, width: 1.75 });
  diag(slide, bnCx + bnW / 2, bnY + boxH / 2, decCx, dec[2].by, { color: INK2, width: 1.75 });
  enc.forEach((e, i) => slide.addShape(pres.shapes.LINE, {
    x: e.rx, y: e.cy, w: dec[i].lx - e.rx, h: 0,
    line: { color: SKIP, width: 1.25, dashType: "dash", endArrowType: "triangle" },
  }));
  const labY = lvlY[2] + boxH + 0.12;
  txt(slide, "encoder", { x: encCx - 0.9, y: labY, w: 1.8, h: 0.3, fontSize: 11.5, color: MUTE, align: "center" });
  txt(slide, "decoder", { x: decCx - 0.9, y: labY, w: 1.8, h: 0.3, fontSize: 11.5, color: MUTE, align: "center" });
  txt(slide, "skip connections", { x: bnCx - 1.3, y: y - 0.38, w: 2.6, h: 0.3, fontSize: 11, color: MUTE, align: "center" });
}

function slide_placeholder(slide, x, y, w, h, title, sub) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: PANEL }, line: { color: INK2, width: 1.5, dashType: "dash" }, rectRadius: 0.06 });
  txt(slide, title, { x: x + 0.3, y: y + h / 2 - 0.5, w: w - 0.6, h: 0.5, align: "center", fontSize: 18, bold: true, color: INK2 });
  txt(slide, sub, { x: x + 0.3, y: y + h / 2 + 0.05, w: w - 0.6, h: 0.6, align: "center", fontSize: 13, color: MUTE });
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f)).catch((e) => { console.error(e); process.exit(1); });
