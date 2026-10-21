// Build the CA4 result-analysis report (format: Resources/Result Analysis Format For CA4.docx).
// usage (from report/): python report_charts.py .. && npm install && node make_report.js .. CA4_Result_Analysis_Report.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, PageNumber, TabStopType,
} = require("docx");

const REPO = process.argv[2];
const OUT = process.argv[3];
const R = (...p) => path.join(REPO, "results", ...p);
const FONT = "Times New Roman";
// The format's margins are 1440 (left) / 707 (right); we split the same total evenly so the page is
// centred, which keeps the text width (and every table and figure size) unchanged.
const MARGIN_LR = 1074;
const CONTENT_W = 11906 - 2 * MARGIN_LR; // A4 width minus left/right margins (DXA)
const MAX_IMG_PX = 640;

// ---------- helpers -------------------------------------------------------------------------
const run = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 24, bold: o.bold,
  italics: o.italics, color: o.color, superScript: o.sup, subScript: o.sub });
const para = (children, o = {}) => new Paragraph({ children: Array.isArray(children) ? children : [children],
  alignment: o.align || AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 120, line: 300 },
  numbering: o.numbering, keepNext: o.keepNext, indent: o.indent });
const text = (s, o = {}) => para([run(s, o)], o);
// "Label: rest" with the label in bold, as in the format
const labelled = (label, rest = "") => para([run(label, { bold: true }), run(rest)], { after: 120 });
const heading = (s) => para([run(s, { bold: true })], { after: 120, keepNext: true, align: AlignmentType.LEFT });
const sub = (s) => para([run(s, { bold: true, italics: true })], { after: 80, keepNext: true, align: AlignmentType.LEFT });
const bullet = (parts) => para(parts.map((p) => (typeof p === "string" ? run(p) : run(p[0], { bold: true }))),
  { numbering: { reference: "bullets", level: 0 }, after: 60 });
const numbered = (s, ref) => para([run(s)], { numbering: { reference: ref, level: 0 }, after: 60 });
const blank = () => para([run("")], { after: 0 });
const caption = (s) => para([run(s, { italics: true, size: 20 })], { align: AlignmentType.CENTER, after: 200 });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
const JPG_DIR = process.env.REPORT_JPG_DIR || path.join(__dirname, "jpg"); // JPEG copies of the large figure grids keep the .docx small
function image(rel, cap, maxW = MAX_IMG_PX) {
  const file = R("figures", rel);
  if (!fs.existsSync(file)) return [text(`[figure not available: ${rel}]`, { italics: true })];
  const { w, h } = pngSize(file);
  const jpg = path.join(JPG_DIR, rel.replace(/\.png$/, ".jpg"));
  const useJpg = fs.existsSync(jpg);
  const width = Math.min(maxW, w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { after: 60 },
      children: [new ImageRun({ type: useJpg ? "jpg" : "png", data: fs.readFileSync(useJpg ? jpg : file),
        transformation: { width, height: Math.round((h * width) / w) },
        altText: { title: rel, description: cap, name: rel } })] }),
    caption(cap),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths, boldRow = () => false) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (s, w, o = {}) => new TableCell({ width: { size: w, type: WidthType.DXA }, borders,
    shading: o.head ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ alignment: o.left ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [run(String(s), { bold: o.head || o.bold, size: 20 })] })] });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], { head: true, left: i === 0 })) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, widths[i],
        { left: i === 0, bold: boldRow(ri, i) })) }))] });
}
const splitW = (n, first) => [first, ...Array(n - 1).fill(Math.floor((CONTENT_W - first) / (n - 1)))];

// ---------- results -------------------------------------------------------------------------
const load = (tag) => {
  const f = R("metrics", `${tag}_summary.json`);
  return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f)) : {};
};
const runSummary = (name) => {
  const f = R("runs", name, "summary.json");
  return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f)) : null;
};
const fmt = (v, d = 2) => (v === undefined || v === null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const LABEL = { input: "Raw input", clahe: "CLAHE", gamma: "Gamma correction (γ = 0.7)",
  grayworld: "Gray-world white balance", ancuti_rc: "Red-channel compensation + WB (Ancuti)",
  full: "Proposed method (ours)", w_ce: "w/ Ce (full model)", wo_gnet: "w/o Gnet", wo_rnet: "w/o Rnet",
  wo_rnet_gnet: "w/o Rnet + Gnet", w_rnet_gnet: "w/ Rnet + Gnet (no Ce)" };
const ORDER = ["input", "clahe", "gamma", "grayworld", "ancuti_rc", "full"];
const HIGHER = { psnr: true, ssim: true, uiqm: true, uciqe: true, pcqi: true, niqe: false, psnr_rgb: true };

function metricTable(summary, methods, metrics, names, decimals) {
  const rows = methods.filter((m) => summary[m]).map((m) => [LABEL[m] || m,
    ...metrics.map((k, i) => fmt(summary[m][k], decimals[i]))]);
  const present = methods.filter((m) => summary[m]);
  const best = metrics.map((k) => {
    const vals = present.filter((m) => m !== "input").map((m) => summary[m][k]).filter((v) => v !== undefined && !Number.isNaN(v));
    if (!vals.length) return null;
    return HIGHER[k] === false ? Math.min(...vals) : Math.max(...vals);
  });
  const isBest = (ri, ci) => ci > 0 && best[ci - 1] !== null && present[ri] !== "input" &&
    Math.abs(summary[present[ri]][metrics[ci - 1]] - best[ci - 1]) < 1e-12;
  return table(["Method used", ...names], rows, splitW(metrics.length + 1, metrics.length > 5 ? 2300 : 3300), isBest);
}

// ---------- document ------------------------------------------------------------------------
const ugan = load("ugan_test"), uieb = load("uieb_test"), c60 = load("uieb_c60");
const abl = load("ugan_ablation"), lam = load("ugan_lambda");
const full = runSummary("full");
const ABL_ITERS = (runSummary("wo_gnet") || {}).iters || 4000;
const meta = fs.existsSync(path.join(REPO, "report", "meta.json")) ? JSON.parse(fs.readFileSync(path.join(REPO, "report", "meta.json"))) : {};

const body = [];
// header block (identical layout to the format)
body.push(blank());
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [run("MGM UNIVERSITY", { bold: true })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [run("Jawaharlal Nehru Engineering College, Chh. Sambhajinagar", { size: 28 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [run("Computer Science & Engineering", { size: 28 })] }));
body.push(blank());
body.push(labelled("Department Name: ", "AI&DS"));
body.push(labelled("CA4: ", "Integrating Deep Learning and Traditional Image Enhancement Techniques for Underwater Image Enhancement (Z. Shi, Y. Wang, Z. Zhou, W. Ren, IET Image Processing, vol. 16, pp. 3471–3484, 2022, doi:10.1049/ipr2.12544)"));
// one line per student, roll numbers aligned on a tab stop
body.push(para([run("Name of students in a group:", { bold: true })], { after: 40, align: AlignmentType.LEFT }));
(meta.students || []).forEach((st) => body.push(new Paragraph({ spacing: { after: 40 }, indent: { left: 360 },
  tabStops: [{ type: TabStopType.LEFT, position: 4300 }],
  children: [run(`${st.name}`), run("\tRoll No: ", { bold: true }), run(st.roll)] })));
body.push(blank());

// Introduction
body.push(heading("Introduction:"));
body.push(text("Underwater images suffer from colour distortion, blur and low contrast because water absorbs light selectively by wavelength (red is absorbed fastest, blue travels farthest) and scatters it. This degrades every downstream task: marine biology surveys, underwater robotics, pipeline inspection and object detection."));
body.push(text("The paper by Shi et al. starts from a statistical observation. For each image, the mean of each channel in the ground truth minus its mean in the underwater image is consistently positive for red (red is attenuated), negative for green (green is over-strengthened) and close to zero for blue. Based on this prior, the authors compensate the red and green channels differently: red with an attention-guided residual module that adds a gated compensation term, green with one that subtracts the excess. Contrast enhancement and deblurring then follow in a multi-scale CNN (MSCNN) that fuses two traditional enhancements, CLAHE and Gamma correction, using learned confidence maps."));
body.push(sub("Work done in this CA:"));
[
  "Implemented the complete method from scratch in PyTorch (Rnet, Gnet, blue residual module, MSCNN with CLAHE and Gamma, and the L1 + SSIM + VGG-19 perceptual loss), following the equations and figures of the paper.",
  "Reproduced the paper's motivating statistical analysis (Fig. 1) and its post-enhancement counterpart (Fig. 11).",
  "Trained the network on the same UGAN synthetic dataset (4000 training / 100 test pairs) and evaluated it on synthetic (UGAN-100) and real (UIEB-90 with references, UIEB-Challenging-60 without) underwater images.",
  "Compared against four traditional enhancement algorithms and reproduced the ablation study (Table 4) and the loss-weight study (Table 5).",
  "Reported full-reference (PSNR, SSIM) and no-reference (UIQM, UCIQE, PCQI) underwater quality metrics, with visual results for every stage of the network.",
].forEach((s) => body.push(bullet([s])));
body.push(sub("Contribution of the paper:"));
[
  "A prior that both the red and the green channel contribute to underwater colour distortion.",
  "Attention-guided (channel + spatial attention) residual modules that compensate red and green in different ways.",
  "A fusion of traditional techniques (CLAHE, Gamma correction) with a multi-scale CNN for contrast enhancement and deblurring.",
].forEach((s) => body.push(bullet([s])));

// Methods
body.push(heading("Methods / Algorithms Used in paper and Comparative algorithm used by you"));
body.push(sub("A. Method of the paper (implemented)"));
body.push(text("The input image I is split into its R, G and B channels, and each channel is corrected by its own module:"));
body.push(bullet([["Rnet (red compensation, Eq. 1): "], "Irc = Ir + CAB(Ir) · SAB · (1 − Ir). CAB is channel attention (Global Average Pooling → FC → Sigmoid) and SAB is spatial attention (two 3×3 convolutions → Sigmoid, an H×W×1 map). The factor (1 − Ir) stops already-bright red pixels from being over-compensated. A long skip connection preserves data fidelity."]));
body.push(bullet([["Gnet (green compensation, Eq. 2): "], "Iga = Ig − CAB(Ig) · SAB. Same structure as Rnet but without (1 − I); part of the over-strong green is subtracted."]));
body.push(bullet([["Blue residual enhancement: "], "three residual blocks (conv–LeakyReLU–conv–LeakyReLU–conv + skip) refine the well-preserved blue channel."]));
body.push(bullet([["Contrast enhancement module Ce (MSCNN, Eq. 3): "], "the colour-corrected image I1 is enhanced by CLAHE (I_CE) and by Gamma correction (I_GC). The top and bottom branches (3×3, 5×5, 7×7 convolutions) refine I_CE and I_GC into Z_CE and Z_GC. The middle branch (3×3, 5×5, 7×7, 1×1, 3×3, 5×5, 7×7 convolutions + Sigmoid) predicts the confidence maps W_CE and W_GC. Output: I_out = Z_CE · W_CE + Z_GC · W_GC."]));
body.push(bullet([["Loss (Eq. 8): "], "L = λ1 (L1 + L_SSIM) + λ2 · L_per, with λ1 = 1 and λ2 = 0.1. L_SSIM uses grayscale images with an 11×11 window; L_per is the L1 distance between VGG-19 relu4_3 features."]));
body.push(text(`The implemented network has ${full ? full.params.toLocaleString("en-US") : "317,697"} trainable parameters (about 1.3 MB), in line with the paper's emphasis on a light model.`));
body.push(sub("Implementation and training settings"));
const trainRows = [
  ["Framework", "PyTorch 2.8, kornia (differentiable-image library) for CLAHE"],
  ["Optimiser", "Adam, learning rate 2 × 10⁻⁴ (as in the paper)"],
  ["Batch size / input size", "8 / 256 × 256 (as in the paper)"],
  ["Training length", full ? `${full.iters.toLocaleString("en-US")} iterations ≈ ${full.epochs_completed} epochs in ${full.elapsed_min} min (paper: 50 epochs)` : "–"],
  ["Hardware", "NVIDIA L40S (main model), NVIDIA A10G (ablations, λ sweep) on Modal"],
  ["Feature width", "32 channels in every module"],
  ["CLAHE / Gamma", "clip limit 2.0, 8 × 8 tiles, per RGB channel / γ = 0.7"],
  ["Precision", "fp32 with TF32 (fp16 mixed precision made training unstable)"],
];
body.push(table(["Setting", "Value"], trainRows, [3000, CONTENT_W - 3000]));
body.push(blank());
body.push(sub("B. Comparative algorithms used by us"));
[
  [["CLAHE"], " – contrast-limited adaptive histogram equalisation on the L channel of CIELab (clip 2.0, 8 × 8 tiles)."],
  [["Gamma correction"], " – I^0.7, which brightens dark regions."],
  [["Gray-world white balance"], " – scales each channel so that the channel means become equal."],
  [["Red-channel compensation + white balance (Ancuti et al., 2018)"], " – Irc = Ir + (Īg − Īr)(1 − Ir) Ig followed by gray-world. It is the hand-crafted counterpart of Rnet."],
  [["Ablation variants of the network"], " – w/o Gnet, w/o Rnet, w/o Rnet + Gnet, and w/ Rnet + Gnet without the contrast module (Table 4 of the paper)."],
].forEach((p) => body.push(bullet(p)));
body.push(sub("C. Evaluation metrics"));
[
  [["PSNR / SSIM"], " (full-reference) – PSNR on RGB and on luminance (Y), SSIM on luminance – higher is better."],
  [["NIQE"], " (natural image quality evaluator, no reference) – lower is better."],
  [["UIQM"], " (colourfulness, sharpness and contrast; weights 0.0282 / 0.2953 / 3.5753) – higher is better."],
  [["UCIQE"], " (chroma, saturation and contrast; weights 0.4680 / 0.2745 / 0.2576) – higher is better."],
  [["PCQI"], " (patch-based contrast quality between the raw input and the enhanced image) – higher means more contrast gain."],
].forEach((p) => body.push(bullet(p)));

// Datasets
body.push(heading("Datasets Used:"));
body.push(table(["Dataset", "Source", "Type", "Images used", "Purpose"], [
  ["UGAN (Fabbri et al., ICRA 2018)", "Same dataset as the paper ('underwater_imagenet', EUVP/UGAN release)", "Synthetic, CycleGAN-distorted, with ground truth", "6128 pairs → 4000 train / 100 test (seed 0)", "Training; Table 1 and Table 4"],
  ["UIEB (Li et al., IEEE TIP 2020) – test", "Official UIEB release, seed-42 split", "Real, with reference", "90 pairs", "Real images with reference (Table 3)"],
  ["UIEB – Challenging-60", "Official UIEB release", "Real, no reference", "60 images", "No-reference evaluation (Table 2)"],
], [2000, 2300, 1900, 1800, CONTENT_W - 8000]));
body.push(blank());
body.push(text("All images are resized to 256 × 256. As in the paper, the network is trained only on the synthetic UGAN pairs; the UIEB images measure how well it generalises to real underwater scenes."));

// Result analysis
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(heading("Result Analysis:"));
body.push(sub("1. Images as an output for every stage"));
body.push(text("The figures below show the output of every stage of the network: input → I1 (after Rnet / Gnet / Bnet colour compensation) → I_CE (CLAHE of I1) → I_GC (Gamma of I1) → confidence maps W_CE and W_GC → final output."));
body.push(...image("intermediates_ugan_test.png", "Figure 1: Output of every stage on UGAN test images (synthetic)."));
body.push(...image("intermediates_uieb_test.png", "Figure 2: Output of every stage on real UIEB images."));
body.push(...image("comparisons_ugan_test.png", "Figure 3: Visual comparison on the UGAN test set (columns: methods, last column: ground truth)."));
body.push(...image("comparisons_uieb_test.png", "Figure 4: Visual comparison on real UIEB test images with reference."));
body.push(...image("comparisons_uieb_c60.png", "Figure 5: Visual comparison on UIEB-Challenging-60 (no reference available)."));
body.push(...image("comparisons_ugan_ablation.png", "Figure 6: Visual ablation of the colour-correction and contrast modules."));

body.push(sub("2. Results in table format"));
body.push(text("Table 1 – Synthetic UGAN test set (100 images, full reference). Best result among the enhancement methods is in bold.", { bold: true, keepNext: true }));
const REF_M = ["psnr_rgb", "psnr", "ssim", "uiqm", "uciqe", "pcqi", "niqe"];
const REF_N = ["PSNR-RGB (dB) ↑", "PSNR-Y (dB) ↑", "SSIM ↑", "UIQM ↑", "UCIQE ↑", "PCQI ↑", "NIQE ↓"];
const REF_D = [2, 2, 3, 3, 3, 3, 3];
body.push(metricTable(ugan, ORDER, REF_M, REF_N, REF_D));
body.push(text("PSNR-RGB is computed on all three colour channels; PSNR-Y and SSIM on the luminance channel, as the paper's text describes. The paper's input PSNR on this dataset (16.85 dB) matches our PSNR-RGB of the raw input, which suggests its PSNR values were computed on RGB.", { italics: true, size: 20 }));
body.push(blank());
body.push(text("Table 2 – UIEB-Challenging-60 (real, no reference).", { bold: true, keepNext: true }));
const c60Metrics = ["uiqm", "uciqe", "pcqi"].concat(Object.values(c60).some((m) => m.niqe !== undefined) ? ["niqe"] : []);
body.push(metricTable(c60, ORDER, c60Metrics, c60Metrics.map((k) => ({ uiqm: "UIQM ↑", uciqe: "UCIQE ↑", pcqi: "PCQI ↑", niqe: "NIQE ↓" })[k]), c60Metrics.map(() => 3)));
body.push(blank());
body.push(text("Table 3 – UIEB test set (90 real images with reference).", { bold: true, keepNext: true }));
body.push(metricTable(uieb, ORDER, REF_M, REF_N, REF_D));
body.push(blank());
body.push(text(`Table 4 – Ablation study on UGAN-100. Every variant is trained for the same ${ABL_ITERS.toLocaleString("en-US")} iterations.`, { bold: true, keepNext: true }));
body.push(metricTable(abl, ["wo_rnet_gnet", "wo_rnet", "wo_gnet", "w_rnet_gnet", "w_ce"], ["psnr_rgb", "psnr", "ssim", "uiqm", "uciqe"], ["PSNR-RGB (dB) ↑", "PSNR-Y (dB) ↑", "SSIM ↑", "UIQM ↑", "UCIQE ↑"], [2, 2, 3, 3, 3]));
body.push(blank());
if (Object.keys(lam).length) {
  const lamRows = Object.keys(lam).map((k) => { const m = k.match(/l1_([\d.]+)_l2_([\d.]+)/); return [m ? +m[1] : NaN, m ? +m[2] : NaN, lam[k]]; })
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const bestP = Math.max(...lamRows.map((r) => r[2].psnr_rgb));
  body.push(text(`Table 5 – Loss weights λ1 / λ2 on UGAN-100 (each run ${(runSummary("sweep_l1_1_l2_1") || {}).iters || 2000} iterations).`, { bold: true, keepNext: true }));
  body.push(table(["λ1", "λ2", "λ2 / λ1", "PSNR-RGB (dB) ↑", "PSNR-Y (dB) ↑", "SSIM ↑"],
    lamRows.map((r) => [r[0], r[1], +(r[1] / r[0]).toPrecision(2), fmt(r[2].psnr_rgb), fmt(r[2].psnr), fmt(r[2].ssim, 3)]),
    splitW(6, 1400), (ri) => lamRows[ri][2].psnr_rgb === bestP));
  body.push(blank());
}
if (ugan.full) {
  body.push(text("Table 6 – Comparison with the values reported in the paper.", { bold: true, keepNext: true }));
  body.push(table(["Quantity", "Paper", "Our implementation"], [
    ["UGAN test: input PSNR / SSIM", "16.85 / 0.692", `${fmt(ugan.input.psnr_rgb)} (RGB), ${fmt(ugan.input.psnr)} (Y) / ${fmt(ugan.input.ssim, 3)}`],
    ["UGAN test: proposed PSNR / SSIM", "25.67 / 0.891", `${fmt(ugan.full.psnr_rgb)} (RGB), ${fmt(ugan.full.psnr)} (Y) / ${fmt(ugan.full.ssim, 3)}`],
    ["UIEB (real, reference): proposed PSNR / SSIM", "24.89 / 0.837", uieb.full ? `${fmt(uieb.full.psnr_rgb)} (RGB), ${fmt(uieb.full.psnr)} (Y) / ${fmt(uieb.full.ssim, 3)}` : "–"],
    ["PSNR gain over the raw input (UGAN, RGB)", "+8.82 dB", `${ugan.full.psnr_rgb - ugan.input.psnr_rgb >= 0 ? "+" : ""}${fmt(ugan.full.psnr_rgb - ugan.input.psnr_rgb)} dB`],
    ["Training length", "50 epochs (GTX 1650)", full ? `${full.epochs_completed} epochs (${full.elapsed_min} min, L40S)` : "–"],
  ], [4200, 2400, CONTENT_W - 6600]));
  body.push(text("UIQM and UCIQE values are not directly comparable with the paper's absolute numbers, because implementations of these metrics differ in normalisation. The relative ranking of methods is what matters.", { italics: true, size: 20 }));
}

body.push(sub("3. Charts and graphs based on the result"));
body.push(...image("report_metrics_bar.png", "Figure 7: PSNR and SSIM of all methods on the synthetic and real test sets."));
body.push(...image("report_noref_bar.png", "Figure 8: No-reference quality (UIQM, UCIQE) on UIEB-Challenging-60."));
body.push(...image("report_ablation_bar.png", "Figure 9: Ablation study – contribution of Rnet, Gnet and the contrast module."));
body.push(...image("report_lambda_heatmap.png", "Figure 10: PSNR for each λ1 / λ2 combination (Table 5)."));
body.push(...image("training_curves.png", "Figure 11: Training loss curves of the full model and the ablation variants."));
body.push(...image("channel_dev_ugan_test_input.png", "Figure 12: Per-image channel deviation mean(GT) − mean(input) on UGAN – reproduction of the paper's Fig. 1."));
body.push(...image("channel_dev_ugan_test_full.png", "Figure 13: Channel deviation after enhancement – reproduction of the paper's Fig. 11."));
body.push(...image("channel_dev_ugan_test_summary.png", "Figure 14: Mean ± std channel deviation before and after enhancement (UGAN)."));
body.push(...image("channel_dev_uieb_test_summary.png", "Figure 15: Mean ± std channel deviation before and after enhancement (real UIEB)."));

body.push(sub("Observations"));
(meta.observations || []).forEach((s) => body.push(bullet([s])));

body.push(heading("4. Conclusion"));
(meta.conclusion || []).forEach((s) => body.push(text(s)));

body.push(heading("References"));
[
  "Z. Shi, Y. Wang, Z. Zhou, W. Ren, “Integrating deep learning and traditional image enhancement techniques for underwater image enhancement,” IET Image Processing, 16(13), 3471–3484, 2022.",
  "C. Fabbri, M. J. Islam, J. Sattar, “Enhancing underwater imagery using generative adversarial networks,” ICRA, 2018.",
  "C. Li et al., “An underwater image enhancement benchmark dataset and beyond,” IEEE TIP, 29, 4376–4389, 2020.",
  "C. O. Ancuti, C. Ancuti, C. De Vleeschouwer, P. Bekaert, “Color balance and fusion for underwater image enhancement,” IEEE TIP, 27(1), 379–393, 2018.",
  "K. Panetta, C. Gao, S. Agaian, “Human-visual-system-inspired underwater image quality measures,” IEEE JOE, 41(3), 2016 (UIQM).",
  "M. Yang, A. Sowmya, “An underwater color image quality evaluation metric,” IEEE TIP, 24(12), 2015 (UCIQE).",
  "S. Wang et al., “A patch-structure representation method for quality assessment of contrast changed images,” IEEE SPL, 22(12), 2015 (PCQI).",
  "Code: https://github.com/akashch1013/UIENetHybrid",
].forEach((s) => body.push(numbered(s, "refs")));

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 24 } } } },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 540 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 523, right: MARGIN_LR, bottom: 730, left: MARGIN_LR, header: 0, footer: 300 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
