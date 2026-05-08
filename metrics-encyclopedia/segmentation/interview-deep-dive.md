---
title: Segmentation metrics — interview deep dive
---

# Segmentation metrics — interview deep dive

[← Family page](./index.md) · [← Home](../index.md)

---

## Q1. "Dice vs IoU — same info?"

**Answer.** Same monotone information. `Dice = 2·IoU / (1 + IoU)`. So if Dice ranks model A > B, IoU ranks them the same way. **One is enough**; reporting both is redundant. Dice tends to be reported in medical imaging; mIoU in general computer vision. Match the audience.

> **F1.1** "Then why does anyone bother with both?"
>
> **Answer.** Different communities pin different ones for historical reasons. The numerical scale differs — Dice is always ≥ IoU — so people informally remember "Dice 0.85 = IoU 0.74" as a sanity check. But a paper that reports Dice and IoU and a third overlap metric is just adding columns.

---

## Q2. "When does Dice/IoU lie to you?"

**Answer.** Boundary errors. A predicted mask that's pixel-perfect except for a single 50-pixel spurious tail has IoU ≈ 0.95 but Hausdorff = 50. For radiation therapy or surgical planning, the tail matters more than the bulk. Always co-report Hausdorff95 when boundary geometry matters.

> **F2.1** "Why Hausdorff *95* and not raw Hausdorff?"
>
> **Answer.** Raw Hausdorff is the *worst-case* point-to-point distance — one outlier pixel from segmentation noise dominates. Hausdorff95 takes the 95th percentile of the per-point distance distribution: robust to a handful of noisy pixels while still catching systematic boundary errors.

---

## Q3. "Why use Generalized Dice instead of Dice?"

**Answer.** Class imbalance. Multi-class segmentation with background = 90% of pixels: standard Dice (mean over classes, equal weight) is dominated by background's near-1.0 score, hiding small-class failures. Generalized Dice weights each class by `1 / (volume²)`, giving small classes much more weight. The loss/metric becomes balanced.

---

## Q4. "Cityscapes vs ADE20K — what changes about how you report mIoU?"

**Answer.**
- Cityscapes has 19 evaluation classes; the convention is to *exclude* the void/background label. `include_background=False`.
- ADE20K has 150 classes including a "background" pseudo-class — convention varies; check the leaderboard.

Mismatched `include_background` shifts mIoU 5-10 points. Always pin to match the leaderboard you're competing on.

---

## Q5. "Per-class IoU on a class with 50 pixels in the test set — what does it mean?"

**Answer.** Almost nothing. IoU = `TP / (TP + FP + FN)` with denominator ≈ 50; per-pixel noise makes the estimate range over a wide CI. Either pool small classes ("rare-class" macro), bootstrap the estimate, or skip per-class for support < some threshold.

---

## Q6. "How does TorchMetrics implement DDP for `MeanIoU`?"

**Answer.** State is per-class running counters `(intersection, union)` of pixel counts. `_sync_dist` is `all_reduce(SUM)` — O(C) bandwidth, exact. Very cheap, scales to thousands of GPUs. Hausdorff is harder: state is the full coordinate sets, O(N) to gather; not done online during training, only at eval time.

---

## Q7. "Cross-entropy loss vs Dice loss for training a segmentation model?"

**Answer.**
- **Cross-entropy**: pixel-wise classification loss. Good gradient signal everywhere. **Class-imbalanced data → biased toward majority class.**
- **Dice loss** (`1 − Dice`): directly optimises overlap. Naturally handles class imbalance because the metric isn't pixel-weighted; minority class still has full weight.
- **Combined** (`CE + Dice`): the canonical recipe in medical imaging. CE gives smooth gradients early; Dice tightens boundaries late.

The interview signal: explain the imbalance trade-off and recommend the combined loss.

---

## Q8. "Panoptic quality vs mIoU vs mAP — which when?"

**Answer.**
- **mIoU** — semantic segmentation (every pixel a class, no instances).
- **mAP segm** — instance segmentation (countable objects, instance IDs matter).
- **PQ** — panoptic (unified things+stuff).

Picking wrong: a self-driving stack reporting mIoU only loses the per-car instance identity that matters for tracking. A tumor segmentation reporting mAP only loses the boundary precision that matters surgically.

---

## Q9. "What does it mean for Dice to drop from 0.85 to 0.83 when adding a new class?"

**Answer.** Adding a class with low Dice pulls the mean down. Doesn't necessarily mean other classes regressed. Always look at *per-class* Dice for the existing classes alongside the new one; drops on existing classes mean a real regression, drops only on the new class are expected.

---

## Q10. "How do you measure a 3D medical segmentation model?"

**Answer.** Stack:
- **Dice (per organ)** — primary KPI, what radiologists track.
- **Hausdorff95 (per organ)** — boundary precision (in mm, not pixels — provide voxel spacing).
- **Volume difference** — predicted vs true volume (bias check).
- **NSD (Normalized Surface Distance)** at a tolerance — fraction of boundary pixels within `τ` mm of the truth.

mIoU is technically equivalent to Dice (monotone) but the medical-imaging community uses Dice exclusively.

---

## Q11. "When is mIoU a worse metric than per-class IoU vector?"

**Answer.** When the dataset is unbalanced and the failure modes are class-specific. mIoU = 0.62 hides "8 classes at 0.85, 11 classes at 0.45 (the rare ones)." Always log per-class IoU; mIoU is the headline, the vector is the diagnostic.

---

[← Back to family page](./index.md)
