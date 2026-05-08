---
title: Segmentation metrics — deep dive
---

# Segmentation metrics — deep dive

> Semantic / instance segmentation metrics evaluate predicted **masks** vs ground-truth masks, pixel-wise. Same family of overlap measures as detection but on per-pixel binary masks.

[← Home](../index.md) · [Interview deep-dive →](./interview-deep-dive.md)

## The segmentation contract

Per image, `preds` and `target` are tensors of shape `(C, H, W)` (multi-class one-hot or per-class probs) or `(H, W)` (class-id labels). Metrics are computed per-class then averaged.

---

### MeanIoU (`tm.segmentation.MeanIoU`)

**What it computes.** Per class: `IoU = TP / (TP + FP + FN)` over all pixels of all images. Mean over classes.

**Intuition.** Same as Jaccard, computed at the pixel level. The single most reported semantic-segmentation metric.

**Range / direction.** `[0, 1]`. Higher better.

**Real-world scenario.** Cityscapes / ADE20K — every paper reports mIoU. Internal models track per-class IoU plus mIoU.

**Code.**
```python
from torchmetrics.segmentation import MeanIoU
m = MeanIoU(num_classes=19, include_background=True)
m(preds, target)
```

**Pitfalls.**
- `include_background=True` includes the often-dominant background class — inflates mIoU. Toggle to match the convention you're being compared to (Cityscapes: false, COCO-stuff: true).
- Per-class IoU is unstable for tiny classes (small denominator). Use a `support` filter or report per-class.

---

### Dice / F1 for masks (`tm.segmentation.Dice`)

**What it computes.** `Dice = 2·TP / (2·TP + FP + FN)`. Equivalent to F1 score on pixel labels.

**Intuition.** Like IoU but with `2·TP` in numerator. Always `Dice ≥ IoU`. Same monotonic order: improving Dice ↔ improving IoU.

**Range / direction.** `[0, 1]`. Higher better.

**When to use.** Medical imaging (the dominant convention). Anywhere F1-on-pixels is the natural KPI.

**Real-world scenario.** Tumor-segmentation challenge (BraTS, LiTS): Dice is the canonical metric — `2·TP / (2·TP + FP + FN)` matches the radiologists' "how much of the tumour did you trace correctly."

**Pitfalls.**
- Dice and IoU report the *same* monotone information; one is enough. Reporting both doubles the table without adding info.

---

### GeneralizedDiceScore (`tm.segmentation.GeneralizedDiceScore`)

**What it computes.** Dice with a per-class weight `w_c = 1 / (Σ_pixels y_c)²`. Down-weights frequent classes.

**Intuition.** Standard Dice averages classes equally; large classes (background) dominate the gradient/score. Generalized Dice corrects for class size.

**Real-world scenario.** Multi-class medical segmentation where one organ is 95% of voxels and the lesion is 0.5% — generalized Dice gives the lesion a fair share of the metric. Weighted Dice loss for training is the standard fix for class imbalance in 3D segmentation.

---

### HausdorffDistance (`tm.segmentation.HausdorffDistance`)

**What it computes.** Maximum (over points in A) of the minimum distance to B. Symmetric: `max(h(A,B), h(B,A))`.

**Intuition.** Worst-case mismatch in *boundaries*. IoU/Dice can be high while a thin spurious appendage extends 50 pixels — Hausdorff catches that, IoU/Dice don't.

**Range / direction.** `[0, ∞)`. **Lower better.** In pixels (or mm if you provide spacing).

**When to use.** Medical imaging where boundary precision matters surgically (radiation therapy planning, anatomical structures).

**Real-world scenario.** Radiation oncology contouring: a 10-pixel Hausdorff between predicted tumour and ground truth could mean radiation misses or overtreats sensitive tissue. Reported alongside Dice in every challenge.

**Pitfalls.**
- Sensitive to outliers — a single far pixel dominates. Use `Hausdorff95` (the 95th-percentile variant) for noise robustness; specify via `directed=False, percentile=0.95`.
- Computing on 3D volumes is expensive — O(N²) naive, O(N log N) with kdTrees. TorchMetrics supports both.

---

## Semantic vs instance vs panoptic

- **Semantic** segmentation: every pixel gets a class. No per-instance identity. `MeanIoU` is the standard.
- **Instance** segmentation: every pixel gets a *(class, instance_id)*. `MeanAveragePrecision(iou_type="segm")` (in detection family) is standard.
- **Panoptic** segmentation: unified — things get instance IDs, stuff gets only class. `PanopticQuality` (in detection family) is standard.

These often confuse interview candidates. Get clear which contract you're evaluating before picking the metric.

---

## Quick-reference: which segmentation metric for which scenario?

| Scenario | Primary | Secondary |
|---|---|---|
| Semantic segmentation (Cityscapes/ADE20K) | mIoU | per-class IoU |
| Medical imaging (organ/tumor) | Dice | Hausdorff95 |
| Class-imbalanced multi-class | Generalized Dice | Dice |
| Boundary-critical (radiation) | Hausdorff95 | Dice |
| Instance segmentation | mAP@[.5:.95] segm | mAR |
| Panoptic | PQ | SQ, RQ |

---

[Continue → interview deep-dive](./interview-deep-dive.md)
