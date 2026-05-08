---
title: Detection metrics — deep dive
---

# Detection metrics — deep dive

> Object detection metrics evaluate **predicted boxes vs ground-truth boxes**. Every metric here is a different way to answer "did the predicted box overlap the right ground-truth box well enough?"

[← Home](../index.md) · [Interview deep-dive →](./interview-deep-dive.md)

## The detection contract

Each prediction has: a **box** (4 coords), a **class label**, and a **confidence score**. Each ground truth has: a **box** and a **class label** only. We need to:

1. **Match** predictions to ground truth — usually by IoU > threshold within a class.
2. **Score** — TP / FP / FN counts, then averaged over classes (and IoU thresholds, for COCO mAP).

`MeanAveragePrecision` does all of this for you with the COCO conventions baked in.

---

### IntersectionOverUnion / IoU (`tm.detection.IntersectionOverUnion`)

**What it computes.** `|A ∩ B| / |A ∪ B|` between two boxes (or sets of boxes).

**Intuition.** Box overlap as a fraction. Standard threshold for "match" is IoU ≥ 0.5 (PASCAL-VOC) or averaged over [0.5:0.95:0.05] (COCO).

**Range / direction.** `[0, 1]`. Higher better (for matched boxes).

**Real-world scenario.** Per-prediction overlap diagnostic: "Box predicted has IoU 0.42 with the closest ground-truth — close to threshold but a miss at 0.5." Logged per detection for failure analysis.

**Pitfalls.**
- IoU is non-differentiable at zero (boxes don't overlap). For training losses, use GIoU/DIoU/CIoU which extend gracefully.
- Two boxes far apart have IoU = 0 — the metric can't tell "very far" from "barely missed."

---

### GeneralizedIoU / GIoU (`tm.detection.GeneralizedIntersectionOverUnion`)

**What it computes.** `IoU − (|C \ (A ∪ B)| / |C|)`, where C is the smallest enclosing box of A and B.

**Intuition.** Penalises non-overlapping boxes by the wasted area in the enclosing rectangle. GIoU = IoU when boxes overlap; gracefully degrades to negative values for non-overlapping pairs.

**Range / direction.** `[-1, 1]`. Higher better.

**Real-world scenario.** Bounding-box regression loss in a one-stage detector — `1 − GIoU` is a smooth loss that pulls non-overlapping boxes toward each other.

---

### DistanceIoU / DIoU (`tm.detection.DistanceIntersectionOverUnion`)

**What it computes.** `IoU − (ρ²(centre_A, centre_B) / c²)`, where ρ is centre-to-centre distance and c is the enclosing-box diagonal.

**Intuition.** GIoU + a centre-distance penalty. Faster training convergence than GIoU because the gradient pushes centres directly toward each other.

---

### CompleteIoU / CIoU (`tm.detection.CompleteIntersectionOverUnion`)

**What it computes.** DIoU + an aspect-ratio penalty.

**Intuition.** Adds a third correction (aspect ratio) on top of DIoU. The standard YOLO-v4+ training loss.

---

### MeanAveragePrecision / mAP (`tm.detection.MeanAveragePrecision`)

**What it computes.** Per class: build the PR curve over predictions sorted by confidence, where TP = predicted box has IoU ≥ threshold with an unmatched ground-truth of the same class. AP = area under PR curve. Mean over classes = mAP. COCO-style: average AP further over IoU thresholds [0.5, 0.95] in steps of 0.05.

**Intuition.** Combines localisation (IoU) and classification (precision-recall) into one number. The standard detection KPI.

**Range / direction.** `[0, 1]`. Higher better. COCO mAP of 0.4 = SOTA-ish; 0.6 = excellent.

**When to use.** Any object detector — this is *the* number.

**Real-world scenario.** COCO benchmark — every detection paper reports mAP@[.5:.95]. Internal models track mAP@0.5 (more interpretable) and mAP@[.5:.95] (rigorous).

**Code.**
```python
from torchmetrics.detection import MeanAveragePrecision
m = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    iou_thresholds=[0.5, 0.55, ..., 0.95],
)
m.update(preds, target)  # preds/target are dicts of boxes/labels/scores
result = m.compute()
# result["map"], result["map_50"], result["map_75"],
# result["map_small"], result["map_medium"], result["map_large"]
```

The dict format expected:
```python
preds = [
    {"boxes": tensor[N, 4], "labels": tensor[N], "scores": tensor[N]},
    ...
]
target = [
    {"boxes": tensor[M, 4], "labels": tensor[M]},
    ...
]
```

**Pitfalls.**
- `box_format`: `"xyxy"` (corner-corner), `"xywh"` (corner+size), `"cxcywh"` (centre+size). Mixing breaks the metric silently — IoU computes wrong overlap.
- Empty predictions / empty ground truths handled subtly. Test with edge cases.
- Default IoU thresholds match COCO; for VOC use `[0.5]`.
- COCO has `iou_type="segm"` for instance segmentation masks; `"bbox"` for boxes. Pick to match your task.

---

### PanopticQualities (`tm.detection.PanopticQuality`)

**What it computes.** PQ = SQ × RQ where SQ (Segmentation Quality) is mean IoU of TP matches, RQ (Recognition Quality) is F1 across classes. Used for panoptic segmentation (every pixel gets *either* a "thing" instance ID or a "stuff" category).

**Intuition.** Decouples *did we find the thing* (RQ) from *did we segment it well* (SQ). Multiply for one number.

**Range / direction.** `[0, 1]`. Higher better.

**Real-world scenario.** Self-driving stack — needs to segment every pixel into things (cars, pedestrians, individually instance-tagged) and stuff (road, sky). Standard KPI for the panoptic head.

---

## Detection sub-metrics inside MeanAveragePrecision

`MeanAveragePrecision.compute()` returns a dict with:

- **`map`** — primary KPI, mean over IoU thresholds.
- **`map_50`** — at IoU 0.5 (PASCAL-VOC).
- **`map_75`** — at IoU 0.75 (stricter).
- **`map_small`**, **`map_medium`**, **`map_large`** — per-area-bucket. Tells you *which size* objects you're missing.
- **`mar_*`** — Mean Average Recall variants. Used to debug "is precision low because recall is bad?"
- **`map_per_class`** — per-class breakdown. Use when one class is regressing.

Always log all of these together. Top-line `map` alone can hide that the model lost 10 points on small objects while gaining 2 on large.

---

## Quick-reference: which detection metric for which scenario?

| Scenario | Primary | Secondary |
|---|---|---|
| General object detection | mAP@[.5:.95] | mAP@0.5 |
| Loss training | CIoU | GIoU |
| Per-prediction diagnostic | IoU per box | match logs |
| Panoptic segmentation | Panoptic Quality | mIoU |
| PASCAL VOC reproduction | mAP@0.5 | per-class AP |
| Real-time / edge model | mAP@0.5 | latency |
| Small-object regression debug | mAP_small | per-area mAR |

---

[Continue → interview deep-dive](./interview-deep-dive.md)
