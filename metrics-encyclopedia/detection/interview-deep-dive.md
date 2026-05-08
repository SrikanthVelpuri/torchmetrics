---
title: Detection metrics — interview deep dive
---

# Detection metrics — interview deep dive

[← Family page](./index.md) · [← Home](../index.md)

---

## Q1. "Walk me through how mAP is computed."

**Answer.** Three stages:
1. **Match** predictions to ground truth per class. Sort predictions by confidence; for each, mark it TP if its highest-IoU match exceeds threshold *and* that ground-truth wasn't already matched, else FP. Unmatched ground-truths at the end are FN.
2. **Per-class AP** = area under the PR curve from those TP/FP/FN counts. (Step-wise integral or all-points interpolation depending on convention.)
3. **mAP** = mean of per-class AP. COCO-style: also average over IoU thresholds [0.5, 0.95] in steps 0.05.

> **F1.1** "Why average over IoU thresholds?"
>
> **Answer.** PASCAL-VOC's mAP@0.5 says "any IoU ≥ 0.5 is fine" — rewards models that get loose-but-correct boxes. Real applications often need tight boxes (e.g., self-driving). Averaging over [0.5:0.95] forces models to be precise; mAP@.95 alone is too strict.

> **F1.2** "Why is mAP@0.5 always ≥ mAP@0.75?"
>
> **Answer.** A higher IoU threshold demotes more matches to FP. Strictly more failures, so AP can only decrease as the threshold rises. The same model with mAP@0.5 = 0.6 and mAP@0.75 = 0.4 has roughly half its detections at IoU between 0.5 and 0.75 — i.e., loose boxes.

---

## Q2. "What's the difference between IoU, GIoU, DIoU, CIoU?"

**Answer.**
- **IoU**: intersection over union. Zero when boxes don't overlap; non-differentiable.
- **GIoU**: IoU minus the area of the smallest enclosing box wasted. Negative for non-overlapping; smooth.
- **DIoU**: GIoU + centre-distance penalty. Faster convergence.
- **CIoU**: DIoU + aspect-ratio penalty. The standard YOLO-v4+ training loss.

For evaluation we use IoU. For training losses, the rest in increasing sophistication.

> **F2.1** "When does CIoU underperform DIoU?"
>
> **Answer.** When the dataset has very heterogeneous aspect ratios within a class (e.g., books photographed at every angle). CIoU's aspect-ratio penalty ends up fighting itself. Pick DIoU.

---

## Q3. "Your mAP regressed from 0.42 to 0.40. How do you debug?"

**Answer.** Drill down by sub-metric *first*, top-line second:
1. `map_50` vs `map_75` — is the regression at strict IoU (localisation) or loose (detection)?
2. `map_small` / `_medium` / `_large` — which size class lost?
3. `map_per_class` — which class lost? Is it one class or distributed?
4. For the regressing class+size, look at confusion: are detections becoming FP (precision drop) or FN (recall drop)?

A 2-point top-line drop with one specific class+size collapsing is a data/augmentation issue, not a model issue.

---

## Q4. "What's COCO mAP vs PASCAL VOC mAP?"

**Answer.**
- **PASCAL VOC**: mAP@0.5 only. Per-class AP averaged. Uses 11-point interpolated AP (legacy) or all-points AP (modern).
- **COCO**: mAP averaged over IoU [0.5, 0.95] in steps of 0.05 (10 thresholds). Per-class. Step-wise AP integral. Plus per-area-bucket and per-max-detection variants.

TorchMetrics defaults to COCO. The numbers are not directly comparable across the two — never quote a "mAP" without saying which.

---

## Q5. "Why are predictions sorted by confidence in mAP?"

**Answer.** Because the PR curve depends on the order: as you lower the threshold, more predictions enter the top-k, both TP and FP. Sorting by confidence means the curve is built greedily, simulating a precision-recall trade-off at every threshold. The AP integral is over this curve.

> **F5.1** "What if my model returns unsorted boxes?"
>
> **Answer.** TorchMetrics sorts internally by score. But if your model emits *unconfident* predictions interspersed with confident ones (e.g., bug in NMS), you'll see precision drop fast at high recall — diagnostic of the issue.

---

## Q6. "How is panoptic quality different from mIoU?"

**Answer.** mIoU is per-class IoU averaged over classes — doesn't track instance-level identity. Panoptic Quality decomposes into:
- **SQ (Segmentation Quality)**: mean IoU of *matched* prediction-truth instances (only TP matches contribute).
- **RQ (Recognition Quality)**: F1 over instance matches across classes.
- PQ = SQ × RQ.

So PQ punishes both bad pixels and bad instance counts. mIoU only the former.

> **F6.1** "Self-driving — PQ for cars and `stuff` for roads. Why?"
>
> **Answer.** Cars are *things* — countable instances; we need to segment "this car vs that car" because tracking and ego-motion depend on it. Roads are *stuff* — uncountable; we just need pixel-level "is this road." PQ handles both: things use full PQ; stuff degenerates to per-class IoU. The unified framework gives one number for the panoptic head.

---

## Q7. "Confidence threshold for detection — how do you pick it?"

**Answer.** mAP is threshold-free (it sweeps), but a deployed system uses one. Pick by operating point: required precision (don't fire on phantom cars) or required recall (must catch every pedestrian). Use the PR curve from the validation set. For safety-critical systems, fix recall at regulatory floor and pick the highest-confidence threshold that hits it. Keep the chosen threshold in version control with the model.

---

## Q8. "What's the IoU-NMS gotcha?"

**Answer.** Non-Max Suppression uses IoU between *predictions* (not predictions vs truth). NMS removes overlapping predictions of the same class. Two issues:
1. **Crowded scenes** (a flock of birds): NMS at IoU=0.5 deletes nearby true positives, hurting mAR.
2. **Class confusion**: NMS is per-class; if the model predicts the same box as both `car` and `truck` with low confidence, both can survive.

Solutions: Soft-NMS, class-aware NMS, learned NMS. The mAP impact can be 1-2 points.

---

## Q9. "How does TorchMetrics handle DDP for `MeanAveragePrecision`?"

**Answer.** State is per-image lists of (boxes, labels, scores). `_sync_dist` gathers all lists to all ranks (or just rank 0 with `dist_sync_fn`). Memory cost is O(total predictions across all images). For a 100k-image validation set with hundreds of predictions per image, this is large — compute on a single GPU at the end, or batch the dataset across multiple ranks and only `_sync_dist` once at epoch end.

---

## Q10. "I see mAP_small = 0.05 and mAP_large = 0.65. Diagnosis?"

**Answer.** The model is failing on small objects. Common causes (in order of probability):
1. **Resolution mismatch**: training resolution too low — small objects in the training data span < 8×8 pixels at network input.
2. **FPN levels**: feature pyramid network missing the highest-resolution level (P2/P3) for small objects.
3. **Scale jitter** weak in augmentation.
4. **Ground-truth noise** — small-object boxes are often loose in COCO; our test set may be tighter.

Fix #1 first (multi-scale training); fix #2 by adding finer FPN levels.

---

## Q11. "Why per-class mAP varies dramatically — diagnosis?"

**Answer.** Three reasons:
1. **Class imbalance** — rare classes have few examples → noisy AP.
2. **Visual confusability** — `bicycle` vs `motorcycle` per-class AP both suffer because matches collide across classes.
3. **Annotation quality** — some classes have systematically loose boxes.

Always show per-class AP table; don't rely on top-line mAP alone.

---

## Q12. "Can mAP go up while business KPI goes down?"

**Answer.** Yes. mAP rewards average precision over an entire PR curve at all IoU thresholds; the deployed system uses *one* threshold and *one* IoU cutoff. A model that improves the deep PR curve (rare positives at low confidence) but doesn't help the deployed operating point shows higher mAP with no business win. Always co-log: mAP for benchmarking, *operating-point precision/recall* for ship/no-ship.

---

## Q13. "Walk me through TorchMetrics' implementation of MeanAveragePrecision under the hood."

**Answer.** Two-step:
1. **Per-image matching**: build IoU matrix between preds and targets per class. Greedy match in confidence order. Mark TP/FP. Unmatched targets → FN at end.
2. **Per-class AP**: pool all per-image TP/FP/FN per class, sort by confidence, compute step-wise PR curve, integrate area.

For COCO-style, repeat over IoU thresholds and averaged. Implementation lives in `_mean_ap.py`. The wrapper `mean_ap.py` handles the dict-based input format.

---

## Q14. "What's the difference between Recall and Mean Average Recall in detection?"

**Answer.** Recall is at one IoU threshold and confidence cutoff. mAR averages recall over IoU thresholds and over a fixed *number of detections per image* (typically 100). It's a complement to mAP that tells you "if I let the detector emit 100 boxes, how much of the ground truth do I cover?" Useful to diagnose whether mAP is precision-bound or recall-bound.

---

## Q15. "How do you measure detection latency vs accuracy trade-off?"

**Answer.** Two-axis report: latency (p50/p99 ms on target hardware) and mAP. Pareto frontier: for each model, plot (latency, mAP). The deployable models live on the upper-left edge. Single-number scoring (e.g., "mAP per ms") collapses the curve and hides the choice.

---

## Q16. "What's the most common detection bug in TorchMetrics usage?"

**Answer.** **Wrong `box_format`.** Default is `xyxy`. If your model outputs `xywh` (a common YOLO-style convention), the IoU computation is silently wrong — the metric runs and reports a garbage number. Always set `box_format=` explicitly. Second most common: forgetting that `target` is a list of dicts, not a single dict.

---

[← Back to family page](./index.md)
