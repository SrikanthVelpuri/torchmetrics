---
title: Classification Metrics
nav_order: 5
---

# Classification Metrics

Classification is the largest family in TorchMetrics — 25+ public metrics under `torchmetrics.classification`. This page explains the **task taxonomy**, the **shapes of inputs**, the **threshold mechanics**, and the math behind the most-used metrics.

---

## The task taxonomy

Almost every classification metric in TorchMetrics requires a `task` argument:

```python
Accuracy(task="binary")
Accuracy(task="multiclass", num_classes=10)
Accuracy(task="multilabel", num_labels=14)
```

| Task | Each sample has | Typical `preds` shape | Typical `target` shape |
|---|---|---|---|
| `binary` | 1 of 2 mutually exclusive labels | `(N,)` floats (probs/logits) or ints (0/1) | `(N,)` ints in {0, 1} |
| `multiclass` | exactly 1 of K mutually exclusive classes | `(N, K)` logits or `(N,)` int labels | `(N,)` ints in [0, K) |
| `multilabel` | any subset of K independent labels | `(N, K)` logits/probs | `(N, K)` ints in {0, 1} |

The exact contract for each metric is in its docstring — every classification metric documents the `preds` and `target` shapes explicitly.

> The **older** API (pre-1.0) accepted `num_classes=...` only. The current API is task-prefixed (`BinaryAccuracy`, `MulticlassAccuracy`, `MultilabelAccuracy`) with a thin `Accuracy(task=...)` wrapper for backwards compatibility.

---

## Where the math actually lives: `StatScores`

Look at `src/torchmetrics/classification/`. Most classification metrics inherit from `BinaryStatScores`, `MulticlassStatScores`, or `MultilabelStatScores`. Those base classes maintain just **four counts**:

```text
TP — true  positives
FP — false positives
TN — true  negatives
FN — false negatives
```

From these four counts, you can derive almost every "single-number" classification metric:

| Metric | Formula |
|---|---|
| Accuracy | (TP + TN) / (TP + FP + TN + FN) |
| Precision | TP / (TP + FP) |
| Recall (Sensitivity, TPR) | TP / (TP + FN) |
| Specificity (TNR) | TN / (TN + FP) |
| F1 | 2 · P · R / (P + R) |
| Fβ | (1+β²) · P · R / (β² · P + R) |
| Matthews CC | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) |
| Balanced Accuracy | (TPR + TNR) / 2 |
| Negative Predictive Value | TN / (TN + FN) |

**Why this matters**: when you build a `MetricCollection` of {Accuracy, Precision, Recall, F1}, TorchMetrics can detect they all derive from the same StatScores and compute them once — that's the **compute-groups** optimization mentioned in [Core Concepts]({{ "./core-concepts.md" | relative_url }}).

---

## Threshold mechanics

For binary / multilabel tasks, predictions are typically continuous and need a threshold to become discrete.

```python
BinaryAccuracy(threshold=0.5)        # default
MultilabelAccuracy(num_labels=14, threshold=0.3)
```

If `preds` are floats *outside* `[0, 1]` (i.e. raw logits), TorchMetrics auto-applies sigmoid before thresholding. If they're already in `[0, 1]`, no sigmoid is applied. This is documented in the input contract of each metric.

For **multiclass**, no threshold is needed — `argmax` over the class dim selects the prediction.

---

## "Average" — the trap

When `num_classes > 2`, you must specify `average=...`. The choices:

| `average` | Meaning |
|---|---|
| `"micro"` | Pool all TP/FP/TN/FN across classes, then compute one number. Equivalent to overall accuracy when all samples have one label. |
| `"macro"` | Compute per-class metric, then unweighted mean. Treats every class equally — sensitive to rare classes. |
| `"weighted"` | Per-class metric weighted by support (true count). Reduces rare-class impact. |
| `"none"` (or `None`) | Return per-class array — useful for breakdowns and dashboards. |

**Practical guidance**:
- Imbalanced classification → report **macro F1** (and per-class for diagnosis).
- Cost-sensitive product metrics → use **weighted**.
- Reproducing a paper that says "F1" with no qualifier on a balanced dataset → usually micro / accuracy.

---

## Curve-based metrics

Some metrics need the *full* prediction distribution, not just argmax/threshold:

| Metric | What it integrates | State |
|---|---|---|
| `BinaryAUROC` / `MulticlassAUROC` / `MultilabelAUROC` | Area under the ROC curve | List of preds, list of targets |
| `BinaryAveragePrecision` / `MulticlassAveragePrecision` | Area under the PR curve | Same |
| `PrecisionRecallCurve` | The curve itself, returned as `(precision, recall, thresholds)` | Same |
| `ROC` | The curve itself, returned as `(fpr, tpr, thresholds)` | Same |
| `CalibrationError` (ECE) | Reliability of predicted probabilities | Bucketized counts |
| `LogAUC` | AUC on log-scaled FPR axis (drug discovery, anomaly detection) | List of preds, targets |

These are necessarily **list-state** metrics (you can't reduce to a fixed-size summary statistic without losing information). Memory grows with eval-set size — consider `compute_on_cpu=True` for large evaluations.

---

## Multilabel quirks

Multilabel metrics are independent binary problems per label, then aggregated. Common gotchas:

- **`exact_match`** — fraction of samples for which *every* label was predicted correctly. This is a brutal metric on long-tail label sets.
- **`hamming_distance`** — average per-label error. Much friendlier.
- **Subset accuracy ≠ multilabel accuracy** — be careful when copying numbers across papers.

---

## Probabilistic / ranking metrics

| Metric | When to use |
|---|---|
| `LogAUC` | Pharma / virtual screening — care more about top-of-list FPR. |
| `PrecisionAtFixedRecall` / `RecallAtFixedPrecision` | Production thresholding — "what's the precision at 95 % recall?" |
| `SensitivityAtSpecificity` / `SpecificityAtSensitivity` | Medical imaging operating points. |
| `EER` (Equal Error Rate) | Speaker / face verification. |
| `CohenKappa` | Inter-rater agreement; chance-corrected accuracy. |
| `ExactMatch` | Sequence labeling / multilabel exact-set match. |

---

## Group fairness

`torchmetrics.classification.BinaryFairness` (a.k.a. `group_fairness`) computes:

- Demographic parity ratio
- Equal opportunity
- Predictive equality
- Equalized odds

…over a `groups` tensor that tags each sample. Useful when you must report fairness alongside performance.

---

## Reading the source

A typical classification metric file looks like this (paraphrasing `accuracy.py`):

```python
class BinaryAccuracy(BinaryStatScores):
    is_differentiable: ClassVar[bool] = False
    higher_is_better:  ClassVar[bool] = True
    full_state_update: ClassVar[bool] = False

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _accuracy_reduce(tp, fp, tn, fn,
                                average="binary",
                                multidim_average=self.multidim_average)
```

Three patterns to notice:

1. The metric **inherits state** from `BinaryStatScores` — it doesn't redeclare TP/FP/TN/FN.
2. The math (`_accuracy_reduce`) lives in `torchmetrics.functional.classification.accuracy` — that's the functional API entry point. The same function is called by both APIs.
3. Class attributes set the metadata (`higher_is_better=True` so trackers know "bigger = better").

---

## A senior-engineer checklist for picking a classification metric

1. **What's the task** — binary, multiclass, or multilabel? Wrong axis = wrong number.
2. **Is the data balanced?** If not, never report micro-F1 alone.
3. **Do I need the curve, or one operating point?** AUROC for the curve, Precision@Recall for the point.
4. **Do I care about top-of-list?** Pick `LogAUC`, `PrecisionAtRecall`, or a retrieval metric.
5. **Is calibration important?** Add `CalibrationError`.
6. **Do I need fairness?** Add `BinaryFairness`.
7. **Will I run multi-GPU?** Tensor-state metrics scale fine; list-state metrics may need `compute_on_cpu=True`.
