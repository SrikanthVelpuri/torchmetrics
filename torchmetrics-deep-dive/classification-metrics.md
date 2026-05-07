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

**Why this matters**: when you build a `MetricCollection` of {Accuracy, Precision, Recall, F1}, TorchMetrics can detect they all derive from the same StatScores and compute them once — that's the **compute-groups** optimization mentioned in [Core Concepts](./core-concepts.md).

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

---

## Interview Drill-Down (multi-level follow-ups)

Format: **Q** → **F1** → **F1.1** → **F1.1.1**, four levels deep.

### Q1. AUROC vs Average Precision — when do you prefer which?

> AUROC integrates over the FPR axis, so it's dominated by negatives. AP integrates over recall on positives. On highly imbalanced data, AUROC stays high even when the model is bad; AP doesn't. **Imbalanced ⇒ AP. Balanced ⇒ either, but AP is still safer.**

  **F1.** What's the sklearn equivalent and does TorchMetrics match it numerically?

  > `sklearn.metrics.roc_auc_score` and `average_precision_score`. TorchMetrics matches to within ~1e-6 — the parity is enforced by `tests/unittests/classification/test_auroc.py`. Differences usually come from how ties in scores are broken; TorchMetrics matches sklearn's "step interpolation" for AP.

    **F1.1.** What if your eval set is too big to fit predictions in GPU memory?

    > Two options. (a) `compute_on_cpu=True` keeps the list state on host RAM. (b) Move to a binning approximation: bucket predictions into K bins; track `(positives_per_bin, count_per_bin)`. AUROC becomes a finite sum over bins. Loses fidelity at the extremes; saves orders of magnitude of RAM.

      **F1.1.1.** How accurate is the K-bin approximation?

      > For K=10000 bins on a typical eval set, agreement with full AUROC is ~1e-4. For K=1000, ~1e-3. At K=100, you're meaningfully off — usable for monitoring, not for final reporting. The biased direction depends on score distribution; usually mild overestimation when scores are concentrated.

### Q2. Why do `MulticlassF1Score(average="macro")` and `MulticlassF1Score(average="micro")` differ on the same data?

> Micro pools TP/FP/FN across all classes, then computes one F1. Macro computes per-class F1 and unweighted-averages. On imbalanced data, micro tracks the dominant class; macro treats every class equally. Different numbers, different stories.

  **F1.** Which should you put in a paper?

  > Both, with a footnote on imbalance. Hiding either is the easy way to get a reviewer 2.

    **F1.1.** Macro-F1 silently treats a 1-sample class the same as a 1M-sample class. Is that a bug or a feature?

    > Feature *if* you care about long-tail performance (medical imaging, rare-disease detection). Bug *if* you have spurious or label-noise classes. Use `weighted` average when class size is the right importance measure.

      **F1.1.1.** What if some classes have *zero* support in the eval set?

      > Per-class F1 is 0/0. TorchMetrics has a `zero_division` policy (NaN by default in newer versions). The honest report: per-class F1 with NaN for empty classes; macro-F1 over present classes only — not silently coercing to 0.

### Q3. You show me a confusion matrix. How do you decide if the model is "good"?

> First, normalize by row (per-true-class precision-of-prediction breakdown). Look at the diagonal: that's per-class recall. Look at off-diagonal mass: which classes are confused with which? A model that's accurate on average but systematically confuses two semantically-similar classes (cat ↔ dog) is different from one that scatters errors uniformly.

  **F1.** TorchMetrics' `ConfusionMatrix` returns counts. How do you turn that into a useful diagnostic?

  > Pass `normalize="true" | "pred" | "all"`. "true" gives per-row normalization (recall view). "pred" gives per-column (precision view). "all" gives joint probability. For diagnosis, "true" is usually the most informative.

    **F1.1.** What about a 1000-class problem where the matrix doesn't fit on screen?

    > Aggregate to a coarser hierarchy first (super-class → class). Or surface the **top-K confused pairs** programmatically: `argsort` off-diagonal entries, log the top 20.

      **F1.1.1.** How would you build a custom metric that returns "top-K confused pairs"?

      > Inherit `Metric`. State: a confusion-matrix tensor with `dist_reduce_fx="sum"`. `compute()` zeroes the diagonal, flattens, calls `topk` to get indices, returns the (true_class, pred_class, count) triples. ~30 lines.

### Q4. Why is calibration error not just "good enough" with a softmax?

> Cross-entropy training optimizes likelihood, not calibration. Modern deep nets are systematically over-confident — softmax probabilities are pushed to extremes. Calibration error (ECE) measures the gap between predicted confidence and observed accuracy.

  **F1.** What do you do when ECE is bad?

  > Post-hoc calibration: temperature scaling (one parameter, divide logits by `T`). Pick `T` on a validation set to minimize ECE. Doesn't change argmax, just sharpens / softens probabilities.

    **F1.1.** Why isn't temperature scaling enough sometimes?

    > It only fixes a global multiplicative bias. If the bias is class-dependent ("over-confident on dog, under-confident on cat"), you need vector / matrix scaling. TorchMetrics doesn't ship calibration models — it ships the *measurement*. The calibration model lives outside.

      **F1.1.1.** What's the right number of bins for ECE?

      > 10-20 is standard. Too few bins: hides miscalibration in narrow confidence ranges. Too many bins: each bin has too few samples → noisy estimate. Adaptive binning (equal-mass bins) is more stable than equal-width bins for skewed predictions.

### Q5. A multilabel problem with 10000 labels — what changes?

> Three things. (a) `MultilabelF1Score(num_labels=10000, average="macro")` is fine but expensive — per-label state. (b) Many labels will have zero support per batch — `update` must handle empty slices. (c) `exact_match` is meaningless at this scale — never even try.

  **F1.** What metric should headline the dashboard for 10000-label classification?

  > **Macro-F1 over the support-positive subset**, plus per-label F1 for the top-100 labels by frequency. Aggregate macro over all 10000 is dominated by 9000 zero-support labels and is uninformative.

    **F1.1.** How do you efficiently track per-label metrics for 10000 labels in TorchMetrics?

    > `MultilabelStatScores(num_labels=10000)` keeps a `(num_labels, 4)` state tensor for TP/FP/TN/FN. Tensor state, sum reduction → DDP-friendly. Compute the per-label metric in `compute()`.

      **F1.1.1.** What's the GPU memory cost?

      > Negligible. 10000 × 4 × 4 bytes = 160 KB of state. The full prediction-tensor batch is what costs memory, and that's already in the model forward.
