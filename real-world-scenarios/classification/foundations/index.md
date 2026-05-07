# Foundations: Baselines, Metric Design, and Business Correlation

Every classification system eventually turns into a decision. That decision creates true positives, false positives,
true negatives, and false negatives, or the multiclass and multilabel equivalents of those outcomes. If those outcomes
do not map cleanly to business value, the evaluation setup is incomplete.

```{mermaid}
flowchart LR
    A[Model score or class prediction] --> B[Decision rule]
    B --> C[Confusion matrix outcomes]
    C --> D[Operational effects]
    D --> E[Business outcomes]
    C --> C1[TP]
    C --> C2[FP]
    C --> C3[TN]
    C --> C4[FN]
    C1 --> E1[Recovered value or prevented loss]
    C2 --> E2[Manual work, friction, or wasted action]
    C4 --> E3[Escaped risk, missed revenue, or safety event]
```

## 1. Frame the task before you choose the metric

- Use a **binary** task when the live decision is fundamentally yes or no.
- Use a **multiclass** task when each example must land in one mutually exclusive bucket.
- Use a **multilabel** task when multiple labels can be true at once and ranking among labels matters.

The task framing determines which metrics are meaningful. A content moderation system with many simultaneous policy
labels should not be evaluated like a single-label classifier, and a routing system with one required destination
should not be evaluated like a multilabel tagging model.

## 2. Build a baseline stack, not a single baseline

A strong baseline stack usually includes all of the following:

- **Class-prior baseline**: always predict the majority class, or predict by label prevalence.
- **Current production baseline**: the rule system, heuristic, or current model that users already live with.
- **Simple trainable baseline**: a shallow, interpretable model that is hard to beat by accident.
- **Operational baseline**: the threshold or routing policy that satisfies today's staffing, SLA, or safety limits.
- **Human baseline**: when human review is involved, compare against actual reviewer precision, recall, or agreement.

The goal is not just to beat a weak number. The goal is to prove that the new system outperforms what the organization
already trusts.

## 3. Use metric families, not isolated metrics

A reliable classification evaluation usually combines three metric families:

### Threshold-free metrics

These answer, "Can the model rank better examples above worse ones?"

- [AUROC](../../../docs/source/classification/auroc.rst)
- [Average Precision](../../../docs/source/classification/average_precision.rst)
- [Precision-Recall Curve](../../../docs/source/classification/precision_recall_curve.rst)
- [ROC](../../../docs/source/classification/roc.rst)
- [Multilabel Ranking Average Precision](../../../docs/source/classification/label_ranking_average_precision.rst)
- [Multilabel Coverage Error](../../../docs/source/classification/coverage_error.rst)

### Thresholded decision metrics

These answer, "What happens at the operating point we would actually deploy?"

- [Accuracy](../../../docs/source/classification/accuracy.rst)
- [Precision](../../../docs/source/classification/precision.rst)
- [Recall](../../../docs/source/classification/recall.rst)
- [F1 Score](../../../docs/source/classification/f1_score.rst)
- [Matthews Correlation Coefficient](../../../docs/source/classification/matthews_corr_coef.rst)
- [Specificity](../../../docs/source/classification/specificity.rst)
- [Recall at Fixed Precision](../../../docs/source/classification/recall_at_fixed_precision.rst)
- [Precision at Fixed Recall](../../../docs/source/classification/precision_at_fixed_recall.rst)

### Diagnostics and reliability metrics

These answer, "Why does the system behave the way it does, and will operators trust it?"

- [Confusion Matrix](../../../docs/source/classification/confusion_matrix.rst)
- [Calibration Error](../../../docs/source/classification/calibration_error.rst)
- [Group Fairness](../../../docs/source/classification/group_fairness.rst)
- [Stat Scores](../../../docs/source/classification/stat_scores.rst)

## 4. Correlate model metrics to business metrics explicitly

For many teams, the cleanest correlation layer starts with outcome accounting:

```text
business_value
= TP * value_of_correct_action
- FP * cost_of_unnecessary_action
- FN * cost_of_missed_action
+ TN * value_of_safe_non_action
```

That equation becomes the bridge between model metrics and business metrics.

Examples:

- In fraud detection, `FP` increases review cost and customer friction, while `FN` increases fraud loss.
- In medical screening, `FN` carries safety risk, while `FP` increases downstream testing burden.
- In support routing, confusion between classes raises handle time, reroute rate, and SLA misses.
- In moderation, per-label `FN` increases policy escape rate, while per-label `FP` increases appeal rate and reviewer load.

If an offline metric improves but the equation above does not move in the right direction at the chosen threshold, the
metric is not aligned strongly enough with the business objective.

## 5. Validate the setup like production, not like a benchmark

Before launch, evaluation should answer these questions:

- Is the validation split **time-aware** and consistent with how data arrives in production?
- Are metrics reported by **important slices** such as geography, device type, merchant segment, hospital, or label?
- Are thresholds chosen against a **real operational constraint** such as review capacity, maximum miss rate, or SLA?
- Are scores sufficiently **calibrated** for any workflow that uses probabilities directly?
- Are rare but high-cost classes or labels visible in **macro** averages and per-class reports?

## 6. Example TorchMetrics setups

### Binary system with both ranking and operating-point checks

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryMatthewsCorrCoef,
    BinaryRecallAtFixedPrecision,
)

summary_metrics = MetricCollection(
    {
        "auroc": BinaryAUROC(),
        "average_precision": BinaryAveragePrecision(),
        "ece": BinaryCalibrationError(n_bins=15, norm="l1"),
        "mcc_at_0_35": BinaryMatthewsCorrCoef(threshold=0.35),
        "confusion_matrix_at_0_35": BinaryConfusionMatrix(threshold=0.35),
    }
)

summary = summary_metrics(preds, target)
recall_95p, threshold_95p = BinaryRecallAtFixedPrecision(min_precision=0.95)(preds, target)
```

### Multiclass system with class balance and routing diagnostics

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCalibrationError,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
)

routing_metrics = MetricCollection(
    {
        "top1_accuracy": MulticlassAccuracy(num_classes=12, average="micro"),
        "top3_accuracy": MulticlassAccuracy(num_classes=12, top_k=3, average="micro"),
        "macro_f1": MulticlassF1Score(num_classes=12, average="macro"),
        "mcc": MulticlassMatthewsCorrCoef(num_classes=12),
        "ece": MulticlassCalibrationError(num_classes=12, n_bins=20, norm="l1"),
        "confusion_matrix": MulticlassConfusionMatrix(num_classes=12),
    }
)
```

### Multilabel system with both classification and ranking behavior

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelCoverageError,
    MultilabelF1Score,
    MultilabelHammingDistance,
    MultilabelRankingAveragePrecision,
)

moderation_metrics = MetricCollection(
    {
        "macro_ap": MultilabelAveragePrecision(num_labels=8, average="macro"),
        "macro_f1_at_0_40": MultilabelF1Score(num_labels=8, average="macro", threshold=0.40),
        "macro_hamming_at_0_40": MultilabelHammingDistance(num_labels=8, average="macro", threshold=0.40),
        "ranking_ap": MultilabelRankingAveragePrecision(num_labels=8),
        "coverage_error": MultilabelCoverageError(num_labels=8),
    }
)
```

## 7. The minimum launch bar

A classification system is usually much closer to launch when all of these are true:

- It beats the current production baseline on the metric family that matches the real decision.
- It has a threshold policy tied to staffing, safety, or revenue constraints.
- It has calibration or confidence behavior that operators can trust.
- It has slice-level visibility for high-risk groups, classes, or labels.
- It has a monitoring plan for drift, threshold decay, and business outcomes after launch.
