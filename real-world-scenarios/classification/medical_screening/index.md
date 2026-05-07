# Scenario: Medical Screening and Safety-Critical Triage

Medical screening is usually a **binary classification** problem, but unlike fraud detection the main operational goal
is often to miss as few true cases as possible while keeping downstream follow-up load manageable. That changes the
metric mix dramatically.

```{mermaid}
flowchart TD
    A[Patient or case] --> B[Risk score]
    B -->|Below screening threshold| C[Negative screen]
    B -->|Above screening threshold| D[Follow-up test or clinician review]
    C --> E[TN or FN]
    D --> F[TP or FP]
    E --> G[Safety risk if FN]
    F --> H[Clinical workload and cost]
```

## What good looks like

A medical screening model is usually healthy when it:

- Maintains high recall for the condition of interest.
- Preserves a strong negative predictive value for patients who are screened out.
- Uses specificity as a workload control, not as the primary success definition.
- Is checked across hospitals, age groups, devices, and other protected or clinically relevant subgroups.

## What usually causes production failure

Medical screening projects fail when teams:

- Optimize **accuracy** on a relatively balanced benchmark rather than recall or safety-oriented metrics.
- Tune thresholds on random splits that leak patient or temporal information.
- Ignore calibration and subgroup disparities, creating uneven miss rates in deployment.
- Report one overall number and miss site-specific or demographic failure modes.

## Baseline setup

A strong screening baseline often contains:

- **Current clinical heuristic** or legacy scorecard.
- **Always-positive or low-threshold baseline** to show the recall-workload frontier.
- **Simple interpretable baseline** for clinician trust and sanity checking.
- **Hospital or site-specific baselines** when deployment spans multiple operating environments.

For this kind of system, label definition and label latency matter as much as the model. A label that arrives months
after the initial decision can make an offline score look better than the real triage workflow it is supposed to support.

## Metrics that should be central

- [Binary Recall](../../../docs/source/classification/recall.rst) because missed positives are often the dominant risk.
- [Binary Sensitivity at Specificity](../../../docs/source/classification/sensitivity_at_specificity.rst) when the workflow must keep a minimum specificity floor.
- [Binary Negative Predictive Value](../../../docs/source/classification/negative_predictive_value.rst) when negative screens need to be safe to trust.
- [Binary Calibration Error](../../../docs/source/classification/calibration_error.rst) when clinicians interpret the score as risk.
- [Binary Confusion Matrix](../../../docs/source/classification/confusion_matrix.rst) at the deployment threshold.
- [Binary Fairness](../../../docs/source/classification/group_fairness.rst) for subgroup parity checks such as equal opportunity.

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryFairness,
    BinaryNegativePredictiveValue,
    BinaryRecall,
    BinarySensitivityAtSpecificity,
)

screening_metrics = MetricCollection(
    {
        "recall_at_0_20": BinaryRecall(threshold=0.20),
        "npv_at_0_20": BinaryNegativePredictiveValue(threshold=0.20),
        "ece": BinaryCalibrationError(n_bins=15, norm="l1"),
        "confusion_matrix_at_0_20": BinaryConfusionMatrix(threshold=0.20),
    }
)

summary = screening_metrics(preds, target)
sensitivity_95spec, operating_threshold = BinarySensitivityAtSpecificity(min_specificity=0.95)(preds, target)
fairness = BinaryFairness(num_groups=4, task="equal_opportunity", threshold=0.20)(preds, target, groups)
```

## Business and clinical correlation

The cleanest bridge from technical metrics to clinical operations is:

```text
missed_cases = FN
avoidable_follow_ups = FP
safe_negative_rate = NPV
follow_up_load = TP + FP
clinical_value = cases_detected_early - unnecessary_follow_ups - missed_case_harm
```

In this setting:

- **Recall** maps to case capture.
- **Negative Predictive Value** maps to the trustworthiness of a negative screen.
- **Specificity** or **Sensitivity at Specificity** helps cap follow-up burden.
- **Fairness** matters because unequal recall across groups can become an unacceptable safety failure.

## Production-ready path

These systems are more likely to ship when:

- Thresholds are selected jointly with clinicians or operations leaders.
- Recall and negative predictive value beat the current baseline on recent data.
- Site-level and subgroup-level gaps are visible before launch.
- Calibration is acceptable for how the score will actually be interpreted.

## Failure pattern and learning

The recurring failed pattern is a model that improves headline accuracy while worsening missed cases or subgroup recall.
The learning is that in safety-critical screening, the deployment threshold is a clinical policy decision, and metrics
must reflect that policy directly.
