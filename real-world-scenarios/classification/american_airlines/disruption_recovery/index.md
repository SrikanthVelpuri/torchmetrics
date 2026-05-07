# Scenario: Disruption Recovery Action Selection

Disruption recovery is usually a **multiclass classification** problem where the system predicts the most appropriate
next action for a disrupted itinerary, such as hold, same-day reaccommodation, partner reaccommodation, overnight
recovery, or manual specialist handling.

```{mermaid}
flowchart TD
    A[Disrupted itinerary and network state] --> B[Action probabilities]
    B --> C[Top-1 recovery action]
    B --> D[Top-3 planner shortlist]
    C --> E[Automated or guided action]
    C --> F[Wrong action with recovery cost]
    D --> G[Planner-assisted workflow]
```

## Why this scenario matters at airline scale

- The action classes are not equally costly; recommending a hotel stay when a same-day rebooking was possible is expensive.
- Label quality can be noisy because human planners may choose different acceptable actions for similar cases.
- Irregular operations change the action distribution dramatically across weather days and station conditions.
- The same model can behave differently at hubs, focus cities, and outstations because inventory and crew options differ.

## How metrics fail at scale

Recovery-action models usually fail when teams:

- Report only micro [Multiclass Accuracy](../../../../docs/source/classification/accuracy.rst), which is dominated by the most common recovery class.
- Ignore [Multiclass Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) patterns, even though some mistakes are far costlier than others.
- Evaluate only top-1 when planners actually use a shortlist.
- Ignore calibration and then auto-apply low-confidence recovery actions.
- Compare model versions across different disruption policies or inventory rules.

## Why traditional metrics failed

The traditional metrics failed because they assumed all recovery mistakes had the same cost:

- **Micro accuracy** passed because common classes such as routine same-day reaccommodation dominated.
- **Top-1-only evaluation** failed because planners often work from the top few suggestions.
- **No calibration check** failed because confidence is part of the decision to automate versus escalate.
- **Unversioned labels** failed because the meaning of the "correct" action changes when policy or inventory rules change.

## Metric stack that usually works

- [Multiclass Accuracy](../../../../docs/source/classification/accuracy.rst) for top-1 and top-k behavior.
- [Multiclass F1 Score](../../../../docs/source/classification/f1_score.rst) with `average="macro"` for rare but high-cost classes.
- [Multiclass Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) for action-cost inspection.
- [Multiclass Calibration Error](../../../../docs/source/classification/calibration_error.rst) when confidence gates automation.
- [Multiclass Cohen Kappa](../../../../docs/source/classification/cohen_kappa.rst) when human-label variability is material.
- [Multiclass Matthews CorrCoef](../../../../docs/source/classification/matthews_corr_coef.rst) as a balanced summary score.

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCalibrationError,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
)

recovery_metrics = MetricCollection(
    {
        "top1_accuracy": MulticlassAccuracy(num_classes=5, average="micro"),
        "top3_accuracy": MulticlassAccuracy(num_classes=5, top_k=3, average="micro"),
        "macro_f1": MulticlassF1Score(num_classes=5, average="macro"),
        "cohen_kappa": MulticlassCohenKappa(num_classes=5),
        "mcc": MulticlassMatthewsCorrCoef(num_classes=5),
        "ece": MulticlassCalibrationError(num_classes=5, n_bins=20, norm="l1"),
        "confusion_matrix": MulticlassConfusionMatrix(num_classes=5),
    }
)
```

## Why the updated production metrics passed

The updated production metrics passed because they matched how disruption planners actually work:

- [Multiclass Accuracy](../../../../docs/source/classification/accuracy.rst) with top-k passed because shortlist quality matters in planner-assisted recovery.
- [Multiclass F1 Score](../../../../docs/source/classification/f1_score.rst) with `average="macro"` passed because rare but expensive action classes remained visible.
- [Multiclass Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) passed because it exposed whether the model confused cheap versus expensive actions.
- [Multiclass Calibration Error](../../../../docs/source/classification/calibration_error.rst) passed because confidence determined whether the action could be automated.
- [Multiclass Cohen Kappa](../../../../docs/source/classification/cohen_kappa.rst) passed because the label set contained real human-policy ambiguity.

## How we established the relationship between updated production metrics and business metrics

We established the relationship by replaying disrupted itineraries through the candidate action policy and measuring the
passenger and operations outcomes of each decision path:

```text
recovery_cycle_time = time_to_protected_itinerary
overnight_cost = unnecessary_hotel_or_meal_cost
manual_escalation_load = routed_to_specialist_cases
same_day_recovery_rate = passengers_recovered_same_day / disrupted_passengers
net_recovery_value = passenger_recovery_gain - unnecessary_recovery_cost - escalation_cost
```

The linkage was:

- Better **Top-3 Accuracy** correlated with faster planner decision time.
- Better **Macro F1** correlated with fewer failures on rare but high-cost disruption cases.
- Better **Calibration Error** correlated with safer automation on high-confidence decisions.
- Better **Confusion Matrix** structure correlated with lower unnecessary overnight or partner-rebooking cost.

## How to handle the failures

The best MLOps controls are:

- Version the **recovery taxonomy and policy rules**.
- Evaluate by **hub, station, disruption type, aircraft family, and passenger importance tier**.
- Keep automation thresholds separate from planner-assist thresholds.
- Recalibrate after schedule changes or major policy changes.
- Review confusion matrices with operations-control stakeholders, not only ML stakeholders.
- Shadow-test on live disruption days before automating any action.

## Production success criteria

This system is much closer to production success when:

- Planner shortlists are useful on live disruption days.
- Rare, high-cost action classes remain visible in macro metrics.
- Automation happens only in well-calibrated score bands.
- Same-day recovery improves without excessive overnight or manual-escalation cost.
- The model remains stable across hubs and disruption types.
