# Scenario: Flight Delay Risk and Early Irregular-Operations Triage

Flight-delay prediction is usually a **binary classification** problem: predict whether a flight or turn is at high
risk of breaching the operational threshold that matters to dispatch, stations, or customer-connection planning.

```{mermaid}
flowchart LR
    A[Flight state and network context] --> B[Delay risk score]
    B -->|Below watch threshold| C[Normal monitoring]
    B -->|Within mitigation band| D[Ops review]
    B -->|Above intervention threshold| E[Preemptive action]
    C --> F[TN or FN]
    D --> G[TP or FP with planner cost]
    E --> H[TP or FP with intervention cost]
```

## Why this scenario matters at airline scale

- Delay prevalence can change sharply with weather, hub congestion, aircraft routing, and crew legality constraints.
- Labels can be deceptively noisy if the threshold definition changes across teams such as departure delay, arrival delay, or controllable delay.
- A false positive can trigger wasted mitigation work, while a false negative can cascade into misconnections and network instability.
- Strong global metrics can still hide poor performance at one hub or in one weather pattern.

## How metrics fail at scale

Delay-risk models usually fail in production when teams:

- Rely on [Binary AUROC](../../../../docs/source/classification/auroc.rst) alone even though the live question is whether the top-risk band is actionable.
- Tune one threshold on normal operations and keep it unchanged through storms, holidays, or fleet disruptions.
- Report one network-wide score and miss weak recall at a particular hub, aircraft family, or departure bank.
- Evaluate on random splits that leak recurring routes, tails, or station patterns across train and validation.
- Ignore calibration even though controllers and planners interpret the score as an operational probability.

## Why traditional metrics failed

The traditional metrics failed because they over-rewarded ranking quality and under-protected the live intervention decision:

- **Accuracy** passed because most flights are not severely delayed on most days.
- **AUROC alone** passed offline but failed in operations because it did not tell us whether high-risk flights were precise enough to justify intervention.
- **One global threshold** failed because intervention cost and congestion profile vary by station and operating regime.
- **Global averages** failed because hub-specific degradation was hidden inside a healthy network number.

## Metric stack that usually works

- [Binary Average Precision](../../../../docs/source/classification/average_precision.rst) for actionable ranking quality in the top-risk band.
- [Binary Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst) when the operation needs strong trust before dispatching mitigation work.
- [Binary Calibration Error](../../../../docs/source/classification/calibration_error.rst) because planners consume score bands directly.
- [Binary Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) at the deployed threshold.
- [Binary Matthews CorrCoef](../../../../docs/source/classification/matthews_corr_coef.rst) as a balanced threshold summary.

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryMatthewsCorrCoef,
    BinaryRecallAtFixedPrecision,
)

delay_metrics = MetricCollection(
    {
        "average_precision": BinaryAveragePrecision(),
        "ece": BinaryCalibrationError(n_bins=20, norm="l1"),
        "mcc_at_0_78": BinaryMatthewsCorrCoef(threshold=0.78),
        "confusion_matrix_at_0_78": BinaryConfusionMatrix(threshold=0.78),
    }
)

summary = delay_metrics(preds, target)
recall_95p, threshold_95p = BinaryRecallAtFixedPrecision(min_precision=0.95)(preds, target)
```

## Why the updated production metrics passed

The updated production metrics passed because each one covered an operational failure mode:

- [Binary Average Precision](../../../../docs/source/classification/average_precision.rst) passed because it measured whether the top alert queue was worth planner attention.
- [Binary Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst) passed because it enforced a minimum trust level before triggering mitigations.
- [Binary Calibration Error](../../../../docs/source/classification/calibration_error.rst) passed because planners needed stable probability bands, not only ranks.
- [Binary Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) and [Binary Matthews CorrCoef](../../../../docs/source/classification/matthews_corr_coef.rst) passed because they described the actual thresholded production policy.

## How we established the relationship between updated production metrics and business metrics

We established the relationship by replaying recent, time-ordered operations days and translating each threshold into
intervention and network outcomes:

```text
planner_alert_load = TP + FP
avoidable_delay_minutes = TP * average_minutes_recovered
wasted_intervention_cost = FP * average_intervention_cost
missed_cascade_cost = FN * average_connection_and_crew_impact
net_operational_value = avoidable_delay_minutes_value - wasted_intervention_cost - missed_cascade_cost
```

The observed relationship was:

- Better **Average Precision** correlated with more recoverable delay minutes per alert reviewed.
- Better **Recall at Fixed Precision** correlated with fewer missed delay cascades while keeping planner workload stable.
- Better **Calibration Error** correlated with more reliable action bands for dispatch and station teams.

## How to handle the failures

The strongest MLOps controls are usually:

- Use **time-aware, route-aware, and tail-aware splits**.
- Track metrics by **hub, station, departure bank, aircraft family, weather regime, and fleet subtype**.
- Retune thresholds after major schedule changes, storm seasons, and irregular-operations events.
- Compare score-band calibration against actual delay incidence after launch.
- Simulate planner queue volume before rollout so better recall does not create alert fatigue.
- Canary on a subset of stations before using the model network-wide.

## Production success criteria

This system is much more likely to succeed when:

- High-risk bands identify flights where mitigation is actually worth the operational cost.
- Alert volume stays within planner and station capacity.
- Hub- and weather-specific slices remain stable.
- Calibration remains good enough for teams to trust action bands.
- Delay reduction and connection protection improve after launch, not just the offline score.
