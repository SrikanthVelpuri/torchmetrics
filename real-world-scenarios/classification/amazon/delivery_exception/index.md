# Scenario: Delivery Exception Prediction and Promise-Breach Triage

Delivery-exception systems are often **multiclass classification** problems where the model predicts whether an order
is on time or likely to breach promise because of carrier, weather, address, or station-level issues.

```{mermaid}
flowchart TD
    A[Shipment state and event stream] --> B[Exception probabilities]
    B --> C[Top-1 exception code]
    B --> D[Top-2 assistive suggestions]
    C --> E[Automated intervention]
    D --> F[Ops planner review]
    E --> G[Recovered promise or wasted action]
```

## Why this scenario is hard at scale

- The dominant class is usually "on time," which makes naive accuracy look stronger than the operational reality.
- Labels are noisy because stations and carriers do not always use exception codes consistently.
- Weather, route topology, and carrier mix create strong regional drift.
- A threshold that is safe in one station can flood another station with unnecessary alerts.

## How metrics fail at scale

Delivery models usually fail when teams:

- Optimize top-1 accuracy and ignore macro behavior on rare but costly exception classes.
- Evaluate on pooled data and miss station-specific or carrier-specific failures.
- Trust raw probabilities without calibration even though alerts are generated from score bands.
- Ignore label noise from inconsistent exception coding and then overreact to a small offline gain.
- Ship a model that improves classification metrics but increases intervention load without improving promise adherence.

## Why traditional metrics failed

The traditional metrics failed because they rewarded the easiest prediction and ignored operational cost:

- **Top-1 accuracy** passed because the dominant on-time class hid the failure modes that actually cost money.
- **Global pooled metrics** failed because station, carrier, and weather differences were hidden.
- **Uncalibrated probability use** failed because intervention policies were driven by score bands, not just class rank.
- **Ignoring label noise** failed because inconsistent exception codes made small metric deltas look more meaningful than they were.

## Worked Example

Suppose a delivery model scores `100,000` shipments:

```text
on_time = 96,000
carrier_delay = 2,000
address_issue = 1,000
station_backlog = 1,000
```

If the model predicts the dominant class very well but misses many rare exceptions, it can still post a strong
headline:

```text
top1_accuracy = 96.5%
station_backlog_recall = 300 / 1,000 = 30%
top2_accuracy = 98.4%
```

That is why the question is not just "How often is the top class correct?" The real question is whether the rare,
costly exception classes are being surfaced early enough to justify intervention.

## Questions I Asked Before Solving

Before I chose metrics or thresholds, I asked:

- What is the prevalence of each exception class by station, carrier, and weather regime?
- Is the workflow top-1 intervention, top-2 diagnosis support, or a hierarchical at-risk-then-type process?
- How costly is a false intervention relative to a missed true exception?
- Are station labels consistent enough to trust as ground truth, or do I need agreement-aware diagnostics?
- Will one threshold overload specific regions during storms or peak periods?

## Why Those Questions Are Backed by Math, Probability, or Research

These questions came directly from the operational math:

```text
macro_metric protects rare classes from dominant-class masking
expected_cost = FP * wasted_intervention_cost + FN * missed_exception_cost
alert_queue = predicted_at_risk_orders
observed_exception_rate(score_bin) = true_exceptions_in_bin / orders_in_bin
SE(class_recall_slice) ~= sqrt(recall_hat * (1 - recall_hat) / class_support_slice)
```

- I asked about per-class prevalence because rare-class masking is exactly why macro metrics matter.
- I asked about workflow type because top-1 action support and top-2 diagnosis support are different objectives.
- I asked about intervention cost because the threshold should follow expected value, not convenience.
- I asked about label consistency because weak agreement makes small metric changes hard to trust.
- I asked about regional overload because queue capacity depends on slice-specific class priors and traffic surges.

The research anchor is the shared [Deep Dive](../deep_dive/index): dominant-class problems need macro reasoning,
cost-sensitive thresholding, calibration checks, and slice-level uncertainty awareness.

## How the Questions Changed the Final Metric and Threshold Choice

Because of those answers, I did not accept top-1 accuracy as the main delivery launch metric.

- Because rare exception classes had to stay visible, I chose [Multiclass F1 Score](../../../../docs/source/classification/f1_score.rst) with `average="macro"`.
- Because planners often work with multiple likely diagnoses, I kept [Multiclass Accuracy](../../../../docs/source/classification/accuracy.rst) for both top-1 and top-k.
- Because intervention cost depends on which classes are being confused, I kept [Multiclass Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst).
- Because alert bands had to be operationally trustworthy, I chose [Multiclass Calibration Error](../../../../docs/source/classification/calibration_error.rst).
- Because station labels may be noisy, I kept [Multiclass Cohen Kappa](../../../../docs/source/classification/cohen_kappa.rst) alongside [Multiclass Matthews CorrCoef](../../../../docs/source/classification/matthews_corr_coef.rst).

Because of those answers, we chose macro-aware class monitoring, top-k assistive evaluation, and calibration-aware
thresholding instead of relying on one strong dominant-class accuracy number.

## Metric stack that usually works

- [Multiclass Accuracy](../../../../docs/source/classification/accuracy.rst) for top-1 and top-k operational workflows.
- [Multiclass F1 Score](../../../../docs/source/classification/f1_score.rst) with `average="macro"` to protect rare exception classes.
- [Multiclass Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) to understand which exception classes are being confused.
- [Multiclass Calibration Error](../../../../docs/source/classification/calibration_error.rst) for alert-band reliability.
- [Multiclass Cohen Kappa](../../../../docs/source/classification/cohen_kappa.rst) when label consistency varies across stations.
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

delivery_metrics = MetricCollection(
    {
        "top1_accuracy": MulticlassAccuracy(num_classes=6, average="micro"),
        "top2_accuracy": MulticlassAccuracy(num_classes=6, top_k=2, average="micro"),
        "macro_f1": MulticlassF1Score(num_classes=6, average="macro"),
        "cohen_kappa": MulticlassCohenKappa(num_classes=6),
        "mcc": MulticlassMatthewsCorrCoef(num_classes=6),
        "ece": MulticlassCalibrationError(num_classes=6, n_bins=20, norm="l1"),
        "confusion_matrix": MulticlassConfusionMatrix(num_classes=6),
    }
)
```

## Why the updated production metrics passed

The updated production metrics passed because they reflected planner workflow and operational risk:

- [Multiclass F1 Score](../../../../docs/source/classification/f1_score.rst) with `average="macro"` passed because rare but expensive exception classes stayed visible.
- [Multiclass Accuracy](../../../../docs/source/classification/accuracy.rst) with top-k passed because planners often need a short list of likely exception types.
- [Multiclass Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) passed because some class confusions are harmless while others trigger wasted interventions.
- [Multiclass Calibration Error](../../../../docs/source/classification/calibration_error.rst) passed because alert bands had to mean something stable in live planning.
- [Multiclass Cohen Kappa](../../../../docs/source/classification/cohen_kappa.rst) passed because it exposed whether the model improvement survived noisy station labels.

## How we established the relationship between updated production metrics and business metrics

We established the relationship by replaying shipment timelines and simulated intervention policies, then measuring
whether alerts actually improved promise outcomes:

```text
planner_alert_load = predicted_at_risk_orders
promise_breach_rate = breached_orders / total_orders
intervention_precision = recovered_promises / total_interventions
wasted_intervention_cost = unnecessary_interventions * average_intervention_cost
net_delivery_value = breach_reduction_value - wasted_intervention_cost
```

The resulting mapping was:

- Better **Macro F1** correlated with earlier detection of rare but high-cost exception classes.
- Better **Top-2 Accuracy** correlated with faster planner diagnosis in assistive workflows.
- Better **Calibration Error** correlated with more trustworthy alert bands and fewer wasted interventions.
- Better **Confusion Matrix** structure correlated with reduced misdirected operational actions.

## How to handle the failures

The MLOps controls that usually help most are:

- Validate with **time-aware, station-aware, and carrier-aware splits**.
- Report metrics by **station, region, carrier, weather zone, and fulfillment program**.
- Consider a **hierarchical design**: first predict on-time versus at-risk, then predict the likely exception class.
- Calibrate score bands before using them for automated interventions.
- Simulate alert volume and ops load so a recall improvement does not create dispatch fatigue.
- Audit label consistency with operations teams if Cohen kappa is weak, instead of treating noisy labels as truth.

## Production success criteria

This system is usually ready to scale when:

- High-risk routes are surfaced early without overwhelming planners.
- Rare exception classes stay visible in macro metrics.
- Alert bands are calibrated well enough for intervention policies.
- Metrics remain stable across carriers and stations, not just on the global average.
- Promise adherence, not just model score, improves after rollout.
