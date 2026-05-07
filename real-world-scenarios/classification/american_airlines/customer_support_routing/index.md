# Scenario: Customer Support Routing for Reservations, Baggage, and Irregular Operations

Customer-support routing is typically a **multiclass classification** problem. Each contact needs one primary
destination queue, but airline contact centers have asymmetric queue costs because some intents are simple and others
require urgent disruption, baggage, or loyalty handling.

```{mermaid}
flowchart LR
    A[Customer call, chat, or message] --> B[Intent probabilities]
    B --> C[Top-1 queue]
    B --> D[Top-3 assist list]
    C --> E[Correct destination]
    C --> F[Wrong destination]
    F --> G[Reroute, delay, or repeat contact]
    D --> H[Agent-assisted routing]
```

## Why this scenario matters at airline scale

- Contact mix changes rapidly during storms, cancellations, and loyalty promotions.
- Multilingual demand and channel mix create different failure patterns across voice, chat, and digital messaging.
- Misrouting a disruption or baggage contact is much more expensive than misrouting a low-urgency reservation question.
- Label definitions can drift when queue maps or outsourcing partners change.

## How metrics fail at scale

Support-routing systems usually fail when teams:

- Report only micro accuracy, which is dominated by the largest queues.
- Ignore the confusion matrix and miss catastrophic confusions between baggage, disruption, and refund queues.
- Evaluate only top-1 even though many workflows are agent-assist rather than full automation.
- Ignore confidence calibration while still auto-routing low-confidence contacts.
- Benchmark against an outdated queue map and then launch into a changed operation.

## Why traditional metrics failed

The traditional metrics failed because they rewarded the most common contacts instead of the costliest mistakes:

- **Micro accuracy** passed because high-volume general-reservations contacts dominated.
- **Top-1-only evaluation** failed because the live workflow often relies on ranked assistance.
- **No confidence gating** failed because weak predictions were still forced into automation.
- **Static taxonomy evaluation** failed because queue definitions changed faster than the reporting logic.

## Metric stack that usually works

- [Multiclass Accuracy](../../../../docs/source/classification/accuracy.rst) for top-1 and top-k behavior.
- [Multiclass F1 Score](../../../../docs/source/classification/f1_score.rst) with `average="macro"` for rare but costly queues.
- [Multiclass Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) for operationally expensive confusions.
- [Multiclass Calibration Error](../../../../docs/source/classification/calibration_error.rst) when confidence gates automation.
- [Multiclass Cohen Kappa](../../../../docs/source/classification/cohen_kappa.rst) when label consistency varies across vendors.

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCalibrationError,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

contact_center_metrics = MetricCollection(
    {
        "top1_accuracy": MulticlassAccuracy(num_classes=14, average="micro"),
        "top3_accuracy": MulticlassAccuracy(num_classes=14, top_k=3, average="micro"),
        "macro_f1": MulticlassF1Score(num_classes=14, average="macro"),
        "cohen_kappa": MulticlassCohenKappa(num_classes=14),
        "ece": MulticlassCalibrationError(num_classes=14, n_bins=20, norm="l1"),
        "confusion_matrix": MulticlassConfusionMatrix(num_classes=14),
    }
)
```

## Why the updated production metrics passed

The updated production metrics passed because they matched the real contact-center workflow:

- [Multiclass Accuracy](../../../../docs/source/classification/accuracy.rst) with top-k passed because agent-assist routing uses shortlist quality.
- [Multiclass F1 Score](../../../../docs/source/classification/f1_score.rst) with `average="macro"` passed because expensive escalation queues stayed visible.
- [Multiclass Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) passed because not all misroutes are equally harmful.
- [Multiclass Calibration Error](../../../../docs/source/classification/calibration_error.rst) passed because the system needed safe abstain-versus-automate decisions.
- [Multiclass Cohen Kappa](../../../../docs/source/classification/cohen_kappa.rst) passed because queue labels are not perfectly stable across teams and vendors.

## How we established the relationship between updated production metrics and business metrics

We established the relationship by replaying historical contacts through the routing policy and measuring downstream
contact-center outcomes:

```text
reroute_rate = misrouted_contacts / total_contacts
average_handle_time = base_handle_time + reroute_penalty
repeat_contact_rate = repeated_contacts / total_contacts
sla_miss_rate = delayed_high_priority_contacts / total_contacts
net_contact_value = productivity_gain - reroute_cost - sla_penalty
```

The practical relationship was:

- Better **Top-3 Accuracy** correlated with lower agent triage time.
- Better **Macro F1** correlated with better protection for disruption and baggage queues.
- Better **Calibration Error** correlated with safer automation decisions.
- Better **Confusion Matrix** structure correlated with lower reroute and repeat-contact rates.

## How to handle the failures

The most useful MLOps controls are:

- Version the **queue taxonomy** and evaluate against the same mapping used live.
- Slice metrics by **channel, language, vendor, station, and disruption regime**.
- Use confidence thresholds to abstain when predictions are weak.
- Review confusion costs with operations leaders before launch.
- Canary auto-routing separately from agent-assist ranking.
- Monitor reroute rate, handle time, and repeat contacts after rollout.

## Production success criteria

This system is much more likely to ship well when:

- Rare high-priority queues stay visible in macro metrics.
- Agents see meaningful value from top-k suggestions.
- Reroutes and repeat contacts fall after launch.
- Confidence is calibrated enough for automation gates.
- Queue-map changes are reflected in the data and dashboards before rollout.
