# Scenario: Customer Support Routing and Operational Triage

Support routing is typically a **multiclass classification** problem. Every ticket must land in one destination queue,
but the cost of mistakes is not uniform. Misrouting a rare escalation class may be much worse than confusing two common
low-risk queues.

```{mermaid}
flowchart LR
    A[Incoming ticket] --> B[Class probabilities]
    B --> C[Top-1 route]
    B --> D[Top-k suggestions]
    C --> E[Correct queue]
    C --> F[Wrong queue]
    F --> G[Reroute, delay, or escalation]
    D --> H[Agent-assisted routing]
```

## Why multiclass routing often succeeds

These projects reach production when the team evaluates both automation quality and operator assistance:

- Top-1 accuracy matters for full automation.
- Top-k accuracy matters when agents choose from a ranked shortlist.
- Macro metrics matter because minority queues can be operationally expensive.
- Confusion matrices reveal whether errors are harmless neighbors or serious escalations.

## Why multiclass routing often fails

These systems often fail when teams:

- Report only micro accuracy, which is dominated by the largest queues.
- Ignore calibration, so automation trusts low-confidence predictions.
- Never inspect the confusion matrix and miss catastrophic confusions.
- Overlook the difference between top-1 automation and top-k assistive workflows.

## Baseline setup

Good routing baselines normally include:

- **Majority queue baseline** or queue-prior routing.
- **Current rules or keyword router** in production today.
- **Simple linear baseline** for interpretability.
- **Agent-assist baseline** that measures whether top-k suggestions reduce manual routing time.

The validation split should reflect changes in product taxonomy, queue definitions, and seasonal ticket mix.

## Metrics that matter

- [Multiclass Accuracy](../../../docs/source/classification/accuracy.rst) for top-1 and top-k routing success.
- [Multiclass F1 Score](../../../docs/source/classification/f1_score.rst) with `average="macro"` to expose minority-class behavior.
- [Multiclass Confusion Matrix](../../../docs/source/classification/confusion_matrix.rst) to inspect operationally costly confusions.
- [Multiclass Matthews CorrCoef](../../../docs/source/classification/matthews_corr_coef.rst) as a balanced summary score.
- [Multiclass Calibration Error](../../../docs/source/classification/calibration_error.rst) if confidence gates automation or escalation.

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

## Connecting the metrics to business outcomes

For routing systems, model quality usually maps into:

```text
reroute_rate = misrouted_tickets / total_tickets
average_handle_time = base_time + extra_time_from_misroutes
sla_risk = tickets_that_enter_the_wrong_queue_late
agent_productivity = reduction_in_manual_triage_time
```

This creates a direct interpretation for the metrics:

- **Top-1 accuracy** supports end-to-end automation.
- **Top-k accuracy** supports agent-assist workflows.
- **Macro F1** prevents rare queues from disappearing inside common-class averages.
- **Confusion Matrix** shows where SLA or escalation cost actually comes from.

## Production-ready path

Routing systems tend to ship well when:

- Top-1 or top-k metrics are chosen to match the actual product workflow.
- The confusion matrix is reviewed with operations leaders, not just model builders.
- Rare but costly queues are protected by macro metrics or queue-level targets.
- Confidence is used to abstain or escalate instead of forcing weak predictions.

## Failure pattern and learning

The common failed pattern is a model that looks great on overall accuracy but still creates expensive reroutes or hides
poor performance on rare queues. The learning is that operational routing needs cost-aware multiclass evaluation, not
just one aggregate number.
