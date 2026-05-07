# Scenario: Customer Support Routing Across Marketplaces and Languages

Support-routing systems are usually **multiclass classification** problems. Every contact needs one primary route, but
the cost of confusion is asymmetric because some queues trigger credits, some are easy reroutes, and some are
time-critical escalations.

```{mermaid}
flowchart TD
    A[Customer contact] --> B[Intent probabilities]
    B --> C[Top-1 route]
    B --> D[Top-3 assist list]
    C --> E[Correct queue]
    C --> F[Wrong queue]
    F --> G[Reroute, delay, or SLA miss]
    D --> H[Agent-assist workflow]
```

## Why this scenario is hard at scale

- Contact volume changes quickly during sales events, weather disruptions, and carrier incidents.
- Intent taxonomies evolve as policies, products, and tooling change.
- Marketplaces and languages have different dominant intents, which can make a global metric misleading.
- A high-confidence wrong route can be worse than a low-confidence abstention.

## How metrics fail at scale

Routing models usually fail when teams:

- Report only micro [Multiclass Accuracy](../../../../docs/source/classification/accuracy.rst), which is dominated by the largest queues.
- Never inspect the [Multiclass Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst), so they miss catastrophic confusions between refund, fraud, and delivery-escalation queues.
- Ignore calibration and then auto-route low-confidence predictions anyway.
- Compare against a frozen taxonomy even though the production queue map changed.
- Evaluate on English-heavy data and launch globally.

## Why traditional metrics failed

The traditional metrics failed because they over-rewarded common queues and ignored the live workflow:

- **Micro accuracy** passed because the largest queues dominated the count.
- **Top-1-only evaluation** failed because much of the actual workflow was agent assist, not full automation.
- **No calibration check** failed because the system still used confidence to decide whether to auto-route or abstain.
- **Static taxonomy evaluation** failed because even a good model score is misleading if the production queue map changed.

## Worked Example

Suppose a routing system handles `10,000` tickets in a day.

```text
top1_accuracy = 82%
top3_accuracy = 97%
top1_misroutes = 1,800
not_in_top3 = 300
```

Now suppose `150` of the `1,800` top-1 misses are expensive confusions where fraud, refund, or escalation contacts are
sent into a general-support queue. The headline top-1 number says the model is decent. The workflow-level view says a
small set of costly confusions still matters a lot more than the average miss.

## Questions I Asked Before Solving

Before I chose metrics or thresholds, I asked:

- Is the live workflow full automation, agent assist, or a mixture of both?
- How different are queue priors by language, marketplace, and channel?
- Which class confusions are harmless neighbors, and which ones create expensive reroutes or SLA risk?
- Do we need an abstain band for low-confidence predictions?
- Are queue definitions stable across locales, marketplaces, and time?
- Do I have enough data by language and rare queue to trust slice-level performance estimates?

## Why Those Questions Are Backed by Math, Probability, or Research

These questions came straight from the underlying decision structure:

```text
top1_success = correct_top_class / total_tickets
topk_success = tickets_with_true_queue_in_topk / total_tickets
queue_prior(language, queue) = tickets_in_queue_and_language / tickets_in_language
expected_cost = sum(confusion_count_ij * cost_ij)
triage_queue = abstained_tickets + rerouted_tickets
observed_correct_rate(score_bin) = correct_predictions_in_bin / predictions_in_bin
SE(queue_recall) ~= sqrt(recall_hat * (1 - recall_hat) / queue_support)
```

- I asked about automation versus assist because top-1 and top-k are different success events.
- I asked about queue priors because a rare expensive queue can disappear inside the global average if I do not look at conditional class probability by slice.
- I asked about confusion costs because multiclass routing is almost never a uniform 0/1 loss problem in production.
- I asked about abstention because low-confidence automation is a calibration problem, not just an accuracy problem.
- I asked about taxonomy stability because if queues change, historical labels no longer represent the same target.
- I asked about language and rare-queue support because slice estimates can be unstable when support is small.

The research anchor is the shared [Deep Dive](../deep_dive/index): once the workflow is assistive or cost-asymmetric,
aggregate micro accuracy is not enough to protect the production decision.

## How the Questions Changed the Final Metric and Threshold Choice

Because of those answers, I did not use micro top-1 accuracy as the only launch bar.

- Because the workflow mixed automation and assistive routing, I chose [Multiclass Accuracy](../../../../docs/source/classification/accuracy.rst) for both top-1 and top-k.
- Because rare but expensive queues had to stay visible, I chose [Multiclass F1 Score](../../../../docs/source/classification/f1_score.rst) with `average="macro"`.
- Because confusion costs were not uniform, I kept [Multiclass Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) as a primary diagnostic.
- Because auto-routing depended on confidence, I chose [Multiclass Calibration Error](../../../../docs/source/classification/calibration_error.rst).
- Because label consistency across agents and vendors mattered, I kept [Multiclass Cohen Kappa](../../../../docs/source/classification/cohen_kappa.rst).

Because of those answers, we chose a top-1 plus top-k evaluation stack, explicit abstain handling, and cost-aware
confusion inspection rather than one headline routing number.

## Metric stack that usually works

- [Multiclass Accuracy](../../../../docs/source/classification/accuracy.rst) for top-1 and top-k routing behavior.
- [Multiclass F1 Score](../../../../docs/source/classification/f1_score.rst) with `average="macro"` so rare queues stay visible.
- [Multiclass Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) for cost-aware error inspection.
- [Multiclass Calibration Error](../../../../docs/source/classification/calibration_error.rst) if confidence gates automation.
- [Multiclass Matthews CorrCoef](../../../../docs/source/classification/matthews_corr_coef.rst) as a balanced overall summary.
- [Multiclass Cohen Kappa](../../../../docs/source/classification/cohen_kappa.rst) when label noise from multiple human taxonomies is a concern.

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

support_metrics = MetricCollection(
    {
        "top1_accuracy": MulticlassAccuracy(num_classes=18, average="micro"),
        "top3_accuracy": MulticlassAccuracy(num_classes=18, top_k=3, average="micro"),
        "macro_f1": MulticlassF1Score(num_classes=18, average="macro"),
        "mcc": MulticlassMatthewsCorrCoef(num_classes=18),
        "cohen_kappa": MulticlassCohenKappa(num_classes=18),
        "ece": MulticlassCalibrationError(num_classes=18, n_bins=20, norm="l1"),
        "confusion_matrix": MulticlassConfusionMatrix(num_classes=18),
    }
)
```

## Why the updated production metrics passed

The updated production metrics passed because they matched both automation and assistive routing behavior:

- [Multiclass Accuracy](../../../../docs/source/classification/accuracy.rst) for top-1 and top-k passed because it separately measured full automation and shortlist usefulness.
- [Multiclass F1 Score](../../../../docs/source/classification/f1_score.rst) with `average="macro"` passed because rare but costly queues stayed visible.
- [Multiclass Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) passed because it exposed which queue confusions were operationally expensive.
- [Multiclass Calibration Error](../../../../docs/source/classification/calibration_error.rst) passed because automation gates depended on score confidence, not only class rank.
- [Multiclass Cohen Kappa](../../../../docs/source/classification/cohen_kappa.rst) passed because queue labels were not perfectly consistent across agents and vendors.

## How we established the relationship between updated production metrics and business metrics

We established the relationship by replaying historical contacts through the candidate routing policy and measuring the
downstream contact-center outcome for each threshold and queue grouping:

```text
reroute_rate = misrouted_contacts / total_contacts
average_handle_time = baseline_handle_time + reroute_penalty
sla_miss_rate = late_contacts / total_contacts
agent_triage_time_saved = manual_triage_time_before - manual_triage_time_after
net_routing_value = productivity_gain - reroute_cost - sla_penalty
```

The relationship became clear in practice:

- Better **Top-3 Accuracy** correlated with lower agent triage time in assistive workflows.
- Better **Macro F1** correlated with fewer failures on rare escalation queues.
- Better **Calibration Error** correlated with safer abstain-versus-auto-route decisions.
- Better **Confusion Matrix** structure correlated with lower reroute rate and fewer expensive queue mistakes.

## How to handle the failures

The strongest MLOps moves are usually:

- Maintain a **versioned queue taxonomy** and backtest against the same taxonomy the model will see live.
- Track metrics by **marketplace, language, contact channel, and agent vendor**.
- Use confidence thresholds to **abstain or route to human triage** instead of forcing weak top-1 predictions.
- Review the confusion matrix with operations leaders so the cost of each confusion is explicit.
- Canary auto-routing separately from agent-assist ranking because top-1 and top-k quality solve different workflows.
- Monitor reroute rate, average handle time, and SLA misses alongside the model metrics.

## Production success criteria

This system is much closer to a successful launch when:

- Top-1 or top-k metrics match the actual contact-center workflow.
- Rare escalation queues remain visible in macro metrics and confusion reports.
- Confidence is calibrated well enough for automation gates.
- Queue-map changes are reflected in data pipelines and dashboards before rollout.
- The model reduces reroutes or handle time without increasing downstream escalations.
