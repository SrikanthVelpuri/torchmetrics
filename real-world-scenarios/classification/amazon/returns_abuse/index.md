# Scenario: Returns Abuse, Refund Risk, and Concessions Control

Returns-abuse systems are also **binary classification** problems, but they differ from pure fraud systems because the
business is balancing abuse prevention against customer trust, refund latency, and category-specific return behavior.

```{mermaid}
flowchart TD
    A[Return or refund request] --> B[Abuse score]
    B -->|Below friction threshold| C[Auto-approve]
    B -->|Within review band| D[Manual review]
    B -->|Above action threshold| E[Restrict or escalate]
    C --> F[TN or FN]
    D --> G[TP or FP with review cost]
    E --> H[TP or FP with customer-friction cost]
```

## Why this scenario is hard at scale

- Labels often mature only after the return window closes and the item condition is verified.
- Behavior changes sharply across categories such as apparel, electronics, and consumables.
- Holiday periods can multiply review volume and change the base rate of abusive behavior.
- A false positive can damage a high-value customer experience even when the model is statistically "good."

## How metrics fail at scale

Returns-abuse systems usually fail when teams:

- Optimize [Binary F1 Score](../../../../docs/source/classification/f1_score.rst) on a balanced training sample and forget the live class ratio.
- Evaluate before labels mature, which makes abuse recall look weaker or stronger than it really is.
- Use one threshold across all categories even though abuse economics differ by product type and price band.
- Ignore calibration and then use the raw score to decide whether to slow refunds or trigger investigation.
- Launch a model that improves offline metrics but overloads the post-holiday review queue.

## Why traditional metrics failed

The traditional metrics failed because they were too far from the live refund decision:

- **Accuracy** and balanced-sample **F1** passed on curated evaluation sets but did not reflect the live abuse rate.
- **One shared threshold** failed because the cost of a false positive in luxury electronics is different from the cost in low-value consumables.
- **Uncalibrated scores** failed because refund and review workflows consumed score bands as if they were trustworthy risk estimates.
- **Global results** failed because category economics and seasonal prevalence shifts were hidden.

## Worked Example

Suppose the returns workflow sees `100,000` mature returns and the true abusive-return prevalence is `1.5%`.

```text
volume = 100,000
prevalence = 0.015
positives = 1,500
negatives = 98,500

if recall = 0.80 and specificity = 0.95:
TP = 1,200
FN = 300
FP = 4,925
precision = TP / (TP + FP) = 19.6%
review_load = TP + FP = 6,125
```

That sounds respectable until the business asks whether `6,125` reviews are acceptable during peak season and whether
`4,925` good customers can tolerate friction or delay. A model can improve recall and still damage the customer
experience if the threshold ignores queue capacity and refund-speed expectations.

## Questions I Asked Before Solving

Before I chose metrics or thresholds, I asked:

- How does abusive-return prevalence change by category, price band, season, and customer tenure?
- When do abuse labels become mature for each return category and price band?
- Do apparel, electronics, and consumables require different thresholds because their false-positive costs differ?
- How much customer-friction cost does one unnecessary review or restriction create?
- How many reviewed returns per day can operations absorb during normal weeks and holiday peaks?
- Are score bands being consumed directly by policy teams, and if so, how trustworthy are they by category and region?

## Why Those Questions Are Backed by Math, Probability, or Research

Those questions came directly from the operational math:

```text
prevalence(category) = abusive_returns_in_category / total_returns_in_category
precision = TP / (TP + FP)
expected_cost = FP * customer_friction_cost + FN * abuse_escape_cost
review_queue = TP + FP
observed_abuse_rate(score_bin) = positives_in_bin / examples_in_bin
SE(category_recall) ~= sqrt(recall_hat * (1 - recall_hat) / category_positives)
```

- I asked about prevalence drift because a category with a different base rate can produce a very different precision profile at the same threshold.
- I asked about label maturity because incomplete labels bias recall, precision, and calibration.
- I asked about category-specific thresholds because the expected-cost surface is different for luxury electronics versus low-value consumables.
- I asked about queue capacity because a review policy fails if the queue arrival rate exceeds operational throughput.
- I asked about calibration because score bands were being interpreted as direct policy signals.
- I asked about per-category uncertainty because a category with few abusive returns can give unstable slice estimates.

The research anchor is the shared [Deep Dive](../deep_dive/index): cost-sensitive thresholds should follow expected
loss, and imbalanced review systems are better described by precision-recall behavior than by accuracy alone.

## How the Questions Changed the Final Metric and Threshold Choice

Because of those answers, I did not choose one global F1 threshold and call the system ready.

- Because the queue had to stay clean under low prevalence, I chose [Binary Average Precision](../../../../docs/source/classification/average_precision.rst).
- Because the business needed to guarantee a minimum capture rate, I chose [Binary Precision at Fixed Recall](../../../../docs/source/classification/precision_at_fixed_recall.rst).
- Because customer-friction budgets imposed a hard ceiling, I also chose [Binary Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst).
- Because policy actions used score bands directly, I chose [Binary Calibration Error](../../../../docs/source/classification/calibration_error.rst) and threshold-specific [Binary Stat Scores](../../../../docs/source/classification/stat_scores.rst).

Because of those answers, we chose category-aware thresholds, mature-label evaluation windows, and a production
metric stack tied directly to queue health and refund friction.

## Metric stack that usually works

- [Binary Average Precision](../../../../docs/source/classification/average_precision.rst) for rare-event ranking quality.
- [Binary Precision at Fixed Recall](../../../../docs/source/classification/precision_at_fixed_recall.rst) when abuse capture must reach a minimum level.
- [Binary Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst) when customer-friction budgets impose a strong precision floor.
- [Binary Calibration Error](../../../../docs/source/classification/calibration_error.rst) because refund policies often consume score bands directly.
- [Binary Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) at category-specific operating thresholds.
- [Binary Stat Scores](../../../../docs/source/classification/stat_scores.rst) to audit TP, FP, TN, and FN counts during threshold simulation.

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryPrecisionAtFixedRecall,
    BinaryRecallAtFixedPrecision,
    BinaryStatScores,
)

returns_metrics = MetricCollection(
    {
        "average_precision": BinaryAveragePrecision(),
        "ece": BinaryCalibrationError(n_bins=25, norm="l1"),
        "stat_scores_at_0_88": BinaryStatScores(threshold=0.88),
        "confusion_matrix_at_0_88": BinaryConfusionMatrix(threshold=0.88),
    }
)

summary = returns_metrics(preds, target)
precision_70r, threshold_70r = BinaryPrecisionAtFixedRecall(min_recall=0.70)(preds, target)
recall_99p, threshold_99p = BinaryRecallAtFixedPrecision(min_precision=0.99)(preds, target)
```

## Why the updated production metrics passed

The updated production metrics passed because they matched the real policy and customer-trust tradeoff:

- [Binary Average Precision](../../../../docs/source/classification/average_precision.rst) passed because it measured how clean the top abuse queue was under heavy imbalance.
- [Binary Precision at Fixed Recall](../../../../docs/source/classification/precision_at_fixed_recall.rst) passed because the team could guarantee a minimum abuse-capture rate before asking whether the queue was affordable.
- [Binary Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst) passed because it enforced a customer-friction ceiling.
- [Binary Calibration Error](../../../../docs/source/classification/calibration_error.rst) and [Binary Stat Scores](../../../../docs/source/classification/stat_scores.rst) passed because the business operated on score bands and explicit TP, FP, TN, and FN counts.

## How we established the relationship between updated production metrics and business metrics

We tied the updated production metrics to business outcomes by threshold-sweeping mature windows and computing the
customer and operations impact of each operating point:

```text
review_load = TP + FP
refund_delay_cost = reviewed_good_customers * average_delay_cost
abuse_prevented_value = TP * average_abuse_loss_prevented
abuse_escape_cost = FN * average_abuse_loss_missed
net_refund_value = abuse_prevented_value - review_cost - refund_delay_cost
```

That gave us a clear interpretation:

- Better **Average Precision** meant more abusive returns surfaced per manual review slot.
- Better **Recall at Fixed Precision** meant higher abuse capture without breaking the customer-friction budget.
- Better **Calibration Error** meant policy teams could trust risk bands for auto-approve, review, and restrict decisions.

## How to handle the failures

The most useful MLOps defenses are:

- Build evaluation windows around **label maturity**, not just around event time.
- Simulate **queue growth and refund delay** at each threshold before rollout.
- Maintain thresholds by **category, price band, customer tenure, and region** when the economics are meaningfully different.
- Compare against the current concessions workflow, not just against another model.
- Monitor **appeal rate, refund turnaround time, and review backlog** alongside model metrics.
- Refresh thresholds after known seasonal shifts instead of assuming the calibration from October still holds in January.

## Production success criteria

The deployment is usually healthy when:

- Abuse capture improves without unacceptable refund friction for good customers.
- Manual-review volume stays within SLA through peak seasons.
- Score bands remain calibrated enough for policy teams to trust them.
- Category-level metrics do not collapse on long-tail or high-cost categories.
- Retraining waits for mature labels instead of overfitting to incomplete outcomes.
