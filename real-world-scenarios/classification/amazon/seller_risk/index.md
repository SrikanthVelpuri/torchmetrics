# Scenario: Seller Risk, Account Takeover, and Policy Abuse

Seller-risk systems are usually **binary classification** problems with extreme class imbalance, delayed labels, and
high operational cost when a false positive blocks a legitimate seller during a peak event.

```{mermaid}
flowchart LR
    A[Seller or listing event] --> B[Risk score]
    B -->|Below monitor threshold| C[Allow]
    B -->|Within investigation band| D[Manual investigation]
    B -->|Above action threshold| E[Hold or suspend]
    C --> F[TN or FN]
    D --> G[TP or FP with analyst cost]
    E --> H[TP or FP with seller-friction cost]
```

## Why this scenario matters at Amazon scale

- Millions of seller, payment, device, and listing events can arrive each day.
- Positives are rare, but the cost of missing coordinated abuse can be very high.
- Labels mature late because many abuse cases are confirmed only after investigation or downstream customer impact.
- Marketplace-specific behavior means a strong global metric can still hide poor performance in one region or seller segment.

## How metrics fail at scale

Seller-risk models usually fail in production when teams:

- Celebrate a strong [Binary AUROC](../../../../docs/source/classification/auroc.rst) while
  [Binary Average Precision](../../../../docs/source/classification/average_precision.rst) collapses under rare-event prevalence.
- Use random splits that leak seller identity, device clusters, or repeated abuse campaigns across train and validation.
- Tune a threshold on a normal week and keep it fixed through Prime Day or holiday shifts.
- Report one global score and miss weak recall on new sellers, cross-border sellers, or newly launched marketplaces.
- Retrain too early on immature labels, causing the model to learn from incomplete investigations.

## Why traditional metrics failed

The traditional metric stack failed for predictable reasons:

- **Accuracy** passed offline because the legitimate-seller class dominated the dataset.
- **AUROC alone** passed research review but failed operations review because it did not say whether the top score bands were precise enough to justify holds or investigations.
- **One global threshold** looked simple, but it ignored analyst capacity and seller-friction budgets.
- **Global averages** passed dashboards while weak recall on new-seller and cross-border slices still created real abuse escapes.

## Worked Example

Suppose the seller-risk pipeline scores `1,000,000` mature seller events in a day and the true abuse prevalence is
`0.1%`.

```text
volume = 1,000,000
prevalence = 0.001
positives = 1,000
negatives = 999,000

if recall = 0.92 and specificity = 0.994:
TP = 920
FN = 80
FP = 5,994
precision = TP / (TP + FP) = 13.3%
review_queue = TP + FP = 6,914
```

That is the kind of offline result that looks encouraging in a notebook but fails instantly in production if the
investigation team can only process `2,000` cases per day. The model is not judged only by recall. It is judged by
whether the resulting queue is economically and operationally survivable.

## Questions I Asked Before Solving

Before I chose metrics or thresholds, I asked:

- What is the mature abuse prevalence by marketplace, seller age, and fulfillment cohort?
- What precision floor does the analyst queue require to stay within staffing limits?
- What is the expected cost of a false hold compared with the cost of a missed abuse case?
- Do score bands like `0.90+` actually correspond to high observed abuse rates after calibration?
- Which cohorts are most dangerous if recall drops, and do I have enough positive examples in each slice to trust the estimate?

## Why Those Questions Are Backed by Math, Probability, or Research

These questions were not guesswork. They came directly from the math of the problem:

```text
precision = TP / (TP + FP)
expected_cost = FP * seller_friction_cost + FN * missed_abuse_cost
queue_load = TP + FP
observed_abuse_rate(score_bin) = positives_in_bin / examples_in_bin
SE(slice_recall) ~= sqrt(recall_hat * (1 - recall_hat) / slice_positives)
```

- I asked about prevalence because in rare-event systems precision changes dramatically when the base rate changes.
- I asked about false-hold cost because the optimal decision threshold depends on expected cost, not just class rank.
- I asked about analyst capacity because queue overload is a first-order production failure mode.
- I asked about calibration because if a risk band is consumed as probability, then the observed abuse rate inside that band must be stable.
- I asked about cohort support because a slice with very few positives can produce a recall estimate that looks precise but is statistically noisy.

The research anchor is the same one described in [Deep Dive](../deep_dive/index): imbalanced learning favors
precision-recall thinking over accuracy, and cost-sensitive deployment requires probability-aware thresholding.

## How the Questions Changed the Final Metric and Threshold Choice

Because of those answers, I did not use accuracy or AUROC alone as the production gate.

- Because prevalence was tiny and queue quality mattered, I chose [Binary Average Precision](../../../../docs/source/classification/average_precision.rst).
- Because seller friction imposed a hard precision floor, I chose [Binary Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst).
- Because policy teams consumed risk bands directly, I chose [Binary Calibration Error](../../../../docs/source/classification/calibration_error.rst).
- Because the actual deployed threshold had to fit analyst capacity, I kept [Binary Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) and [Binary Matthews CorrCoef](../../../../docs/source/classification/matthews_corr_coef.rst) at the operating threshold.

Because of those answers, we chose a capacity-aware thresholding policy and a metric stack that protects ranking
quality, threshold quality, calibration quality, and slice stability at the same time.

## Metric stack that usually works

- [Binary Average Precision](../../../../docs/source/classification/average_precision.rst) for ranking quality under extreme imbalance.
- [Binary Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst) when seller friction demands a hard precision floor.
- [Binary Calibration Error](../../../../docs/source/classification/calibration_error.rst) for risk-band trustworthiness.
- [Binary Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) at the deployed threshold.
- [Binary Matthews CorrCoef](../../../../docs/source/classification/matthews_corr_coef.rst) as a balanced single-number threshold summary.
- [Binary Fairness](../../../../docs/source/classification/group_fairness.rst) if the team needs visibility by seller cohort or marketplace group.

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryFairness,
    BinaryMatthewsCorrCoef,
    BinaryRecallAtFixedPrecision,
)

seller_risk_metrics = MetricCollection(
    {
        "average_precision": BinaryAveragePrecision(),
        "ece": BinaryCalibrationError(n_bins=20, norm="l1"),
        "mcc_at_0_92": BinaryMatthewsCorrCoef(threshold=0.92),
        "confusion_matrix_at_0_92": BinaryConfusionMatrix(threshold=0.92),
    }
)

summary = seller_risk_metrics(preds, target)
recall_995p, threshold_995p = BinaryRecallAtFixedPrecision(min_precision=0.995)(preds, target)
fairness = BinaryFairness(num_groups=5, task="equal_opportunity", threshold=0.92)(preds, target, groups)
```

## Why the updated production metrics passed

The updated metric stack passed because each metric protected a specific production risk:

- [Binary Average Precision](../../../../docs/source/classification/average_precision.rst) passed because it measured the quality of the top-risk bands where investigators actually spend time.
- [Binary Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst) passed because it enforced a seller-friction guardrail instead of letting recall rise by creating too many false actions.
- [Binary Calibration Error](../../../../docs/source/classification/calibration_error.rst) passed because policy teams needed score bands that mapped to stable hit rates.
- [Binary Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) and [Binary Matthews CorrCoef](../../../../docs/source/classification/matthews_corr_coef.rst) passed because they reflected the real deployed threshold instead of only ranking quality.

## How we established the relationship between updated production metrics and business metrics

We established the relationship by replaying mature, time-ordered traffic and translating confusion-matrix outcomes into
seller operations outcomes:

```text
investigation_load = TP + FP
seller_friction_rate = FP / legitimate_sellers
prevented_abuse_value = TP * average_loss_prevented
missed_abuse_cost = FN * average_loss_per_missed_case
net_operating_value = prevented_abuse_value - analyst_review_cost - seller_friction_cost
```

The practical interpretation was:

- Higher **Average Precision** in the top score bands correlated with more prevented abuse value per analyst hour.
- Better **Recall at Fixed Precision** correlated with higher abuse capture while keeping seller appeals and false holds inside policy limits.
- Better **Calibration Error** correlated with more stable investigation hit rates inside each risk band.

## How to handle the failures

The MLOps controls that usually matter most are:

- Use **time-based and seller-aware splits** so repeated entities and campaigns do not leak across evaluation sets.
- Score only on **mature labels** and keep a separate dashboard for pending investigations.
- Simulate the **analyst queue** before launch so a recall gain does not overwhelm investigations.
- Track metrics by **marketplace, seller age, fulfillment model, payment instrument, and device cohort**.
- Recalibrate or retune thresholds when prevalence shifts, instead of assuming one threshold works forever.
- Canary new models on a narrow traffic slice and compare queue volume, precision, and seller appeals against the current system.

## Production success criteria

This system is much more likely to ship safely when:

- High-risk score bands map to stable investigation hit rates.
- Investigation volume stays inside staffing limits.
- False-positive seller actions remain below a defined friction threshold.
- Recall on high-risk cohorts is stable across marketplaces, not just in the aggregate.
- The team has a rollback plan if queue quality, seller appeals, or calibration degrades after launch.
