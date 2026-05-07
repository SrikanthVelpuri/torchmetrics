# Scenario: Baggage Exception Prediction and Bag-Recovery Operations

Baggage-exception prediction is usually a **binary classification** problem: predict whether a bag journey is likely
to fail because of transfer complexity, late bag drop, connection compression, station congestion, or tag/read issues.

```{mermaid}
flowchart LR
    A[Bag journey and scan history] --> B[Exception risk score]
    B -->|Below watch threshold| C[Normal handling]
    B -->|Within recovery band| D[Bag-room review]
    B -->|Above intervention threshold| E[Proactive bag recovery]
    C --> F[TN or FN]
    D --> G[TP or FP with station workload]
    E --> H[TP or FP with intervention cost]
```

## Why this scenario matters at airline scale

- True mishandling events are rare relative to successfully delivered bags.
- Labels may be delayed until final delivery, tracing closure, or customer claim resolution.
- Station layout, transfer path complexity, aircraft type, and international recheck rules all change the failure rate.
- A false positive consumes bag-room labor, while a false negative creates customer inconvenience and compensation cost.

## How metrics fail at scale

Baggage models usually fail when teams:

- Report accuracy or AUROC and ignore whether the top-risk bag queue is actually useful.
- Evaluate before tracing and claim labels mature.
- Use one threshold across hubs and outstations even though bag-room capacity differs widely.
- Hide station-level failures inside a strong system-wide average.
- Trust raw probabilities without calibration while supervisors act on score bands.

## Why traditional metrics failed

The traditional metrics failed because they were not tied to the bag-recovery workflow:

- **Accuracy** passed because the vast majority of bags are delivered normally.
- **AUROC alone** passed because it rewarded ranking quality even when the intervention queue was noisy.
- **Static thresholds** failed because station workload and transfer complexity vary by airport.
- **Global averages** failed because one transfer-heavy hub could degrade without moving the enterprise headline enough.

## Metric stack that usually works

- [Binary Average Precision](../../../../docs/source/classification/average_precision.rst) for top-queue quality.
- [Binary Precision at Fixed Recall](../../../../docs/source/classification/precision_at_fixed_recall.rst) when the baggage team must catch a minimum share of likely failures.
- [Binary Calibration Error](../../../../docs/source/classification/calibration_error.rst) because supervisors act on risk bands.
- [Binary Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) at the operational threshold.
- [Binary Stat Scores](../../../../docs/source/classification/stat_scores.rst) for explicit TP, FP, TN, and FN accounting.

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryPrecisionAtFixedRecall,
    BinaryStatScores,
)

baggage_metrics = MetricCollection(
    {
        "average_precision": BinaryAveragePrecision(),
        "ece": BinaryCalibrationError(n_bins=20, norm="l1"),
        "stat_scores_at_0_81": BinaryStatScores(threshold=0.81),
        "confusion_matrix_at_0_81": BinaryConfusionMatrix(threshold=0.81),
    }
)

summary = baggage_metrics(preds, target)
precision_75r, threshold_75r = BinaryPrecisionAtFixedRecall(min_recall=0.75)(preds, target)
```

## Why the updated production metrics passed

The updated production metrics passed because they protected the recovery queue rather than only the global ranking:

- [Binary Average Precision](../../../../docs/source/classification/average_precision.rst) passed because it measured whether the top flagged bags were truly worth intervention.
- [Binary Precision at Fixed Recall](../../../../docs/source/classification/precision_at_fixed_recall.rst) passed because the team needed a minimum recovery rate without flooding the bag room.
- [Binary Calibration Error](../../../../docs/source/classification/calibration_error.rst) passed because supervisors use risk bands for staffing and escalation.
- [Binary Stat Scores](../../../../docs/source/classification/stat_scores.rst) passed because baggage operations reason explicitly about recovered versus unnecessary actions.

## How we established the relationship between updated production metrics and business metrics

We established the relationship by replaying bag journeys through the proposed intervention policy and measuring the
downstream baggage outcomes:

```text
bag_recovery_queue = TP + FP
bags_saved_before_claim = TP
unnecessary_searches = FP
mishandled_bag_cost = FN * average_recovery_and_compensation_cost
net_baggage_value = prevented_claim_cost - unnecessary_search_cost - station_queue_cost
```

The observed correlation was:

- Better **Average Precision** correlated with more bags recovered per proactive search.
- Better **Precision at Fixed Recall** correlated with stable bag-room workload at the desired recovery coverage.
- Better **Calibration Error** correlated with more trustworthy supervisor risk bands for staffing and escalation.

## How to handle the failures

The strongest MLOps defenses are:

- Score only on **mature tracing and delivery labels**.
- Track metrics by **station, transfer type, aircraft family, connection time, and international recheck pattern**.
- Tune thresholds by station or severity tier when workloads differ materially.
- Validate distributed event joins so duplicate or missing bag scans do not corrupt the labels.
- Re-run threshold simulations before peak travel periods.
- Canary at a small number of hubs before rolling out system-wide.

## Production success criteria

This deployment is much more likely to succeed when:

- High-risk bags are surfaced early enough to improve recovery.
- Station workload remains within operational limits.
- Mishandled-bag rate and tracing load improve after launch.
- Station-specific failures do not hide inside global averages.
- Supervisors trust the calibration of the risk bands they use.
