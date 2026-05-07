# Scenario: Missed Connection Prediction and Passenger Protection

Missed-connection prediction is usually a **binary classification** problem where the model estimates whether an
itinerary is likely to break because of inbound delay, gate distance, recheck burden, or tight bank structure.

```{mermaid}
flowchart TD
    A[Passenger itinerary and live flight state] --> B[Misconnect risk score]
    B -->|Below protect threshold| C[No action]
    B -->|Within review band| D[Manual monitoring]
    B -->|Above protection threshold| E[Proactive reaccommodation]
    C --> F[TN or FN]
    D --> G[TP or FP with support cost]
    E --> H[TP or FP with reaccommodation cost]
```

## Why this scenario matters at airline scale

- Misconnects are rare relative to completed connections, which makes naive metrics look stronger than they should.
- A missed connection is expensive because it creates rebooking load, hotel and meal costs, and customer dissatisfaction.
- Prevalence and transfer difficulty vary sharply by hub, international connection type, and time bank.
- Labels can arrive late because the final passenger outcome depends on downstream protection and same-day recovery.

## How metrics fail at scale

Misconnect models usually fail when teams:

- Use accuracy or headline F1 on balanced samples and ignore the live rarity of true misconnects.
- Tune thresholds without considering reaccommodation inventory and airport support capacity.
- Report one network-wide number and miss poor recall on tight banks, customs connections, or weather-affected hubs.
- Evaluate on stale schedules that do not reflect the current bank plan or hub topology.
- Ignore calibration even though protection teams use score bands to decide who to rebook proactively.

## Why traditional metrics failed

The traditional metrics failed because they did not match the passenger-protection workflow:

- **Accuracy** passed because most passengers still make their connection.
- **Balanced-sample F1** passed because it hid the real operating prevalence and over-promised queue quality.
- **One threshold** failed because the cost of proactive rebooking varies by load factor, fare mix, and remaining inventory.
- **Aggregate reporting** failed because misconnect risk is highly bank- and hub-specific.

## Metric stack that usually works

- [Binary Average Precision](../../../../docs/source/classification/average_precision.rst) for rare-event ranking quality.
- [Binary Precision at Fixed Recall](../../../../docs/source/classification/precision_at_fixed_recall.rst) when the operation must catch a minimum share of likely misconnects.
- [Binary Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst) when proactive rebooking capacity is limited.
- [Binary Calibration Error](../../../../docs/source/classification/calibration_error.rst) for score-band trust.
- [Binary Confusion Matrix](../../../../docs/source/classification/confusion_matrix.rst) at the production threshold.

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryPrecisionAtFixedRecall,
    BinaryRecallAtFixedPrecision,
)

misconnect_metrics = MetricCollection(
    {
        "average_precision": BinaryAveragePrecision(),
        "ece": BinaryCalibrationError(n_bins=20, norm="l1"),
        "confusion_matrix_at_0_84": BinaryConfusionMatrix(threshold=0.84),
    }
)

summary = misconnect_metrics(preds, target)
precision_80r, threshold_80r = BinaryPrecisionAtFixedRecall(min_recall=0.80)(preds, target)
recall_97p, threshold_97p = BinaryRecallAtFixedPrecision(min_precision=0.97)(preds, target)
```

## Why the updated production metrics passed

The updated production metrics passed because they matched the tradeoff between catching vulnerable itineraries and
avoiding unnecessary rebooking:

- [Binary Average Precision](../../../../docs/source/classification/average_precision.rst) passed because it measured the cleanliness of the proactive-protection queue.
- [Binary Precision at Fixed Recall](../../../../docs/source/classification/precision_at_fixed_recall.rst) passed because the team needed a minimum protection coverage level.
- [Binary Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst) passed because proactive actions are expensive when seats are scarce.
- [Binary Calibration Error](../../../../docs/source/classification/calibration_error.rst) passed because score bands were consumed directly by airport and customer teams.

## How we established the relationship between updated production metrics and business metrics

We established the relationship by replaying live itineraries and simulating the proactive-protection policy at
different thresholds:

```text
protection_queue = TP + FP
misconnects_avoided = TP
unnecessary_rebookings = FP
misconnect_cost = FN * average_reaccommodation_and_recovery_cost
net_passenger_value = avoided_recovery_cost - unnecessary_rebooking_cost - support_queue_cost
```

The operational linkage was:

- Better **Average Precision** correlated with more useful proactive rebookings per action taken.
- Better **Precision at Fixed Recall** correlated with keeping protection inventory focused on truly at-risk passengers.
- Better **Calibration Error** correlated with more reliable risk bands for gate agents and airport support teams.

## How to handle the failures

The most useful MLOps controls are:

- Use **time-aware and connection-aware splits** that respect schedule-bank structure.
- Monitor metrics by **hub, connection type, international versus domestic, customs exposure, and departure bank**.
- Recompute thresholds when schedule banks or protection policies change.
- Validate on mature passenger outcomes rather than only inbound-flight delay labels.
- Simulate queue size and seat-consumption effects before rollout.
- Run canaries on a small set of hubs before scaling to the full network.

## Production success criteria

This system is usually healthy when:

- High-risk passengers are protected early without exhausting reaccommodation inventory.
- Misconnect reduction is visible in the live operation.
- Airport support queues stay manageable.
- Hub- and bank-level slices remain stable.
- The score is calibrated well enough to drive proactive protection bands.
