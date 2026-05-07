# Scenario: Fraud Detection and Review Queues

Fraud detection is a classic **binary classification** problem with severe class imbalance, costly false negatives, and
real operational capacity constraints. It is also one of the easiest places to overestimate model readiness if the
team relies only on aggregate metrics such as accuracy or AUROC.

```{mermaid}
flowchart LR
    A[Transaction] --> B[Fraud score]
    B -->|score below review threshold| C[Approve]
    B -->|score in review band| D[Manual review]
    B -->|score above block threshold| E[Block or hold]
    C --> F[TN or FN]
    D --> G[TP or FP with review cost]
    E --> H[TP or FP with friction cost]
```

## Why this scenario reaches production

Fraud systems reach production when the evaluation is tied directly to queue economics:

- The model is good at ranking rare positives above negatives.
- Thresholds are chosen from review capacity and customer-friction limits, not from convenience.
- The score is calibrated enough that risk bands mean something operationally.
- Slice behavior is stable across merchants, geographies, devices, and payment types.

## Why this scenario fails in production

Fraud systems often fail when teams:

- Report a strong **AUROC** and ignore poor **Average Precision** under extreme imbalance.
- Choose a single threshold on a stale validation distribution and never retune it.
- Ignore queue growth, so improved recall overwhelms manual review operations.
- Evaluate on random splits instead of time-based splits, hiding prevalence and behavior drift.

## Recommended baseline setup

Use a layered baseline:

- **Majority baseline**: predict every transaction as legitimate. This usually gives high accuracy and proves why accuracy is weak here.
- **Current rules baseline**: the existing fraud rules engine or previous production model.
- **Simple model baseline**: an interpretable logistic or tree baseline to verify that the new model is really adding signal.
- **Capacity-aware threshold baseline**: the best threshold that fits the current review budget.

Offline validation should be time-ordered. Fraud patterns drift quickly, and a random split can make a weak model look
much stronger than it will be after deployment.

## Metrics that usually matter most

The strongest metric mix for this scenario usually combines:

- [Binary Average Precision](../../../docs/source/classification/average_precision.rst) for rare-event ranking quality.
- [Binary AUROC](../../../docs/source/classification/auroc.rst) as a secondary ranking metric, not the launch metric on its own.
- [Binary Recall at Fixed Precision](../../../docs/source/classification/recall_at_fixed_precision.rst) when review quality or customer friction demands a precision floor.
- [Binary Calibration Error](../../../docs/source/classification/calibration_error.rst) when score bands drive review or blocking policies.
- [Binary Confusion Matrix](../../../docs/source/classification/confusion_matrix.rst) at the candidate operating threshold.
- [Binary Matthews CorrCoef](../../../docs/source/classification/matthews_corr_coef.rst) for a balanced single-number summary at a chosen threshold.

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryMatthewsCorrCoef,
    BinaryRecallAtFixedPrecision,
)

fraud_metrics = MetricCollection(
    {
        "auroc": BinaryAUROC(),
        "average_precision": BinaryAveragePrecision(),
        "ece": BinaryCalibrationError(n_bins=20, norm="l1"),
        "mcc_at_0_85": BinaryMatthewsCorrCoef(threshold=0.85),
        "confusion_matrix_at_0_85": BinaryConfusionMatrix(threshold=0.85),
    }
)

summary = fraud_metrics(preds, target)
recall_98p, operating_threshold = BinaryRecallAtFixedPrecision(min_precision=0.98)(preds, target)
```

## How to correlate the metrics to business outcomes

A useful offline translation layer looks like this:

```text
review_rate = (TP + FP) / total_volume
fraud_capture = TP / actual_fraud_cases
escaped_fraud_cost = FN * average_loss_per_missed_case
review_cost = (TP + FP) * average_review_cost
customer_friction_cost = blocked_legitimate_transactions * average_friction_cost
net_value = prevented_loss - review_cost - customer_friction_cost
```

This is why **Average Precision** and **Recall at Fixed Precision** usually matter more than raw accuracy:

- Accuracy is inflated by the dominant legitimate class.
- AUROC can look healthy even when precision in the top score bands is too weak to support review operations.
- Calibration matters because the business often acts on score bands, not just on class labels.

## Production-ready path

Fraud models are usually ready to ship when:

- Score ranking beats the current rules engine on recent time-based holdouts.
- The chosen threshold respects review capacity and still captures enough fraud value.
- Calibration is stable enough that score bands map to expected fraud rates.
- The model is monitored by merchant, country, payment method, and acquisition channel.

## Failure pattern and learning

The common failed pattern is a model that improves AUROC but does not improve the economics of intervention. The
learning is simple: in imbalanced fraud problems, launch readiness is decided at the operating threshold and in the
review queue, not by a single aggregate ranking score.
