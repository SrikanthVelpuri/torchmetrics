# Scenario: Maintenance Write-Up and Safety Event Tagging

Maintenance-write-up and safety-event analysis is often a **multilabel classification** problem. A single write-up,
logbook entry, or event narrative can contain multiple system, severity, and operational tags at the same time.

```{mermaid}
flowchart TD
    A[Write-up or event narrative] --> B[Per-label scores]
    B --> C[Maintenance and safety tags]
    B --> D[Ranked engineer review]
    C --> E[Auto-triage, route, or escalate]
    D --> F[Human review]
```

## Why this scenario matters at airline scale

- The label space is long-tailed, with many rare but critical tags.
- Taxonomy and review guidance can change as engineering teams revise procedures.
- Partial correctness is still useful because even one correct tag can improve routing to the right specialist.
- Rare critical labels must be protected, even if common benign labels dominate the volume.

## How metrics fail at scale

Maintenance-tagging systems usually fail when teams:

- Report only micro metrics and hide poor performance on rare critical tags.
- Use exact match as the main gate even though partial label correctness still adds operational value.
- Keep one threshold for every label even though escalation tags and informational tags have very different cost tradeoffs.
- Ignore reviewer workflow and evaluate only final binary labels.
- Compare scores across changing taxonomies without versioning the label space.

## Why traditional metrics failed

The traditional metrics failed because they either hid the long tail or over-penalized useful partial success:

- **Micro averages** passed while rare safety-critical tags still failed.
- **Exact Match** failed as the primary metric because it treated a partially correct triage as a total failure.
- **One threshold per label set** failed because critical escalation tags require a different operating point than routine informational tags.
- **Global reporting** failed because taxonomy and fleet-specific differences were hidden.

## Metric stack that usually works

- [Multilabel Average Precision](../../../../docs/source/classification/average_precision.rst) with `average="macro"` for rare-label protection.
- [Multilabel F1 Score](../../../../docs/source/classification/f1_score.rst) at deployed thresholds.
- [Multilabel Hamming Distance](../../../../docs/source/classification/hamming_distance.rst) for average label-wise noise.
- [Multilabel Ranking Average Precision](../../../../docs/source/classification/label_ranking_average_precision.rst) for ranked engineer review quality.
- [Multilabel Coverage Error](../../../../docs/source/classification/coverage_error.rst) for review-depth estimation.
- [Multilabel Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst) when critical tags require strong precision floors.

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelCoverageError,
    MultilabelF1Score,
    MultilabelHammingDistance,
    MultilabelRankingAveragePrecision,
    MultilabelRecallAtFixedPrecision,
)

maintenance_metrics = MetricCollection(
    {
        "macro_ap": MultilabelAveragePrecision(num_labels=12, average="macro"),
        "macro_f1_at_0_40": MultilabelF1Score(num_labels=12, average="macro", threshold=0.40),
        "macro_hamming_at_0_40": MultilabelHammingDistance(num_labels=12, average="macro", threshold=0.40),
        "ranking_ap": MultilabelRankingAveragePrecision(num_labels=12),
        "coverage_error": MultilabelCoverageError(num_labels=12),
    }
)

summary = maintenance_metrics(preds, target)
recall_at_99p, thresholds_at_99p = MultilabelRecallAtFixedPrecision(num_labels=12, min_precision=0.99)(
    preds, target
)
```

## Why the updated production metrics passed

The updated production metrics passed because they aligned with both rare-critical-label protection and engineer-review
workflow:

- [Multilabel Average Precision](../../../../docs/source/classification/average_precision.rst) with `average="macro"` passed because rare critical tags were no longer hidden.
- [Multilabel F1 Score](../../../../docs/source/classification/f1_score.rst) passed because it evaluated the actual deployed thresholds.
- [Multilabel Ranking Average Precision](../../../../docs/source/classification/label_ranking_average_precision.rst) and [Multilabel Coverage Error](../../../../docs/source/classification/coverage_error.rst) passed because engineers consume ranked candidate tags.
- [Multilabel Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst) passed because safety-relevant labels needed explicit precision floors.

## How we established the relationship between updated production metrics and business metrics

We established the relationship by replaying write-ups through the review workflow and measuring specialist-routing and
review outcomes:

```text
critical_tag_escape_rate = missed_critical_tags / actual_critical_tags
engineer_review_time = average_minutes_per_case
false_escalation_rate = unnecessary_critical_reviews / reviewed_cases
specialist_routing_gain = correctly_routed_cases / total_cases
net_review_value = review_time_saved + routing_gain - false_escalation_cost
```

The relationship was:

- Better **Macro Average Precision** correlated with fewer missed critical tags.
- Better **Ranking Average Precision** and lower **Coverage Error** correlated with faster engineer review.
- Better thresholded **Macro F1** correlated with cleaner triage decisions at the deployed operating point.
- Better **Recall at Fixed Precision** correlated with safer handling of critical labels.

## How to handle the failures

The strongest MLOps controls are:

- Version the **tag taxonomy** and keep historical backtests aligned to the active version.
- Track metrics by **fleet type, station, ATA-like subsystem grouping, and severity tier**.
- Set separate thresholds for critical versus informational labels.
- Keep reviewer-feedback loops for rare labels where examples are sparse.
- Validate that distributed text and metadata joins are complete before computing metrics.
- Re-evaluate thresholds after taxonomy or procedure changes.

## Production success criteria

This system is much closer to success when:

- Rare critical labels meet explicit recall targets.
- Engineer review becomes faster without flooding specialists.
- False escalations stay within an acceptable range.
- Taxonomy changes do not break comparability between model versions.
- The model improves routing and review quality after rollout, not only offline scores.
