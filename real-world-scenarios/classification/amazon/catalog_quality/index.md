# Scenario: Catalog Quality, Attribute Extraction, and Listing Policy Tags

Catalog-quality systems are often **multilabel classification** problems. A single ASIN or listing can be missing
multiple attributes, contain several policy issues, or require multiple downstream enrichment tags.

```{mermaid}
flowchart LR
    A[ASIN or listing update] --> B[Per-label scores]
    B --> C[Attribute and policy tags]
    B --> D[Reviewer ranking]
    C --> E[Auto-fix, suppress, or enrich]
    D --> F[Human catalog review]
```

## Why this scenario matters at Amazon scale

- The label space is long-tailed and changes by category, marketplace, and locale.
- Micro averages are dominated by common labels such as simple attribute completions, not by rare high-impact policy tags.
- Taxonomy changes can silently invalidate historical offline comparisons.
- Partial correctness is operationally useful even when the full label set is not perfect.

## How metrics fail at scale

Catalog systems usually fail when teams:

- Report only micro-averaged metrics and completely hide rare compliance-critical tags.
- Use [Multilabel Exact Match](../../../../docs/source/classification/exact_match.rst) as the launch metric even though it is too strict for most enrichment workflows.
- Keep one threshold for every label across every locale.
- Ignore taxonomy-version drift, so last quarter's metric is not comparable to this quarter's label space.
- Evaluate globally and miss that one category or language has a collapsing recall for high-severity tags.

## Why traditional metrics failed

The traditional metrics failed because they either hid the long tail or punished useful partial wins:

- **Micro averages** passed even when rare compliance-critical labels were collapsing.
- **Exact Match** failed as a launch metric because many catalog workflows benefit from partially correct label sets.
- **Single-threshold evaluation** failed because suppression tags, enrichment tags, and reviewer-assist tags have different cost curves.
- **Global reporting** failed because one locale or category could silently degrade inside a strong overall number.

## Worked Example

Suppose a catalog-quality model predicts two labels across `10,000` listings:

- a common attribute-completion label with `4,000` true positives
- a rare safety-policy label with `20` true positives

Now imagine the model achieves:

```text
common_label_recall = 3,800 / 4,000 = 95%
rare_label_recall = 4 / 20 = 20%
micro_recall = (3,800 + 4) / (4,000 + 20) = 94.6%
macro_recall = (95% + 20%) / 2 = 57.5%
```

A reader looking only at the micro metric might say the model is excellent. A catalog policy owner looking at the rare
safety label would say the model is unusable. That is exactly why long-tail multilabel problems need deeper questions
before metric choice.

## Questions I Asked Before Solving

Before I chose metrics or thresholds, I asked:

- Which labels are rare but operationally critical, and which are common but low-risk?
- Are humans consuming final binary tags or ranked candidate tags in a review workflow?
- What is the average number of true labels per listing, and how noisy are predicted label sets?
- Should thresholds be shared across all labels, or does severity require per-label or per-tier thresholds?
- Has the taxonomy changed recently, and do I have enough support in each label-marketplace slice to trust the estimate?

## Why Those Questions Are Backed by Math, Probability, or Research

These questions came from the structure of multilabel probability and decision-making:

```text
micro_metric weights labels by frequency
macro_metric weights labels equally
expected_cost(label) = FP_label * false_action_cost + FN_label * missed_escape_cost
review_queue = predicted_positive_labels + abstained_items
observed_label_rate(score_bin, label) = positives_in_bin / examples_in_bin
SE(label_recall) ~= sqrt(recall_hat * (1 - recall_hat) / label_positives)
```

- I asked about rare labels because macro-style metrics are the only way to keep the long tail visible.
- I asked about ranked review because reviewer-assist workflows are ranking problems, not just threshold problems.
- I asked about label cardinality because multilabel error behavior changes with the expected number of labels per listing.
- I asked about threshold sharing because the expected-cost tradeoff is different for suppression tags and enrichment tags.
- I asked about label support because a rare label with very few positives can produce unstable slice metrics.

The research anchor is the shared [Deep Dive](../deep_dive/index): when the label distribution is highly uneven, the
right question is not "what is the global average?" but "which probability mass matters operationally?"

## How the Questions Changed the Final Metric and Threshold Choice

Because of those answers, I did not rely on micro metrics or exact match as the main launch gate.

- Because rare labels had to stay visible, I chose [Multilabel Average Precision](../../../../docs/source/classification/average_precision.rst) with `average="macro"`.
- Because deployment still happens at a threshold, I kept [Multilabel F1 Score](../../../../docs/source/classification/f1_score.rst) and [Multilabel Hamming Distance](../../../../docs/source/classification/hamming_distance.rst) at the operating threshold.
- Because reviewers consume ranked tags, I chose [Multilabel Ranking Average Precision](../../../../docs/source/classification/label_ranking_average_precision.rst) and [Multilabel Coverage Error](../../../../docs/source/classification/coverage_error.rst).
- Because full label-set correctness is still informative but too strict for launch, I left [Multilabel Exact Match](../../../../docs/source/classification/exact_match.rst) as a diagnostic only.

Because of those answers, we chose macro-first evaluation, per-label or per-severity thresholds, and a reviewer-aware
metric stack rather than one flattened average.

## Metric stack that usually works

- [Multilabel Average Precision](../../../../docs/source/classification/average_precision.rst) with `average="macro"` for long-tail label protection.
- [Multilabel F1 Score](../../../../docs/source/classification/f1_score.rst) at the deployed thresholds.
- [Multilabel Hamming Distance](../../../../docs/source/classification/hamming_distance.rst) for average per-label noise.
- [Multilabel Ranking Average Precision](../../../../docs/source/classification/label_ranking_average_precision.rst) when reviewers consume ranked tags.
- [Multilabel Coverage Error](../../../../docs/source/classification/coverage_error.rst) to estimate how much of the ranked list a reviewer must inspect.
- [Multilabel Exact Match](../../../../docs/source/classification/exact_match.rst) as a strict diagnostic rather than the primary launch gate.

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelCoverageError,
    MultilabelExactMatch,
    MultilabelF1Score,
    MultilabelHammingDistance,
    MultilabelRankingAveragePrecision,
)

catalog_metrics = MetricCollection(
    {
        "macro_ap": MultilabelAveragePrecision(num_labels=14, average="macro"),
        "macro_f1_at_0_35": MultilabelF1Score(num_labels=14, average="macro", threshold=0.35),
        "macro_hamming_at_0_35": MultilabelHammingDistance(num_labels=14, average="macro", threshold=0.35),
        "ranking_ap": MultilabelRankingAveragePrecision(num_labels=14),
        "coverage_error": MultilabelCoverageError(num_labels=14),
        "exact_match_at_0_35": MultilabelExactMatch(num_labels=14, threshold=0.35),
    }
)
```

## Why the updated production metrics passed

The updated metric stack passed because it measured both label quality and workflow usefulness:

- [Multilabel Average Precision](../../../../docs/source/classification/average_precision.rst) with `average="macro"` passed because it protected rare labels from being washed out by common attributes.
- [Multilabel F1 Score](../../../../docs/source/classification/f1_score.rst) passed because it evaluated the actual deployed thresholds instead of only ranking quality.
- [Multilabel Hamming Distance](../../../../docs/source/classification/hamming_distance.rst) passed because it captured average label-wise noise in large tag sets.
- [Multilabel Ranking Average Precision](../../../../docs/source/classification/label_ranking_average_precision.rst) and [Multilabel Coverage Error](../../../../docs/source/classification/coverage_error.rst) passed because reviewers consume ranked candidates, not just final binary tags.

## How we established the relationship between updated production metrics and business metrics

We established the relationship by running label-level backtests and reviewer studies, then mapping the outputs to
catalog operations outcomes:

```text
policy_escape_rate = missed_high_severity_tags / actual_high_severity_tags
false_suppression_rate = good_listings_flagged / reviewed_good_listings
reviewer_minutes_saved = baseline_review_time - assisted_review_time
attribute_completion_gain = corrected_missing_attributes / total_listings
net_catalog_value = quality_uplift + reviewer_time_saved - false_suppression_cost
```

The observed alignment was:

- Better **Macro Average Precision** correlated with fewer missed long-tail policy tags.
- Better **Ranking Average Precision** and lower **Coverage Error** correlated with faster reviewer decisions.
- Better thresholded **Macro F1** correlated with cleaner auto-fix and suppression decisions at launch thresholds.

## How to handle the failures

The controls that usually make the difference are:

- Report metrics by **label, category, marketplace, and taxonomy version**, not just globally.
- Tune thresholds **per label or per severity tier**, especially when reviewer cost and escape cost differ.
- Version the ontology so historical backtests and production dashboards stay comparable after label changes.
- Monitor **label cardinality drift** and average labels per listing, because a flattening label distribution is often an early warning.
- Keep a reviewer-feedback loop for rare labels where fresh examples are scarce and label guidelines change.
- Validate that distributed data pipelines deduplicate listing events correctly before computing launch metrics.

## Production success criteria

Catalog systems usually succeed when:

- Rare, high-impact policy tags meet explicit recall targets.
- Reviewer workload remains stable because thresholds are tuned by label severity.
- New marketplaces and locales do not disappear inside global averages.
- Taxonomy migrations do not reset the team's ability to compare model versions.
- The system improves either auto-fix quality, reviewer speed, or suppression precision in a measurable way.
