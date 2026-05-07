# Scenario: Content Moderation and Policy Tagging

Content moderation is often a **multilabel classification** problem. A single item can simultaneously contain spam,
hate, self-harm, sexual content, or other policy-relevant signals. In this setting, one threshold for all labels is
rarely enough, and ranking quality matters almost as much as final binary decisions.

```{mermaid}
flowchart TD
    A[Post, image, or message] --> B[Per-label scores]
    B --> C[Per-label thresholds]
    B --> D[Ranked labels for reviewer]
    C --> E[Auto-allow]
    C --> F[Auto-action]
    C --> G[Human review]
    D --> H[Reviewer sees highest-risk labels first]
```

## What makes these systems production-ready

Moderation systems are usually ready for launch when:

- Rare but severe labels are measured separately from common nuisance labels.
- Ranking metrics are used for reviewer-assist workflows.
- Per-label thresholds are tuned by severity and reviewer capacity.
- Exact-match style metrics are used as diagnostics, not as the primary business decision metric.

## What makes them fail in production

These projects often fail when teams:

- Report only micro-averaged metrics, hiding weak performance on rare harmful labels.
- Use one shared threshold across every label.
- Ignore ranking metrics, even though reviewers consume ranked alerts.
- Optimize exact match as the primary goal, which is too strict for most operational tagging workflows.

## Baseline setup

Useful baselines normally include:

- **Keyword and rules baseline** from the current policy engine.
- **Per-label prevalence baseline** to expose how easy common labels are.
- **Simple trainable baseline** for sanity checking.
- **Reviewer-assist baseline** measuring whether ranking improves reviewer speed or consistency.

The label taxonomy must also be part of the baseline discussion. If labels overlap, change over time, or are applied
inconsistently, even a strong model metric can be misleading.

## Metrics that matter most

- [Multilabel Average Precision](../../../docs/source/classification/average_precision.rst) for per-label ranking quality.
- [Multilabel F1 Score](../../../docs/source/classification/f1_score.rst) at the deployed per-label thresholds.
- [Multilabel Hamming Distance](../../../docs/source/classification/hamming_distance.rst) for average per-label error rate.
- [Multilabel Ranking Average Precision](../../../docs/source/classification/label_ranking_average_precision.rst) for reviewer-assist ranking quality.
- [Multilabel Coverage Error](../../../docs/source/classification/coverage_error.rst) when reviewers inspect labels in ranked order.
- [Multilabel Exact Match](../../../docs/source/classification/exact_match.rst) as a strict diagnostic, not usually the launch metric.

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

moderation_metrics = MetricCollection(
    {
        "macro_ap": MultilabelAveragePrecision(num_labels=8, average="macro"),
        "macro_f1_at_0_40": MultilabelF1Score(num_labels=8, average="macro", threshold=0.40),
        "macro_hamming_at_0_40": MultilabelHammingDistance(num_labels=8, average="macro", threshold=0.40),
        "ranking_ap": MultilabelRankingAveragePrecision(num_labels=8),
        "coverage_error": MultilabelCoverageError(num_labels=8),
        "exact_match_at_0_40": MultilabelExactMatch(num_labels=8, threshold=0.40),
    }
)
```

## Correlating metrics with moderation outcomes

Moderation metrics normally connect to operations through:

```text
policy_escape_rate = harmful_items_missed / total_harmful_items
reviewer_load = predicted_positive_labels + abstained_items
appeal_rate = false_positives_that_trigger_user_dispute
time_to_action = how quickly high-risk labels are surfaced
```

This is why metric selection has to be label-aware:

- **Macro AP** protects rare but severe categories.
- **Ranking AP** and **Coverage Error** matter when humans see a ranked list of policy candidates.
- **Hamming Distance** shows average label-wise noise.
- **Exact Match** is helpful to understand full-label-set correctness, but it is often too strict for the main launch gate.

## Production-ready path

These systems usually launch successfully when:

- Thresholds are selected per label, often with severity weights.
- Rare harmful labels have explicit acceptance criteria.
- The reviewer workflow uses ranking outputs, not just hard labels.
- Label drift and taxonomy changes are monitored post-launch.

## Failure pattern and learning

The common failure mode is a model with impressive micro scores that still misses rare high-severity content or floods
reviewers with noisy alerts. The learning is that multilabel moderation must be evaluated label by label and workflow
by workflow, not as one flattened average.
