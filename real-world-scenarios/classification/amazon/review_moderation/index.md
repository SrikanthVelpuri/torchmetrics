# Scenario: Review, Q&A, and Seller-Generated Content Moderation

Moderation in retail and marketplace surfaces is usually a **multilabel classification** problem. One review or seller
message can be spam, abusive, policy-violating, incentivized, or safety-relevant at the same time.

```{mermaid}
flowchart LR
    A[Review, question, or seller message] --> B[Per-label risk scores]
    B --> C[Per-label thresholds]
    B --> D[Ranked reviewer queue]
    C --> E[Allow, suppress, or action]
    D --> F[Human moderation]
```

## Why this scenario is hard at scale

- Rare severe labels matter far more than common nuisance labels.
- Policies evolve quickly, and appeal outcomes can change the effective label distribution.
- Language, locale, and product category all change the base rate and the meaning of certain patterns.
- Reviewer throughput matters almost as much as raw model quality.

## How metrics fail at scale

Moderation systems usually fail when teams:

- Report micro metrics that are dominated by spam and hide rare severe harms.
- Use one threshold for every label, even though severity and review cost differ.
- Measure final labels only and ignore the ranked-review workflow.
- Fail to segment by language or region, so a strong global score hides a weak localized model.
- Keep training on yesterday's policy while evaluating on today's policy taxonomy.

## Why traditional metrics failed

The traditional metrics failed because they optimized the wrong moderation behavior:

- **Micro metrics** passed because common nuisance labels such as spam dominated evaluation.
- **Shared thresholds** failed because severe policy labels need different precision-recall tradeoffs than low-severity nuisance labels.
- **Final-label-only evaluation** failed because reviewers work from ranked queues, not only final binary actions.
- **Global reporting** failed because localized language and marketplace gaps were hidden inside aggregate numbers.

## Worked Example

Suppose a moderation system sees `100,000` items:

- spam-like nuisance label positives = `10,000`
- severe dangerous-content label positives = `50`

Now imagine the model improves nuisance detection strongly but keeps weak severe-label recall:

```text
spam_recall = 9,500 / 10,000 = 95%
severe_recall = 10 / 50 = 20%
micro_recall = (9,500 + 10) / (10,000 + 50) = 94.6%
macro_recall = (95% + 20%) / 2 = 57.5%
```

The micro metric looks excellent. The severe-policy outcome is still poor. That is exactly how moderation systems can
look better offline while policy safety worsens in production.

## Questions I Asked Before Solving

Before I chose metrics or thresholds, I asked:

- Which labels are severe enough to require their own precision floor before auto-action?
- Are reviewers consuming ranked items, hard labels, or both?
- How much reviewer capacity exists per day for each moderation tier?
- Which locales, languages, or marketplaces have different policy behavior or label prevalence?
- How much do appeals and policy changes alter the effective label distribution over time?

## Why Those Questions Are Backed by Math, Probability, or Research

Those questions came from the real moderation math:

```text
micro_metric weights common labels more heavily
expected_cost(label) = FP_label * false_action_cost + FN_label * policy_escape_cost
review_queue = predicted_positive_labels + abstained_items
observed_severe_rate(score_bin) = severe_positives_in_bin / examples_in_bin
SE(label_recall) ~= sqrt(recall_hat * (1 - recall_hat) / label_positives)
```

- I asked about severity because the false-negative cost of rare severe labels is much larger than the nuisance-label cost.
- I asked about ranked review because reviewer throughput depends on ranking quality, not only final thresholded tags.
- I asked about capacity because queue growth determines whether the workflow is sustainable.
- I asked about locale differences because conditional label distributions change across language and marketplace slices.
- I asked about appeals because they change both label validity and post-deployment calibration.

The research anchor is the shared [Deep Dive](../deep_dive/index): rare-label safety problems are governed by
cost-sensitive and imbalanced-learning logic, not by one global micro summary.

## How the Questions Changed the Final Metric and Threshold Choice

Because of those answers, I did not accept micro metrics as the primary moderation gate.

- Because rare severe labels had to remain visible, I chose [Multilabel Average Precision](../../../../docs/source/classification/average_precision.rst) with `average="macro"`.
- Because launch still happens at explicit thresholds, I kept [Multilabel F1 Score](../../../../docs/source/classification/f1_score.rst) and [Multilabel Hamming Distance](../../../../docs/source/classification/hamming_distance.rst) at the operating point.
- Because reviewers consume prioritized queues, I chose [Multilabel Ranking Average Precision](../../../../docs/source/classification/label_ranking_average_precision.rst) and [Multilabel Coverage Error](../../../../docs/source/classification/coverage_error.rst).
- Because high-severity labels required explicit precision protection, I chose [Multilabel Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst).

Because of those answers, we chose per-label severity-aware thresholds, ranking-aware evaluation, and macro-first
policy monitoring rather than one flattened moderation score.

## Metric stack that usually works

- [Multilabel Average Precision](../../../../docs/source/classification/average_precision.rst) with `average="macro"` for rare-label protection.
- [Multilabel F1 Score](../../../../docs/source/classification/f1_score.rst) at the deployed thresholds.
- [Multilabel Ranking Average Precision](../../../../docs/source/classification/label_ranking_average_precision.rst) for reviewer prioritization quality.
- [Multilabel Coverage Error](../../../../docs/source/classification/coverage_error.rst) to estimate ranking depth needed by reviewers.
- [Multilabel Hamming Distance](../../../../docs/source/classification/hamming_distance.rst) for average label-wise noise.
- [Multilabel Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst) when severe labels must meet a precision floor.

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

moderation_metrics = MetricCollection(
    {
        "macro_ap": MultilabelAveragePrecision(num_labels=10, average="macro"),
        "macro_f1_at_0_45": MultilabelF1Score(num_labels=10, average="macro", threshold=0.45),
        "macro_hamming_at_0_45": MultilabelHammingDistance(num_labels=10, average="macro", threshold=0.45),
        "ranking_ap": MultilabelRankingAveragePrecision(num_labels=10),
        "coverage_error": MultilabelCoverageError(num_labels=10),
    }
)

summary = moderation_metrics(preds, target)
recall_at_99p, thresholds_at_99p = MultilabelRecallAtFixedPrecision(num_labels=10, min_precision=0.99)(
    preds, target
)
```

## Why the updated production metrics passed

The updated production metrics passed because they matched the real moderation workflow:

- [Multilabel Average Precision](../../../../docs/source/classification/average_precision.rst) with `average="macro"` passed because it protected rare severe labels.
- [Multilabel F1 Score](../../../../docs/source/classification/f1_score.rst) passed because it evaluated the actual enforcement thresholds.
- [Multilabel Ranking Average Precision](../../../../docs/source/classification/label_ranking_average_precision.rst) and [Multilabel Coverage Error](../../../../docs/source/classification/coverage_error.rst) passed because reviewer productivity depends on ranking quality.
- [Multilabel Recall at Fixed Precision](../../../../docs/source/classification/recall_at_fixed_precision.rst) passed because high-severity labels needed explicit precision floors before auto-action.

## How we established the relationship between updated production metrics and business metrics

We established the relationship by replaying reviewer queues and enforcement policies, then mapping model outcomes to
moderation operations metrics:

```text
policy_escape_rate = missed_severe_labels / actual_severe_labels
reviewer_backlog = surfaced_items - reviewed_items
appeal_overturn_rate = reversed_actions / total_actions
time_to_action = average_time_from_flag_to_decision
net_safety_value = policy_harm_prevented - review_cost - false_action_cost
```

That created a usable interpretation:

- Better **Macro Average Precision** correlated with fewer missed severe-policy escapes.
- Better **Ranking Average Precision** and lower **Coverage Error** correlated with faster reviewer throughput.
- Better **Recall at Fixed Precision** correlated with safer auto-action on high-severity labels.
- Better thresholded **Macro F1** correlated with fewer noisy enforcement actions.

## How to handle the failures

The usual production defenses are:

- Tune thresholds by **label severity, language, and marketplace**, not just globally.
- Version moderation policy definitions and evaluate each model against the policy it will enforce live.
- Track **appeal overturn rate, reviewer backlog, and time-to-action** together with model metrics.
- Monitor rare severe labels separately so they cannot disappear inside macro or micro summaries.
- Use fresh reviewer feedback and appealed cases to keep the model aligned with policy changes.
- Shadow-launch ranking changes before auto-action changes, because ranking quality and enforcement quality are different risks.

## Production success criteria

Moderation usually succeeds in production when:

- Severe-label recall is protected by explicit thresholds and precision floors.
- Reviewer load stays inside capacity targets.
- Appeal and overturn rates do not spike after rollout.
- Localization gaps are visible and managed per marketplace or language.
- Policy changes trigger reevaluation before a model is trusted at the old threshold.
