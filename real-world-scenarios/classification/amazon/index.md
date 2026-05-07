# Amazon-Scale Classification Scenarios

This pack is written from the perspective of an **MLOps engineer operating Amazon-scale classification systems**.
The common pattern is not that the model has no signal. The common pattern is that the model looks strong offline,
then fails when labels arrive late, thresholds go stale, Prime or holiday traffic changes the prevalence, or queue
pressure makes a previously acceptable false-positive rate operationally expensive.

```{toctree}
:maxdepth: 1

deep_dive/index
seller_risk/index
returns_abuse/index
catalog_quality/index
support_routing/index
review_moderation/index
delivery_exception/index
mlops_launch_playbook/index
```

## What this pack covers

These scenarios focus on the classification workflows that repeatedly show up in large retail and marketplace systems:

| Scenario | Task type | What usually breaks at scale | Metrics that usually matter |
| --- | --- | --- | --- |
| Seller risk and policy abuse | Binary | Rare-event prevalence shifts, delayed investigations, marketplace drift | Average Precision, Recall at Fixed Precision, Calibration Error |
| Returns abuse and concessions | Binary | Label maturity windows, holiday spikes, customer-friction costs | Average Precision, Precision/Recall at constrained thresholds, Confusion Matrix |
| Catalog quality and listing tags | Multilabel | Long-tail labels disappear inside micro averages, taxonomy drift | Macro Average Precision, Macro F1, Hamming Distance, Ranking AP |
| Support routing | Multiclass | Queue churn, multilingual drift, expensive confusions | Top-k Accuracy, Macro F1, Confusion Matrix, Calibration Error |
| Review moderation | Multilabel | Rare severe labels are hidden, policy thresholds drift by locale | Macro Average Precision, Ranking AP, Coverage Error, Hamming Distance |
| Delivery exceptions | Multiclass | Dominant on-time class hides high-cost failure classes | Macro F1, Top-k Accuracy, Confusion Matrix, Cohen Kappa |

## The Amazon-scale realities behind these scenarios

- Labels are often **delayed** because fraud, abuse, return, and delivery outcomes mature days or weeks later.
- Behavior changes by **marketplace, carrier, category, seller cohort, locale, and season**.
- Thresholds are constrained by **human review capacity**, contact-center staffing, and SLA commitments.
- Offline metrics are computed on distributed systems, so **duplicate events, partial partitions, and stale feature joins** can corrupt the evaluation if they are not guarded.
- A model can improve a research metric while still hurting the business because the real decision is made in a queue, a score band, or a downstream workflow.

## How to read this pack

- Start with the [Deep Dive](deep_dive/index) if you want the full reasoning process behind the scenario pack.
- Start with the scenario closest to the production system you own.
- For each scenario, focus on the sections named **Why traditional metrics failed**, **Why the updated production metrics passed**, and **How we established the relationship between updated production metrics and business metrics**.
- Then use **How to handle the failures** and **Production success criteria** to turn the metric choices into an operating plan.
- Finish with the [MLOps Launch Playbook](mlops_launch_playbook/index) for the cross-cutting controls that keep classification systems healthy after launch.
