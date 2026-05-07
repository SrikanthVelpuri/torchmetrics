# Real-World Scenarios for Classification Metrics

TorchMetrics exposes a rich set of binary, multiclass, and multilabel metrics under
[`torchmetrics.classification`](../../docs/source/classification/accuracy.rst). This guide explains how to use those metrics in
real deployment settings, how to set strong baselines, how to connect model metrics to business outcomes, and how to
tell the difference between a model that looks good offline and a model that is actually ready for production.

```{toctree}
:maxdepth: 1

foundations/index
amazon/index
american_airlines/index
fraud_detection/index
medical_screening/index
support_routing/index
content_moderation/index
production_failures/index
```

## What this guide is for

Most classification projects fail for one of three reasons:

- They optimize the wrong metric.
- They optimize the right metric, but at the wrong threshold.
- They improve the model score without improving the operational or business outcome.

This guide is organized around the scenarios where those mistakes show up most often.

```{mermaid}
flowchart TD
    A[Define the business decision] --> B[Frame the task: binary, multiclass, or multilabel]
    B --> C[Build baseline stack]
    C --> D[Measure aggregate, slice, and threshold behavior]
    D --> E{Do offline metrics move business outcomes?}
    E -- No --> F[Refine labels, thresholding, or task framing]
    F --> C
    E -- Yes --> G[Shadow or canary deployment]
    G --> H[Monitor drift, calibration, fairness, and queue pressure]
    H --> I{Stable after launch?}
    I -- No --> J[Recalibrate, retune, or roll back]
    I -- Yes --> K[Scale rollout]
```

## Production-ready versus production-failing evaluation

| Dimension | Systems that usually reach production | Systems that usually fail in production |
| --- | --- | --- |
| Task framing | Match the real decision surface | Collapse a ranking, routing, or multilabel workflow into the wrong task type |
| Baselines | Compare against majority, current rules, human, and last-production baselines | Compare only against another research model |
| Metric mix | Use threshold-free, thresholded, and slice-level metrics together | Use one headline metric only |
| Thresholding | Pick thresholds from capacity, safety, or revenue constraints | Use the default `0.5` threshold without business validation |
| Calibration | Check whether scores mean what operators think they mean | Treat probabilities as trustworthy without calibration checks |
| Monitoring | Track drift, subgroup behavior, and queue health after launch | Stop evaluation after offline validation |
| Business alignment | Convert confusion matrix outcomes into cost, risk, or throughput estimates | Assume that better AUROC or F1 automatically means better business value |

## Recommended reading order

- Start with [Foundations](foundations/index) to set baselines and choose metrics.
- Read [Amazon-Scale Scenarios](amazon/index) for MLOps patterns where scale, drift, and queue pressure break otherwise good models.
- Read [American Airlines Scenarios](american_airlines/index) for airline operations workflows where delays, station constraints, and irregular operations change which classification metrics matter.
- Read the scenario closest to your deployment pattern.
- Finish with [Production Failures and Launch Criteria](production_failures/index) to compare systems that ship well against systems that break after launch.
