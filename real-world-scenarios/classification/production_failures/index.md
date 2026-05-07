# Production Failures and Launch Criteria

The biggest difference between a classification system that reaches production and one that fails is not usually the
model architecture. It is whether the evaluation setup mirrors the real operating conditions.

```{mermaid}
flowchart TD
    A[Offline win] --> B{What was optimized?}
    B -->|Wrong metric| C[No business lift]
    B -->|Wrong threshold| D[Queue overload or unsafe misses]
    B -->|Wrong split| E[Performance drops after launch]
    B -->|No monitoring| F[Silent degradation]
    C --> G[Production failure]
    D --> G
    E --> G
    F --> G
```

## Clear differentiation: systems that ship versus systems that fail

| Area | Systems that usually reach production | Systems that usually fail |
| --- | --- | --- |
| Validation split | Time-aware, entity-aware, and close to deployment conditions | Random benchmark split with leakage or stale distributions |
| Metric selection | Mix of ranking, thresholded, and diagnostic metrics | One aggregate metric only |
| Threshold policy | Chosen from queue, safety, or revenue constraints | Default threshold or threshold picked on convenience |
| Slice analysis | Visible by important groups, classes, or labels | Hidden inside global averages |
| Calibration | Measured when scores drive decisions or trust | Ignored even though probabilities are consumed directly |
| Monitoring | Drift, subgroup behavior, and business outcomes tracked after launch | No post-launch checks beyond uptime |
| Ownership | Operators can explain what each launch metric protects | Metrics exist, but no one owns actions when they move |

## Three recurring failure archetypes

### 1. Offline winner, live loser

The model beats the baseline offline, but live performance decays immediately.

Usual cause:

- Temporal leakage, stale features, delayed labels, or population drift.

Main learning:

- The split strategy is part of the metric design. A wrong split creates fake certainty.

### 2. High score, low value

The model improves AUROC, F1, or accuracy but does not improve the business.

Usual cause:

- The measured metric is too far away from the business decision. The real value is created at a threshold, in a queue, or on a subgroup that was not monitored.

Main learning:

- Tie metrics back to confusion-matrix economics or workflow outcomes before launch.

### 3. Useful model, broken deployment

The model itself is strong, but rollout creates overload, friction, or missed cases.

Usual cause:

- Thresholds were not chosen with staffing, SLA, clinical burden, or reviewer capacity in mind.

Main learning:

- Thresholds are product policy knobs, not just model defaults.

## Learnings from failed scenarios

Failed classification launches teach a consistent set of lessons:

- A better research metric is not enough. The deployment rule has to be better too.
- Aggregate metrics are screening tools; launch decisions are usually made on slices and thresholds.
- Calibration matters whenever humans interpret scores or workflows use score bands.
- Monitoring must continue after launch because prevalence, behavior, and label quality all drift.
- Rare classes and rare labels need explicit protection, or common classes will dominate the averages.

## A practical launch checklist

Before rollout, make sure the answer is "yes" to these questions:

- Does the model beat the current production system on the metrics that match the live decision?
- Is the threshold or routing policy tied to capacity, safety, or revenue limits?
- Are rare classes, labels, and sensitive groups visible in the evaluation?
- Are calibration and confidence good enough for how scores will be used?
- Is there a rollback, recalibration, or retuning plan if live behavior drifts?

If the answer is "no" to any of these, the system may still be a promising model, but it is not yet a production-ready
classification system.
