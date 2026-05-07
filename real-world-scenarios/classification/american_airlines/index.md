# American Airlines Classification Scenarios

This pack is written from the perspective of an **MLOps engineer operating American Airlines-scale classification
systems**. In airline operations, a model can look strong offline and still fail in production because weather
disruptions, hub congestion, label delay, crew constraints, and station-specific behavior all change the real decision
surface.

```{toctree}
:maxdepth: 1

flight_delay_risk/index
missed_connection/index
baggage_exception/index
disruption_recovery/index
customer_support_routing/index
maintenance_tagging/index
mlops_launch_playbook/index
```

## What this pack covers

These scenarios focus on airline workflows where classification metrics must survive irregular operations, network
effects, and hard staffing constraints:

| Scenario | Task type | What usually breaks at scale | Metrics that usually matter |
| --- | --- | --- | --- |
| Flight delay risk | Binary | Weather shifts, hub congestion, stale thresholds | Average Precision, Recall at Fixed Precision, Calibration Error |
| Missed connection prediction | Binary | Tight bank structures, skewed prevalence, station drift | Average Precision, Precision/Recall at constrained thresholds, Confusion Matrix |
| Baggage exception prediction | Binary | Label delay, transfer complexity, station-specific failure modes | Average Precision, Precision at Fixed Recall, Calibration Error |
| Disruption recovery action selection | Multiclass | Irregular ops, asymmetric action costs, noisy labels | Top-k Accuracy, Macro F1, Confusion Matrix, Calibration Error |
| Customer support routing | Multiclass | Channel drift, language mix, escalation asymmetry | Top-k Accuracy, Macro F1, Confusion Matrix, Cohen Kappa |
| Maintenance and safety tagging | Multilabel | Long-tail labels, taxonomy drift, rare critical tags | Macro Average Precision, Macro F1, Ranking AP, Coverage Error |

## The American Airlines realities behind these scenarios

- Labels are often **delayed** because baggage outcomes, rebooking success, and disruption resolution finalize after the event.
- Behavior changes by **hub, station, aircraft type, route family, weather regime, season, and daypart**.
- Thresholds are constrained by **gate teams, operations planners, contact-center staffing, baggage desks, and maintenance review capacity**.
- A model can improve an offline metric while still hurting the operation if it creates too many unnecessary interventions.
- Many live workflows consume **score bands** or **top-k recommendations**, so calibration and ranking quality matter as much as the final class label.

## How to read this pack

- Start with the scenario closest to the production system you own.
- For each scenario, focus on the sections named **Why traditional metrics failed**, **Why the updated production metrics passed**, and **How we established the relationship between updated production metrics and business metrics**.
- Then use **How to handle the failures** and **Production success criteria** to turn the metric choices into an operating plan.
- Finish with the [MLOps Launch Playbook](mlops_launch_playbook/index) for the cross-cutting controls that keep airline classification systems healthy after launch.
