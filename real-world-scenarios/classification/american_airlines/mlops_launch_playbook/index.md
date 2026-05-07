# American Airlines MLOps Failure Modes and Launch Playbook

The difference between a model that looks strong in offline airline data and a model that survives real operations is
usually the **MLOps system around the metric**, not the model architecture by itself. This playbook applies regardless
of network size, fleet complexity, or traffic volume — the same failure patterns and the same controls appear whether
you are launching at a single hub or rolling out across the full domestic and international network.

```{mermaid}
flowchart LR
    A[Feature and label contracts] --> B[Time-aware backtest]
    B --> C[Slice and threshold simulation]
    C --> D[Shadow deployment]
    D --> E[Station or hub canary]
    E --> F[Network rollout]
    F --> G[Continuous monitoring and recalibration]
    G -->|drift or queue failure| H[Rollback, retune, or retrain]
    H --> B
```

Every stage is a **gate, not a formality**. A model that cannot pass shadow deployment on real operational traffic has
not yet earned a canary. A canary that degrades a single critical hub or irregular-operations regime must roll back
even if network-wide metrics look healthy.

## Scale-readiness principles

These principles apply at any level of operational complexity. A team launching a model for one regional station and
a team rolling out a global disruption-recovery system face the same structural risks — the consequences scale with
the network.

- **Immutable data contracts.** Every operational feature, outcome label, and action taxonomy is versioned and
  validated before a job runs. Scan-event gaps, stale joins, or missing tail-number records discovered in
  production are a launch failure that happened at contract-definition time.
- **Operational completeness validation.** At airline scale, missing station scans or dropped flight-event records
  look like performance improvements because the hard irregular-operations cases disappear from the metric.
  Validate record counts, key completeness, and time coverage before any metric is computed.
- **Slice-first, aggregate-second.** Network-wide metrics are summaries. Hub, station, fleet, and weather-regime
  metrics are the real acceptance criteria. A launch bar defined only on aggregate numbers will miss the hub or
  disruption pattern that is silently failing.
- **Capacity-aware thresholds.** A threshold is an operational contract with the downstream workflow — the planner
  queue, the bag room, the support team. Setting a threshold without simulating station workload at peak irregular-
  operations volume creates a launch that succeeds on metrics and fails on operations.
- **Rollback before you need it.** Shadow mode, fallback thresholds, and rollback pipelines must be exercised
  before launch, not designed during a ground-stop event.

## The most common scale failures

| Failure mode | How the metric lies | What usually fixes it |
| --- | --- | --- |
| Irregular-operations drift | A threshold from normal operations fails during storms or recovery days | Keep event-specific backtests and retune thresholds on recent irregular-operations windows |
| Label delay | Baggage, disruption, and passenger outcomes are scored before they mature | Score on mature outcome windows and separate pending labels |
| Station or hub heterogeneity | Network-wide metrics look healthy while one hub quietly fails | Gate launches on hub, station, route-family, and fleet-type slices |
| Policy or taxonomy changes | Last quarter's score is not comparable to this quarter's label definition | Version labels, queues, and action taxonomies; backtest within the same version |
| Queue saturation | Better recall floods planners, bag rooms, or support agents | Simulate queue load at p50, p95, and p99 operational volume and use capacity-aware thresholds |
| Distributed data issues | Missing scans, duplicated events, or stale joins inflate metrics | Validate data contracts and deduplicate before metric computation |
| Training-serving skew | Offline features differ from live operational features | Compare online and offline feature distributions continuously |
| Threshold staleness | Score calibration decays as the schedule and disruption profile changes | Recalibrate thresholds on a fixed cadence and after schedule changes or major irregular-operations events |
| Cold-start for new routes or fleets | New routes or recently acquired fleets have insufficient history | Set minimum sample size gates and use conservative thresholds until data matures |
| Cascade failures across operational systems | A gate assignment model feeds a delay model — upstream drift propagates silently | Monitor inter-model contracts as strictly as data contracts; treat each hand-off as a system boundary |
| Feedback loop amplification | Model decisions affect future re-accommodation or bag-routing outcomes, contaminating retraining data | Track label provenance and flag feedback-contaminated windows before retraining |
| Peak-traffic beyond canary headroom | A major weather event hits the canary hub before full-scale behavior is validated | Define canary traffic bounds, headroom limits, and automatic traffic-shed rules before rollout |

## What production-successful teams do differently

Successful teams treat metrics as **airline operating controls, not model scorecards**. The behaviors that separate
them are consistent regardless of network size or fleet complexity:

- They separate **ranking metrics**, **threshold metrics**, and **workflow metrics** instead of relying on one
  headline score. These three families answer different operational questions and must not be collapsed.
- They map every confusion-matrix outcome to a real airline cost: delay minutes, reaccommodation cost, bag-room
  load, repeat contacts, or missed-connection rate. A metric that cannot be connected to an operational cost is
  a monitoring decoration, not a launch criterion.
- They define **slice-level launch bars** for hub, station, route family, fleet type, weather regime, and channel
  before the launch conversation begins.
- They run **irregular-operations simulations** — not normal-operations simulations — because models that pass on
  a clear-sky Tuesday routinely fail during a northeast corridor ground stop or a hub-capacity event.
- They monitor **planner queue health, calibration drift, and operational outcomes** after launch, not only
  technical uptime. An online model with a degraded threshold is an operational failure even if the service is
  healthy.
- They rehearse rollback, threshold fallback, and shadow-mode comparisons before the model is exposed to a major
  irregular-operations event. Runbooks should be boring because they have been practiced.
- They treat **recalibration as a scheduled operation**, not an emergency response. Threshold drift after schedule
  changes is expected; catching it before queue saturation is the goal.

## Why traditional metrics fail and updated production metrics pass

Traditional metrics fail at airline scale because they measure model quality too far from the actual operational
decision:

- **Accuracy** rewards dominant normal-operation classes. The hard cases — storms, ground stops, hub-capacity
  events — are the minority class and the highest-cost class.
- **One global AUROC or F1** can pass while the deployed threshold overloads a planner queue or misses a
  high-cost disruption case at a critical hub.
- **Aggregate reporting** can pass for a full schedule rotation while a specific hub, fleet, or weather regime
  degrades badly.
- **Static thresholds** survive clear-weather evaluation and collapse during irregular operations.

Updated production metrics pass when they are chosen to protect the live workflow:

- **Ranking metrics** pass when the business needs a clean prioritized intervention queue — the model must
  correctly order cases, not just classify them.
- **Threshold metrics** pass when the business needs a safe operational action rule — false positives and false
  negatives must map to real planner capacity and passenger impact.
- **Calibration metrics** pass when planners, stations, or support teams consume probability bands directly — a
  score of 0.8 must mean roughly the same thing on a normal day as during a recovery window.
- **Slice metrics** pass when success requires stability across hubs, stations, and disruption regimes — one
  degraded hub is a degraded operation.

## How we establish the relationship between updated production metrics and business metrics

At airline scale, the relationship is established with the same repeatable loop regardless of the use case:

1. Run time-aware backtests on mature operational outcomes, with data completeness validated before the run.
2. Sweep thresholds and simulate the live planner, station, or support workflow at representative traffic
   percentiles — including peak irregular-operations days.
3. Translate TP, FP, TN, and FN into operational cost, recovery value, and queue impact using real workflow
   unit economics.
4. Compare the candidate model against the current production system on both model metrics and business metrics.
5. Keep only the metric set that predicts live improvement consistently across recent windows and critical slices.

The generic accounting layer applies at any network scale:

```text
business_value
= TP * value_of_correct_intervention
- FP * cost_of_unnecessary_intervention
- FN * cost_of_missed_operational_action
- workflow_overload_cost
- recalibration_and_ops_cost
```

Adding `recalibration_and_ops_cost` is deliberate. A model that requires constant planner override or manual
threshold adjustment has a real operational cost that must appear in the business case.

## Scale-readiness gates

These gates apply before any stage transition in the rollout pipeline. They are enforced at single-station pilots
and at full network rollouts — the stakes change, the gates do not.

| Gate | What it validates | Who owns it |
| --- | --- | --- |
| Data completeness check | All expected scan events, flight records, and outcome windows arrived; deduplication applied | Data engineering |
| Label maturity check | Outcome windows are closed; pending labels are separated | ML engineering |
| Offline metric parity | Candidate beats current system across all slice families and operational windows, including recent irregular-operations periods | ML engineering |
| Threshold simulation | Planner queue and station workload are within capacity at p50, p95, and p99 operational volume | Operations and ML engineering |
| Shadow comparison | Live score distributions match offline distributions within tolerance across hubs and weather regimes | ML engineering |
| Slice acceptance bar | Every designated high-risk hub, station, and disruption cohort meets its explicit acceptance criterion | Product and ML engineering |
| Rollback drill | Rollback, fallback threshold, and shadow-mode reactivation all tested before an irregular-operations event | Engineering and operations |
| Monitoring coverage | All post-launch metrics are live and alerting before traffic is shifted | Engineering |

## A practical launch checklist

Before rollout, every answer must be "yes":

- Does the model beat the current production system on the metric family that matches the live operational workflow?
- Were the metrics computed on time-aware data with mature labels and no entity leakage?
- Were data completeness and deduplication validated before any metric was computed?
- Were backtests run on recent irregular-operations windows, not only on normal-operations data?
- Is the threshold tied to a real operational constraint such as planner capacity, station workload, or support
  staffing?
- Was the threshold simulated at peak irregular-operations volume, not just average daily volume?
- Are high-risk hubs, stations, and weather regimes visible and protected by explicit, pre-agreed acceptance
  criteria?
- Has the rollback path, fallback threshold, and shadow reactivation been exercised in a drill?
- Is every post-launch monitoring metric live and alerting before operational traffic is shifted?
- Is there an escalation path defined for threshold staleness and unexpected calibration drift after schedule
  changes?

## Incident response and rollback runbook

When a post-launch alert fires, the response sequence is the same regardless of network scale:

1. **Triage within SLA.** Is the alert a data pipeline issue, a model drift issue, or a threshold issue? Each has
   a different owner and a different mitigation path. Do not retrain before root cause is known.
2. **Shift traffic to shadow or fallback.** If the cause is unknown and operational impact is growing, route
   traffic to the previous model or the rule-based fallback while root cause is identified. During an active
   irregular-operations event, this transition must be possible within minutes.
3. **Preserve the incident window.** Snapshot the score distribution, the feature distribution, and the label
   distribution from the incident window for post-mortem use. Do not retrain on contaminated data.
4. **Recalibrate or retrain in isolation.** When root cause is known, apply the fix in a held-out validation
   environment and repeat the launch checklist — including irregular-operations backtests — before restoring
   traffic.
5. **Update the acceptance criteria.** Every incident that slipped through the checklist is evidence that a gate
   is missing or too weak. Add or tighten the relevant gate before the next launch cycle.

## Monitoring after launch

The most useful post-launch dashboard includes every layer from model score to operational outcome:

- A ranking metric such as AP or top-k accuracy.
- A threshold metric such as recall at fixed precision or macro F1.
- A reliability metric such as expected calibration error or reliability diagram slope.
- Slice metrics by hub, station, fleet, route family, language, and disruption regime — whichever cohorts carry
  the highest operational cost if they degrade.
- Workflow metrics such as planner queue depth, reroute rate, repeat contacts, bag-recovery load, and
  time-to-resolution.
- Business metrics such as delay minutes saved, misconnect reduction, mishandled-bag reduction, or same-day
  recovery rate.
- Data pipeline health metrics: scan-event completeness per station, flight-record arrival latency, label
  maturity rate, deduplication pass rate.

Each metric should have an owner, an alert threshold, and a defined response action. A metric without an owner
is a decoration.

## The production lesson

At airline scale, metrics fail when they are treated like abstract model scores. They work when they are treated
like operating controls that must survive storms, schedule changes, station constraints, and uneven traffic across
the network. The network size changes the blast radius of a failure, not the underlying failure mode. The same
controls that prevent a regional station pilot from degrading a hub's planner queue also prevent a full network
rollout from collapsing during a major weather event. Apply them early, enforce them as hard gates, and rehearse
the recovery paths before they are needed. That is the difference between a model that looks good in a notebook
and one that remains operationally useful in production.
