---
title: Aggregation metrics — deep dive
---

# Aggregation metrics — deep dive

> Aggregation metrics aren't really *metrics* — they're stateful reducers that compose well with TorchMetrics' update/compute/sync machinery. The main reason to use them: they sync correctly under DDP for free.

[← Home](../index.md) · Wrappers interview is in the [wrappers page](../wrappers/interview-deep-dive.md).

---

### SumMetric (`tm.aggregation.SumMetric`)

**What it does.** Maintains a running sum across `update()` calls. Syncs across ranks via `all_reduce(SUM)`.

**Real-world scenario.** Total tokens processed; cumulative dollars in a batch; "any sample seen?" counters.

---

### MeanMetric (`tm.aggregation.MeanMetric`)

**What it does.** Running mean — `(sum, count)` state, syncs both, divides at compute.

**Real-world scenario.** Custom non-standard scalar that you compute per-batch and want averaged across the epoch correctly under DDP.

**Code.**
```python
from torchmetrics.aggregation import MeanMetric
loss_avg = MeanMetric()
for batch in loader:
    loss_avg.update(custom_loss(batch))
print(loss_avg.compute())  # cross-rank correct mean
```

**Pitfalls.**
- For weighted mean, pass `weight=` explicitly — `update(value, weight)`.

---

### MinMetric / MaxMetric (`tm.aggregation.MinMetric`, `MaxMetric`)

**What they do.** Track the running min / max of a scalar across batches. Sync via `all_reduce(MIN/MAX)`.

**Real-world scenario.** Largest gradient norm across an epoch; quietest sample's loss; per-epoch range monitoring.

---

### CatMetric (`tm.aggregation.CatMetric`)

**What it does.** Concatenates all `update()` values across the epoch, syncs to one big tensor.

**Real-world scenario.** Collecting predictions for offline analysis under DDP without manually managing `all_gather`. Heavy memory cost — only when you need every value, not a summary.

---

### RunningSum / RunningMean (`tm.aggregation.RunningSum`, `RunningMean`)

**What they do.** Sum / mean over a *sliding window* of recent updates.

**Real-world scenario.** Smoothed training loss for a logger — last-100-step running mean filters out per-batch noise.

---

## Quick-reference

| Need | Use |
|---|---|
| Running sum (DDP-safe) | SumMetric |
| Running mean (DDP-safe) | MeanMetric |
| Running min/max | MinMetric / MaxMetric |
| Collect all values | CatMetric |
| Sliding-window smoothing | RunningSum / RunningMean |

---

When *not* to use these:
- For simple Python sums on a single rank, plain `+=` is fine. The wrappers are useful when you want the same code to scale to DDP.
- For named labels (class-wise) on top of an aggregator, compose with `ClasswiseWrapper` (in the [wrappers family](../wrappers/index.md)).

---

[← Back to home](../index.md)
