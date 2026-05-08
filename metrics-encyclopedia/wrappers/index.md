---
title: Wrappers — deep dive
---

# Wrappers — deep dive

> A wrapper takes an existing metric and changes how it's *used* without re-implementing the math. Bootstrap CIs, per-class slicing, running windows, peak tracking — all expressed as wrappers.

[← Home](../index.md) · [Interview deep-dive →](./interview-deep-dive.md)

---

### BootStrapper (`tm.wrappers.BootStrapper`)

**What it does.** Wraps a base metric and computes it `num_bootstraps` times on resampled batches. Returns mean, std, raw values.

**When to use.** When you need a *confidence interval* on the metric — comparing two models, pre-deployment risk, papers.

**Real-world scenario.** "Is model A's F1 0.84 reliably better than B's 0.82?" Bootstrap both with `num_bootstraps=1000`, compute the mean and std, then the 95% CI. If CIs don't overlap, you have signal.

**Code.**
```python
from torchmetrics.wrappers import BootStrapper
from torchmetrics.classification import F1Score
boot = BootStrapper(F1Score(task="binary"), num_bootstraps=1000)
boot.update(preds, target)
result = boot.compute()  # dict with "mean", "std", "raw"
```

**Pitfalls.**
- Computational cost is `num_bootstraps × base metric cost`. For expensive metrics (FID), reduce `num_bootstraps` to 30-100.
- The bootstrap is **on the same data** — captures sampling variance, not training-run variance. For the latter, run multiple models with different seeds.

---

### ClasswiseWrapper (`tm.wrappers.ClasswiseWrapper`)

**What it does.** Takes a per-class metric (`average="none"`) and gives each class its own logger label (`accuracy/class_0`, `accuracy/class_1`, …).

**When to use.** When per-class numbers must end up as separate dashboard rows (often for monitoring rare-class regression).

**Code.**
```python
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import Accuracy
labels = ["cat", "dog", "fish"]
m = ClasswiseWrapper(Accuracy(task="multiclass", num_classes=3, average="none"),
                     labels=labels, prefix="acc/")
self.log_dict(m(preds, target))  # logs acc/cat, acc/dog, acc/fish
```

---

### MetricTracker (`tm.wrappers.MetricTracker`)

**What it does.** Tracks a metric across epochs. Exposes `.best_metric()` to get the peak value and the epoch where it occurred.

**When to use.** Decoupling "current epoch metric" from "best-so-far" without manually tracking. Integrates cleanly with checkpointing.

**Real-world scenario.** Multi-metric dashboard: track val/loss (minimise), val/F1 (maximise), val/AUROC (maximise), each with its own `MetricTracker`. At end of training, call `.best_metric()` on each — no boilerplate over what.

**Code.**
```python
from torchmetrics.wrappers import MetricTracker
tracker = MetricTracker(F1Score(task="binary"), maximize=True)
for epoch in range(num_epochs):
    tracker.increment()
    for batch in val_loader:
        tracker.update(*batch)
best, best_epoch = tracker.best_metric(return_step=True)
```

**Pitfalls.**
- `maximize` must match the metric direction. Hamming distance + `maximize=True` = chasing the worst.

---

### MultioutputWrapper (`tm.wrappers.MultioutputWrapper`)

**What it does.** Applies a 1-output metric independently to each output of a multi-output model.

**Real-world scenario.** Multi-target regression where each target has its own MAE: `MultioutputWrapper(MeanAbsoluteError(), num_outputs=5)` reports 5 MAEs.

---

### MultitaskWrapper (`tm.wrappers.MultitaskWrapper`)

**What it does.** Combines metrics from multiple tasks (one regression head + one classification head) into a single object.

**Real-world scenario.** Multi-task model: `{"price": MeanAbsoluteError(), "category": F1Score(task="multiclass", num_classes=10)}` — wrapped, logged together, no boilerplate.

---

### MinMaxMetric (`tm.wrappers.MinMaxMetric`)

**What it does.** Wraps a base metric and returns its running `(min, max, current)`.

**Real-world scenario.** Track the bounds of validation loss across an epoch — useful for alarm thresholds in production monitoring.

---

### Running (`tm.wrappers.Running`)

**What it does.** Sliding-window version of a metric — reports the metric on the *last K updates* rather than all-time.

**Real-world scenario.** Online monitoring: rolling 1-hour AUROC for a deployed model. `Running(AUROC(...), window=3600)` reports always-fresh quality.

---

### FeatureShare (`tm.wrappers.FeatureShare`)

**What it does.** Allows multiple metrics that share an expensive feature extractor (e.g., FID + KID, both using Inception) to compute features *once*.

**Real-world scenario.** Generative-model evaluation pipeline running FID + KID + LPIPS — without FeatureShare, each metric runs Inception forward separately. With FeatureShare, one forward, three metrics. Saves significant evaluation time.

---

### Transformations (`tm.wrappers.Transformations`)

**What it does.** Apply a function to inputs before passing them to a metric.

**Real-world scenario.** Computing R² in log-space: `Transformations(R2Score(), pred_transform=torch.log1p, target_transform=torch.log1p)`.

---

### Abstract (`tm.wrappers.Abstract`)

**What it does.** Base class for custom wrappers. Inherit and override `update`/`compute`/`reset` for your own composed metric.

---

## Quick-reference

| Goal | Wrapper |
|---|---|
| Confidence intervals | BootStrapper |
| Per-class logging | ClasswiseWrapper |
| Best-epoch tracking | MetricTracker |
| Multi-output regression | MultioutputWrapper |
| Multi-task heads | MultitaskWrapper |
| Min/max bounds | MinMaxMetric |
| Sliding window | Running |
| Shared feature extractor | FeatureShare |
| Pre-transform inputs | Transformations |

---

[Continue → interview deep-dive](./interview-deep-dive.md)
