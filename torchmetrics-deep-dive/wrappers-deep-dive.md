---
title: Wrappers Deep Dive
nav_order: 24
---

# Wrappers — All of Them, in Depth

`torchmetrics.wrappers` are the most under-documented power tools in the library. Every senior interview asks about at least one. This page covers all 11.

---

## Why wrappers?

A wrapper takes any `Metric` and returns a richer `Metric`. The wrapped metric still does its job; the wrapper adds a *new dimension* — confidence intervals, history tracking, per-class breakdown, rolling windows, and so on.

```text
+------------+        +------------+
|   Metric   |  ────► | Wrapper(M) |  ────► same lifecycle
+------------+        +------------+

Wrappers compose. ClasswiseWrapper(BootStrapper(F1Score(...))) works.
```

---

## 1. `BootStrapper` — confidence intervals

```python
from torchmetrics.wrappers import BootStrapper
from torchmetrics import F1Score

f1_with_ci = BootStrapper(
    F1Score(task="binary"),
    num_bootstraps=1000,
    quantile=torch.tensor([0.025, 0.975]),
    sampling_strategy="poisson",   # or "multinomial"
    generator=torch.Generator().manual_seed(42),
)
```

**How it works.** Maintains `num_bootstraps` internal copies of the wrapped metric. Each batch is replicated into each copy, with each row included `~Poisson(1)` times. At `compute()`, returns mean + quantiles across the N replicas.

**When to use.**
- Reporting a metric with a confidence interval (paper, board review).
- Detecting whether a 0.5 % improvement is signal or noise.
- Ablation studies — different model variants compared with overlapping CIs.

**Gotchas.**
- Memory cost: N × (wrapped state size). For list-state metrics, use `num_bootstraps ≤ 100`.
- Without `quantile=`, returns mean only (just a smoother estimator).
- Pass `generator=` for reproducibility.

**Interview drill-down (3 levels):**

> **Q.** Why Poisson(1) sampling specifically?

> **A.** It's the asymptotic limit of with-replacement sampling. Each item appears in each bootstrap copy with the right multinomial expectation, no need to store sample IDs.

>> **F1.** What if you want a *paired* bootstrap for comparing two models?

>> **A.** `BootStrapper` only does unpaired. For paired (same sample indices in both replicas across both models), write a custom wrapper that maintains synchronized indices across the two metrics.

>>> **F1.1.** What's the gain of paired vs unpaired?

>>> **A.** Paired removes the variance from the sample-set baseline. CI on the *difference* is much tighter than CI(model_A) − CI(model_B). For small Δ, paired is the only way to detect real signal.

---

## 2. `MetricTracker` — history across epochs

```python
from torchmetrics.wrappers import MetricTracker
from torchmetrics import Accuracy

tracker = MetricTracker(Accuracy(task="multiclass", num_classes=10))

for epoch in range(10):
    tracker.increment()        # start a new epoch's slot
    for x, y in val_loader:
        tracker.update(model(x), y)
    print(f"epoch {epoch}: {tracker.compute()}")

best_value, best_epoch = tracker.best_metric(return_step=True)
```

**How it works.** Internally maintains a list of metric instances, one per call to `increment()`. `best_metric` uses the wrapped metric's `higher_is_better` flag to find the optimum.

**When to use.**
- Training scripts where you want "best val acc and the epoch it happened on" without writing your own bookkeeping.
- Multi-metric tracking via `MetricTracker(MetricCollection({...}))`.

**Gotchas.**
- The wrapped metric must have `higher_is_better` set, or `best_metric` raises.
- For collections, `best_metric` returns a dict.

---

## 3. `Running` — rolling-window metric

```python
from torchmetrics.wrappers import Running
from torchmetrics import MeanSquaredError

# Rolling 100-batch MSE
running_mse = Running(MeanSquaredError(), window=100)
```

**How it works.** Keeps a circular buffer of state snapshots. `compute()` reduces only the window's snapshots, not the entire history.

**When to use.**
- Production drift monitoring — alert when rolling-window metric crosses threshold.
- Streaming training — monitor smooth metric curves without full-epoch resets.
- Real-time dashboards — last N events' value, not all-time.

**Gotchas.**
- Window size = number of `update()` calls, not number of samples (unless your batch size is constant).
- State buffer's memory = window × wrapped state size.

---

## 4. `MinMaxMetric` — track best/worst alongside current

```python
from torchmetrics.wrappers import MinMaxMetric
from torchmetrics import MeanAbsoluteError

mae = MinMaxMetric(MeanAbsoluteError())
mae.update(preds, target)
out = mae.compute()
# {'raw': tensor(...), 'min': tensor(...), 'max': tensor(...)}
```

**How it works.** On each `compute`, captures the wrapped metric's value and updates internal min/max trackers. Returns a dict.

**When to use.**
- Worst-case-tracking (maximum WER ever seen on accented speech).
- Best-so-far without the full epoch history of `MetricTracker`.
- Lightweight monitoring — much cheaper than `MetricTracker`.

---

## 5. `MultioutputWrapper` — same metric, K parallel outputs

```python
from torchmetrics.wrappers import MultioutputWrapper
from torchmetrics import MeanAbsoluteError

# 68 facial landmarks, each is a 2D point
landmark_mae = MultioutputWrapper(
    MeanAbsoluteError(),
    num_outputs=68,
)
```

**How it works.** Internally clones the wrapped metric N times, dispatches each output column to its own copy.

**When to use.**
- Multi-output regression (each output has its own metric instance).
- Multi-target classification where each target has its own task.

**Gotchas.**
- Memory = N × wrapped state. Cheap for tensor-state metrics, bad for list-state.
- Returns a stacked tensor; index by output dim.

---

## 6. `MultitaskWrapper` — different metrics per task

```python
from torchmetrics.wrappers import MultitaskWrapper
from torchmetrics import F1Score, MeanAbsoluteError

mt = MultitaskWrapper({
    "fraud":  F1Score(task="binary"),
    "ltv":    MeanAbsoluteError(),
    "category": F1Score(task="multiclass", num_classes=15, average="macro"),
})
mt.update(
    {"fraud": fraud_pred, "ltv": ltv_pred, "category": cat_pred},
    {"fraud": fraud_y,    "ltv": ltv_y,    "category": cat_y},
)
```

**How it works.** Each task has an independent metric. `update` and `compute` accept and return dicts keyed by task name.

**When to use.**
- Multi-head models (one model, K heads, K different metrics).
- Reporting per-task numbers cleanly without keying confusion.

**Why prefer this over MetricCollection?**
`MetricCollection` is for *the same input* evaluated by many metrics. `MultitaskWrapper` is for *different inputs* per task. Use the right tool.

---

## 7. `ClasswiseWrapper` — per-class breakdown

```python
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics import Recall

per_class = ClasswiseWrapper(
    Recall(task="multiclass", num_classes=10, average=None),
    labels=class_names,
)
out = per_class.compute()
# {'recall_dog': 0.91, 'recall_cat': 0.87, ...}
```

**How it works.** Wraps a metric with `average=None` (returns per-class array) and turns the array into a labeled dict.

**When to use.**
- Logging per-class metrics in Lightning (`self.log_dict(per_class.compute())`).
- Diagnosis dashboards — find the worst classes.

**Gotchas.**
- The wrapped metric must support `average=None`.
- `labels=` is optional; without it you get `recall_0, recall_1, ...`.

---

## 8. `FeatureShare` — share encoder across multiple feature-based metrics

```python
from torchmetrics.wrappers import FeatureShare
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

shared = FeatureShare({
    "fid": FrechetInceptionDistance(feature=2048),
    "kid": KernelInceptionDistance(feature=2048),
})
```

**How it works.** All wrapped metrics share the *same* feature extractor. Inception V3 runs once per image; both FID and KID consume the same activations.

**When to use.**
- Generative model eval where you have FID + KID + IS + LPIPS — running Inception 4× per image is wasteful.
- Any setup with multiple feature-based metrics from the same backbone.

**Gotchas.**
- All wrapped metrics must use compatible feature shapes.
- Adds a small graph-build cost upfront; net win if you have ≥ 2 feature metrics.

---

## 9. `Running` × `MetricTracker` (composed)

You can stack them. **`MetricTracker(Running(metric, window))`** = "best rolling-window value across epochs." Useful for:

- Recommender engagement curves where the rolling window matters more than full-epoch.
- Long-running RL training where you want best-rolling-reward over training history.

---

## 10. `Transformations` — preprocess inputs before update

```python
from torchmetrics.wrappers import LambdaInputTransformer

# Apply softmax before passing to a metric that expects probabilities
calibrated = LambdaInputTransformer(
    BinaryCalibrationError(n_bins=15),
    transform_pred=lambda x: torch.sigmoid(x),
    transform_target=None,
)
```

**How it works.** Applies a function to `preds` and/or `target` before forwarding to the wrapped metric's `update`.

**When to use.**
- The model outputs logits but the metric expects probabilities.
- The target needs reshaping or one-hot conversion.
- You don't want to pre-transform everywhere — keep transformation co-located with the metric.

---

## 11. The "abstract" wrapper — write your own

The base class lives in `torchmetrics/wrappers/abstract.py`. To write a custom wrapper:

```python
from torchmetrics.wrappers.abstract import WrapperMetric
from torchmetrics import Metric

class MyWrapper(WrapperMetric):
    def __init__(self, metric: Metric, my_param):
        super().__init__()
        self.wrapped = metric
        self.my_param = my_param

    def update(self, *args, **kwargs):
        # do something around the wrapped metric
        self.wrapped.update(*args, **kwargs)

    def compute(self):
        return some_function_of(self.wrapped.compute(), self.my_param)

    def reset(self):
        self.wrapped.reset()
        super().reset()
```

**Cheap rule of thumb**: a wrapper *adds a new dimension* to the metric (CI, history, per-class). If you're just changing one number to another, you don't need a wrapper — subclass the metric instead.

---

## Composition patterns

These compositions show up in real production code. Memorize them.

| Composition | Purpose |
|---|---|
| `BootStrapper(F1Score(...))` | F1 with CI |
| `Running(MeanMetric())` | Rolling mean over window |
| `MetricTracker(Accuracy(...))` | Best val acc across epochs |
| `ClasswiseWrapper(F1Score(num_classes=K, average=None))` | Per-class F1 dict |
| `MetricCollection({...}).clone(prefix="train/")` | Independent train + val collections |
| `MultitaskWrapper({task_a: ..., task_b: ...})` | Multi-head model eval |
| `MultioutputWrapper(MAE(), num_outputs=68)` | Per-output regression |
| `MinMaxMetric(WER())` | Track worst-ever WER on accented speech |

---

## Interview drill-down — wrapper questions

> **Q.** What's the difference between `MultioutputWrapper` and `MultitaskWrapper`?

> **A.** `MultioutputWrapper`: same metric, K parallel outputs. `MultitaskWrapper`: different metrics, K different tasks. Inputs are dict-keyed in MultitaskWrapper.

>> **F1.** Why doesn't `MetricCollection` work for multi-task?

>> **A.** `MetricCollection` assumes a *shared input* and runs each metric on it. Multi-task models have *different inputs per task*. Trying to use `MetricCollection` would force every metric to consume the same input dict — fine in principle but loses the namespacing semantics.

> **Q.** When would you use `FeatureShare`?

> **A.** Whenever you have ≥ 2 metrics that all run the same encoder (typically Inception or CLIP). Run the encoder once, fan out activations to each metric. Generative-image eval is the canonical case (FID + KID + IS).

>> **F1.** What's the failure mode if I forget `FeatureShare`?

>> **A.** Each metric loads + runs Inception independently. 4 metrics × 50k images × 1 GFLOP = 200 GFLOP wasted. Eval time goes up linearly with the number of feature metrics.

> **Q.** Is `BootStrapper` differentiable?

> **A.** No — sampling indices breaks the graph. Use it for evaluation, not as a loss.

>> **F1.** How would you build a differentiable surrogate for "F1 with CI"?

>> **A.** You wouldn't. F1 itself isn't differentiable. The training-time analogue is a soft-F1 loss; the eval-time CI is what `BootStrapper` provides. Don't try to merge them.
