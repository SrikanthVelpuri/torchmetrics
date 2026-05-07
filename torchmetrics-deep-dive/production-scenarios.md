---
title: Production Scenarios
nav_order: 12
---

# Production Scenarios

Metrics in research and metrics in production are different jobs. Research is "compute one number on a fixed eval set." Production is "compute many numbers, on a moving window of streaming traffic, attached to alerts and dashboards." This page is a collection of patterns from real deployments.

---

## Pattern 1 — offline evaluation harness

The simplest case. You have a model and an eval set; you want a CSV of metrics.

```python
from torchmetrics import MetricCollection, F1Score, AUROC, CalibrationError
from torchmetrics.wrappers import BootStrapper

metrics = MetricCollection({
    "f1":     F1Score(task="binary"),
    "auroc":  AUROC(task="binary"),
    "ece":    CalibrationError(task="binary", n_bins=15),
})
metrics_with_ci = MetricCollection({
    name: BootStrapper(metric, num_bootstraps=1000, quantile=torch.tensor([0.025, 0.975]))
    for name, metric in metrics.items()
}).to(device)

with torch.no_grad():
    for x, y in loader:
        metrics_with_ci.update(model(x), y)

print(metrics_with_ci.compute())
```

**Why bootstrap**: a point estimate without an interval is hard to act on. Engineering managers tend to overreact to 0.901 vs. 0.899 unless you tell them the CI is ±0.005.

---

## Pattern 2 — online drift monitoring

You're serving a model, you log `(prediction, eventually-observed-label)` pairs, and you want to know when the model is silently degrading.

```python
from torchmetrics import F1Score
from torchmetrics.wrappers import Running

# rolling 10k-sample F1 over the last hour or so
rolling_f1 = Running(F1Score(task="binary"), window=10_000).to(device)

def on_event(prediction, label):
    rolling_f1.update(prediction.unsqueeze(0), label.unsqueeze(0))
    return rolling_f1.compute()
```

`Running` keeps a circular history of metric states. `compute()` reduces only the recent window. Plug the value into Prometheus / Datadog / your own time series.

For larger deployments, do the rolling window per-segment (per region, per model version, per traffic source) and alert on per-segment regressions.

---

## Pattern 3 — A/B testing two models

```python
metric_A = F1Score(task="binary").to(device)
metric_B = F1Score(task="binary").to(device)

for x, y in loader:
    metric_A.update(model_A(x), y)
    metric_B.update(model_B(x), y)

print("A:", metric_A.compute(), "B:", metric_B.compute())
```

That's the easy part. The hard part is **statistical significance**. Wrap each metric with `BootStrapper` to get confidence intervals, or compute a paired bootstrap difference (sample with replacement from `(pred_A_i, pred_B_i, y_i)` triples and take the difference).

If your metrics are non-decomposable (F1, AUROC), do **not** estimate variance from per-batch numbers — you'll be wrong by orders of magnitude.

---

## Pattern 4 — segmented metrics

Aggregate KPIs hide failure modes. A model with 95 % overall accuracy can be 50 % on the long tail.

```python
from torchmetrics import Accuracy, MetricCollection

# pre-create one metric per segment
metrics = {
    seg: Accuracy(task="multiclass", num_classes=K).to(device)
    for seg in segments
}

for x, y, seg in loader:
    for s in seg.unique():
        idx = (seg == s)
        metrics[s.item()].update(model(x[idx]), y[idx])

for s, m in metrics.items():
    print(s, m.compute())
```

In production, segments are often: country, language, device type, traffic source, model version, time of day. Set up a dashboard that surfaces the **worst** segment, not the average.

---

## Pattern 5 — multi-task / multi-head models

A model with K heads → K loss components → K (or more) metric panels.

```python
from torchmetrics.wrappers import MultitaskWrapper, MultioutputWrapper
from torchmetrics import F1Score, MeanAbsoluteError

metrics = MultitaskWrapper({
    "fraud":    F1Score(task="binary"),
    "ltv":      MeanAbsoluteError(),
    "category": F1Score(task="multiclass", num_classes=15, average="macro"),
})

# In your eval loop
metrics.update(
    {"fraud": fraud_logits, "ltv": ltv_pred, "category": cat_logits},
    {"fraud": fraud_y,      "ltv": ltv_y,    "category": cat_y},
)
```

`MultitaskWrapper` keeps each task's metric independent and namespaced. Pair with `MetricCollection` per-task for richer breakdowns.

---

## Pattern 6 — checkpointing metric state mid-epoch

By default, metric state is *not* persisted. If you crash mid-epoch and resume, you start eval from scratch. That's fine for fast loops, painful for hour-long evaluations.

```python
class CheckpointableF1(F1Score):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # convert states to persistent so they go in state_dict
        for name in self._defaults:
            self._persistent[name] = True
```

Now `model.state_dict()` includes the metric state. Or simpler: pass `persistent=True` per state in your custom metric's `add_state()`.

---

## Pattern 7 — exporting metrics to a tracking system

```python
import wandb

values = metrics.compute()    # dict of tensors
wandb.log({k: v.item() for k, v in values.items()}, step=global_step)
```

Notes:

- Always `.item()` (or `.tolist()` for vectors). W&B / TB / MLflow want Python floats.
- Don't log raw list states (`metric.metric_state`) — they can be huge.
- Snapshot `metric.compute()` once and reuse, since each call returns a fresh dict.

---

## Pattern 8 — reproducibility

To make a metric deterministic:

1. Set seeds (`torch.manual_seed`, `numpy.random.seed`).
2. For stochastic metrics (`BootStrapper`), pass `generator=torch.Generator().manual_seed(...)`.
3. Lock down the input tokenizer / preprocessing (huge for text metrics).
4. Pin TorchMetrics, PyTorch, and any backbone (Inception, BERT) versions in `requirements.txt`. Generative metrics like FID change subtly with backbone weights.

---

## Pattern 9 — fairness metrics in CI

Enforce fairness gates:

```python
from torchmetrics.classification import BinaryFairness

fairness = BinaryFairness(num_groups=4)
fairness.update(preds, target, groups)
report = fairness.compute()  # ratios

assert report["DP"] >= 0.8, "demographic parity below threshold"
```

Run this in your model-validation CI alongside accuracy gates. A model that passes accuracy but fails fairness should not auto-promote.

---

## Pattern 10 — capturing metric regressions per-PR

In a typical research codebase, each PR re-runs eval and compares metrics:

```python
# tests/test_metrics_regression.py
def test_no_regression(snapshot):
    metrics = run_full_eval()
    snapshot.assert_match(metrics, "metrics.json", tolerance=1e-3)
```

This catches subtle bugs: a tokenizer change, a batch-size change, an unintended `eval()`/`train()` swap, etc.
