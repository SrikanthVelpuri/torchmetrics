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

---

## Interview Drill-Down (multi-level follow-ups)

### Q1. Walk through how you'd build the metric layer of a real-time recommender at 50k QPS.

> Per-region worker holds one or more `Running(metric, window=...)` instances. Updates batch incoming events for ~50 ms or 1k items, whichever comes first. Worker emits `metric.compute()` to the time-series store every 30 s. Sharded by region; aggregated offline for global numbers.

  **F1.** Why a rolling window, not a per-minute reset?

  > The window smooths over per-batch noise. A 30 s tick reading a 10k-event window is more stable than a 30 s tick reading the last 30 s of events.

    **F1.1.** What's the right window size?

    > Should cover at least 1k events (statistical stability) and shouldn't be so long that drift is invisible. For 50k QPS, a window of 30k events ≈ 600 ms of traffic — fast-reacting but stable. For lower-QPS workloads, scale up.

      **F1.1.1.** How do you persist window state across worker restarts?

      > `metric.metric_state` is a dict of tensors — pickle it on shutdown, restore on startup. State is small (window-sized), persistence is cheap. Without persistence, every restart forgets the last window's worth of data.

### Q2. You're shipping a model whose F1 improved by 0.005 over baseline on the eval set. Is that a real win?

> Run a paired bootstrap. `BootStrapper(F1, num_bootstraps=1000, quantile=...)` for both models, paired on sample index. If the 95 % CI on `F1_new − F1_baseline` excludes zero, it's a real win. If it includes zero, the eval set isn't large enough to distinguish the two.

  **F1.** What sample size do you need to detect Δ = 0.005 reliably?

  > For per-sample F1 standard deviation σ ~ 0.02, you need `n ≥ (1.96 × σ / Δ)² × 2 ≈ 60 k samples` for paired difference at 95 % confidence with 80 % power. If your eval set is 5k, you cannot reliably detect 0.005.

    **F1.1.** What do you do if the eval set isn't big enough?

    > Two options: (a) collect more eval data (slow, expensive); (b) report the result honestly with the CI and either ship behind a flag or run online A/B as the final arbiter.

      **F1.1.1.** What's the right A/B sample size?

      > Same math: power analysis on online success metric. If you're moving CTR by 0.5 % and per-user variance is large, you need millions of users — typical for online A/B.

### Q3. Production F1 starts drifting downward over 2 weeks. What's your investigation order?

> 1. Input distribution shift — has the feature distribution moved? 2. Label distribution shift — is the positive rate changing? 3. Label-pipeline integrity — is the labeling lag introducing bias? 4. Model staleness — has there been a re-train pause? 5. Infra — has anything changed in feature serving / model deployment?

  **F1.** How do you actually measure (1) and (2) in TorchMetrics?

  > Custom metrics: `KLDivergence` between current feature batches and a reference distribution; `MeanMetric` on the positive rate. Add them to your monitoring stack alongside F1.

    **F1.1.** What's a good alert threshold for KL?

    > Empirical. Establish a baseline KL during a stable period; alert when current KL is >3σ above baseline for sustained windows. Hard thresholds are domain-specific.

      **F1.1.1.** Why sustained windows, not single-point alerts?

      > Single-point alerts are noisy (transient batch oddities). Sustained ⇒ real drift. Standard SRE pattern: "alert when 5 of last 10 windows breach."

### Q4. Two production model versions are A/B tested. Reporting metrics show v2 wins by Δ = 0.5 % on conversion. How do you decide to launch?

> A bad answer: "Δ > 0, ship it." A good answer: (1) check the paired-bootstrap CI excludes zero; (2) check no segment regresses by more than ε; (3) check secondary metrics (latency, fairness, abuse) didn't degrade; (4) check the experiment ran long enough to capture novelty effects (~2 weeks for consumer products).

  **F1.** Why "novelty effects"?

  > New experiences often see a brief uptick from curiosity, then revert. A 1-day A/B that wins by 0.5 % might be flat by week 2.

    **F1.1.** How do you measure novelty effect?

    > Cohort analysis. Bucket users by exposure-day; plot per-cohort metric over time. If new cohorts perform worse than old cohorts on the new arm, novelty is at play.

      **F1.1.1.** Express that as a metric pattern in TorchMetrics?

      > Per-cohort `Running` metric with cohort-tagged updates. `MetricCollection` keyed by cohort. Compare cohort metrics over time as a regression-style trend test.
