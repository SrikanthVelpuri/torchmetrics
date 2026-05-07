---
title: Spot the Bug
nav_order: 27
---

# Spot the Bug — 15 Buggy Snippets

Each snippet looks reasonable. Each has at least one bug. Find them, then check the answer. These are the bugs that bite real engineers in production.

> **How to use.** Read the snippet. Don't peek. Write down what's wrong. *Then* expand the answer. Tally how many you got right out of 15.

---

## Bug 1

```python
from torchmetrics import Metric
import torch

class MyMean(Metric):
    def __init__(self):
        super().__init__()
        self.sum = 0.0
        self.count = 0

    def update(self, x):
        self.sum += x.sum().item()
        self.count += x.numel()

    def compute(self):
        return self.sum / self.count
```

<details>
<summary><b>Answer</b></summary>

**Bug**: state is plain Python floats, not declared via `add_state`. Won't sync under DDP. Won't reset properly. Won't move to the right device.

**Fix**:
```python
self.add_state("sum",   default=torch.tensor(0.0), dist_reduce_fx="sum")
self.add_state("count", default=torch.tensor(0),   dist_reduce_fx="sum")
```

And use tensors in `update`:
```python
self.sum   += x.sum()
self.count += x.numel()
```

</details>

---

## Bug 2

```python
class MyAccuracy(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0))
        self.add_state("total",   default=torch.tensor(0))

    def update(self, preds, target):
        self.correct += (preds.argmax(-1) == target).sum()
        self.total   += target.numel()

    def compute(self):
        return self.correct.float() / self.total
```

<details>
<summary><b>Answer</b></summary>

**Bug**: missing `dist_reduce_fx`. After `_sync_dist`, state has shape `(world_size,)` instead of `()`. `compute()` returns the wrong number.

**Fix**: `add_state(..., dist_reduce_fx="sum")` on both states.

</details>

---

## Bug 3

```python
val_metrics = MetricCollection({
    "acc": Accuracy(task="multiclass", num_classes=10),
}).to(device)

for epoch in range(10):
    for x, y in val_loader:
        val_metrics.update(model(x), y)
    print(val_metrics.compute())
```

<details>
<summary><b>Answer</b></summary>

**Bug**: never calls `reset()`. Epoch 2 includes epoch 1 data. By epoch 10, state is dominated by past epochs.

**Fix**: call `val_metrics.reset()` at the end of each epoch.

</details>

---

## Bug 4

```python
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.metrics = {
            "acc": Accuracy(task="binary"),
            "f1":  F1Score(task="binary"),
        }

    def training_step(self, batch, _):
        x, y = batch
        logits = self.model(x)
        for name, m in self.metrics.items():
            self.log(name, m(logits, y))
        return F.binary_cross_entropy_with_logits(logits, y.float())
```

<details>
<summary><b>Answer</b></summary>

**Bug**: storing metrics in a plain dict bypasses `nn.Module.__setattr__`. Lightning never sees them. They stay on CPU; inputs are on GPU; you get device-mismatch errors.

**Fix**: use `MetricCollection` (a `ModuleDict`).

```python
self.metrics = MetricCollection({
    "acc": Accuracy(task="binary"),
    "f1":  F1Score(task="binary"),
})
```

</details>

---

## Bug 5

```python
def training_step(self, batch, _):
    x, y = batch
    logits = self.model(x)
    self.log("train/acc", self.train_acc.compute())
    self.train_acc.update(logits, y)
    return F.cross_entropy(logits, y)
```

<details>
<summary><b>Answer</b></summary>

**Bug** *(at least)*: passing `metric.compute()` to `self.log` instead of the metric. Lightning sees a scalar, not a metric — it never resets, never syncs correctly.

**Bug 2**: calling `compute()` *before* `update()` for this batch. The logged value is from the previous state.

**Fix**:
```python
self.log("train/acc", self.train_acc, on_step=True, on_epoch=True)
self.train_acc.update(logits, y)
```

Or even better:
```python
self.log("train/acc", self.train_acc(logits, y), on_step=True, on_epoch=True)
```

</details>

---

## Bug 6

```python
class StreamingMedian(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, values):
        self.values.append(values.median())

    def compute(self):
        return torch.tensor(self.values).median()
```

<details>
<summary><b>Answer</b></summary>

**Bug**: storing the *batch median* and computing the median of those — not the same as the global median.

**Fix**: store all values, compute median once.
```python
def update(self, values):
    self.values.append(values)

def compute(self):
    all_vals = torch.cat(self.values) if isinstance(self.values, list) else self.values
    return all_vals.median()
```

</details>

---

## Bug 7

```python
class CustomMetric(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("running_mean", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, x):
        # incrementally update running mean
        self.running_mean = 0.9 * self.running_mean + 0.1 * x.mean()

    def compute(self):
        return self.running_mean
```

<details>
<summary><b>Answer</b></summary>

**Bug**: `full_state_update=False` requires the state's reduction to merge correctly via the registered fx. EMA-style updates can't be merged via `mean`. The fast forward path will produce wrong batch values.

**Fix**: `full_state_update = True`. Or change the state to keep `(sum, count)` summable.

</details>

---

## Bug 8

```python
class FraudDollarLoss(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n",    default=torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, preds, target, fn_costs, fp_costs):
        decisions = (preds > 0.5)
        for i in range(target.numel()):
            if decisions[i] == 0 and target[i] == 1:
                self.loss += fn_costs[i]
            elif decisions[i] == 1 and target[i] == 0:
                self.loss += fp_costs[i]
            self.n += 1

    def compute(self):
        return self.loss / self.n
```

<details>
<summary><b>Answer</b></summary>

**Bug 1**: Python loop over batch elements. Slow on GPU. Should be vectorized.

**Bug 2**: `decisions[i] == 0` compares a bool to int — works but fragile.

**Fix**:
```python
decisions = (preds > 0.5).long()
is_fn = (decisions == 0) & (target == 1)
is_fp = (decisions == 1) & (target == 0)
self.loss += (is_fn.float() * fn_costs).sum() + (is_fp.float() * fp_costs).sum()
self.n    += target.numel()
```

</details>

---

## Bug 9

```python
metric = AUROC(task="binary").to(device)
preds_list, target_list = [], []
for batch in loader:
    preds_list.append(model(batch.x))
    target_list.append(batch.y)
preds = torch.cat(preds_list)
target = torch.cat(target_list)
metric.update(preds, target)
print(metric.compute())
```

<details>
<summary><b>Answer</b></summary>

**Bug** (subtle): not actually wrong for correctness, but **wastefully holds all predictions on GPU** before passing them in. For a 1M-sample eval, this OOMs.

**Fix**: stream — call `metric.update(...)` per batch. Then `compute()` once at the end.
```python
for batch in loader:
    metric.update(model(batch.x), batch.y)
print(metric.compute())
```

</details>

---

## Bug 10

```python
def evaluate_ddp(model, loader, world_size, rank):
    metric = Accuracy(task="multiclass", num_classes=10).to(rank)
    for batch in loader:
        try:
            metric.update(model(batch.x.to(rank)), batch.y.to(rank))
        except Exception:
            pass     # skip bad batches
    return metric.compute()
```

<details>
<summary><b>Answer</b></summary>

**Bug**: silent `try/except` lets one rank skip an `update` while others don't. Mismatched `_update_count` across ranks. At `compute()`, `_sync_dist` calls `all_gather` which expects every rank to participate identically — the skipping rank's empty state may cause hangs or shape mismatches.

**Fix**: never silently skip on one rank. Either log + re-raise, or apply the skip to *all ranks* via a coordinated mechanism (rank 0 broadcasts a "skip" signal).

</details>

---

## Bug 11

```python
class CustomMetric(Metric):
    def __init__(self):
        super().__init__()
        self.threshold = 0.5
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        # update threshold dynamically based on this batch
        self.threshold = preds.median().item()
        self.tp += ((preds > self.threshold) & (target == 1)).sum()

    def compute(self):
        return self.tp
```

<details>
<summary><b>Answer</b></summary>

**Bug**: `self.threshold` is a per-batch state that's not declared via `add_state`. It changes meaning across batches and across ranks (each rank computes its own median). State isn't reproducible.

**Fix**: either freeze the threshold at construction time, or track it as a proper state with a sensible reduction (e.g. accumulate predictions and compute the median in `compute()`).

</details>

---

## Bug 12

```python
metric = MeanAveragePrecision(box_format="xyxy")  # default device cpu
metric.update(preds, targets)
metric.to("cuda")
print(metric.compute())
```

<details>
<summary><b>Answer</b></summary>

**Bug**: `update` ran on CPU; then `.to("cuda")` is called but list states already hold CPU tensors. `compute()` may then mix devices, or `.to` may not move all internal state correctly for this metric.

**Fix**: set the device *before* updating. `metric = MeanAveragePrecision(box_format="xyxy").to("cuda")` first, then `update`.

</details>

---

## Bug 13

```python
class MyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n",   default=torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, x):
        self.sum += x.sum()
        self.n   += x.numel()

    def compute(self):
        result = self.sum / self.n
        self.sum = torch.tensor(0.0)
        self.n   = torch.tensor(0)
        return result
```

<details>
<summary><b>Answer</b></summary>

**Bug**: manually resetting state inside `compute()`. Multiple problems: (a) breaks DDP since `compute` runs on synced state and you reset to zero, then `unsync()` restores the local cache, putting things in a confused state; (b) caching is broken (next `compute()` returns 0).

**Fix**: never mutate state inside `compute()`. Use `reset()` from the outside.

</details>

---

## Bug 14

```python
class StreamingPearson(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("preds",  default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds, target):
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        return PearsonCorrCoef()(torch.cat(self.preds), torch.cat(self.target))
```

<details>
<summary><b>Answer</b></summary>

**Bug** *(subtle)*: this works but **defeats the whole point** of `PearsonCorrCoef`, which already maintains streaming Welford-style state in O(1) memory. Your wrapper holds all data in memory — at large eval scale this OOMs.

**Fix**: just use `PearsonCorrCoef` directly. It already does the right thing.

</details>

---

## Bug 15

```python
class TestMetric(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, x):
        return self.count + x.sum()

    def compute(self):
        return self.count
```

<details>
<summary><b>Answer</b></summary>

**Bug 1**: `update` returns a value (ignored by the wrapper) instead of mutating state. `count` is never updated. `compute()` always returns 0.

**Fix**: `self.count += x.sum()`.

**Bug 2** (latent): `count` defaults to a float tensor but is being summed with `x.sum()` — fine if x is float, but type-mismatched if x is int.

</details>

---

## Scoring

| Score | Meaning |
|---|---|
| 13–15 / 15 | Production-ready. You'd catch these in PR review. |
| 9–12 / 15 | Solid mid-level. Review the ones you missed; pattern-match. |
| 5–8 / 15 | Re-read [Custom Metrics](./custom-metrics.md) and [Distributed Training](./distributed-training.md) carefully. |
| < 5 / 15 | Start with [Core Concepts](./core-concepts.md) and [Metric Class Internals](./metric-class-internals.md). The patterns aren't yet in muscle memory. |

---

## The pattern behind the bugs

If you tally the bugs above, **80 % fall into these categories**:

1. **State not declared via `add_state`** (Bug 1, 11, 13) — silent DDP failure.
2. **Missing `dist_reduce_fx`** (Bug 2) — silent DDP failure.
3. **Lifecycle violations** — forgot reset (Bug 3), wrong order (Bug 5), state mutation in compute (Bug 13).
4. **`MetricCollection` vs dict confusion** (Bug 4) — Lightning device move fails.
5. **Per-batch math instead of state aggregation** (Bug 6, 8) — non-decomposable bugs.
6. **`full_state_update` mismatch** (Bug 7) — fast-path corruption.
7. **Silent `try/except` in DDP** (Bug 10) — hangs.

Internalize these seven failure modes. Almost every TorchMetrics bug you'll ever ship is one of them.
