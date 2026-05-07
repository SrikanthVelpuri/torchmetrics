---
title: PyTorch Lightning Integration
nav_order: 10
---

# PyTorch Lightning Integration

TorchMetrics was born inside PyTorch Lightning, and the integration is by design — but it's not magic. This page explains exactly what Lightning does for you, and what you still have to think about.

---

## The recommended pattern

```python
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, MetricCollection, F1Score

class LitClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Linear(20, num_classes)

        # Make metrics module attributes — Lightning will auto-move them to device
        metrics = MetricCollection({
            "acc": Accuracy(task="multiclass", num_classes=num_classes),
            "f1":  F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        })
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics   = metrics.clone(prefix="val/")

    def training_step(self, batch, _):
        x, y = batch
        logits = self.model(x)
        loss   = nn.functional.cross_entropy(logits, y)
        # Pass the metric module — NOT metric.compute()
        self.log_dict(self.train_metrics(logits, y), on_step=True, on_epoch=True)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self.model(x)
        self.val_metrics.update(logits, y)         # accumulate

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())   # single epoch number
        self.val_metrics.reset()
```

Notice the **two distinct patterns**:

- During **training**, log `self.train_metrics(logits, y)` — this calls `forward()` and gives you both an updated state and a per-step value. Lightning's `on_step=True, on_epoch=True` will automatically log a step value during the epoch and call `compute()` at epoch end.
- During **validation**, prefer **explicit `update` then `compute` then `reset`**. It's harder to get wrong and clearer about when sync happens.

---

## Why register metrics as module attributes

A `Metric` is a `nn.Module`. When you do `self.acc = Accuracy(...)`, Lightning's framework:

1. Calls `.to(device)` on the Lightning module → metric state moves with it.
2. Includes the metric in the `state_dict` *if* the metric has `persistent=True` states. By default it does not, so checkpoints aren't bloated.
3. Calls `.train()` / `.eval()` propagation (no-op for metrics, but valid).
4. Tracks the metric for hooks (`on_validation_epoch_end`, `on_train_epoch_end`).

If you store a metric in a `dict` instead of `setattr`, none of that happens — you'll see device-mismatch errors and stale state.

---

## `self.log` and metrics — the contract

`self.log(name, value, ...)` accepts:

- a scalar tensor / float
- **or a `Metric` instance**

When you pass a `Metric`, Lightning understands the lifecycle:

| Argument | What Lightning does |
|---|---|
| `on_step=True` | Call `metric.forward(...)` and log per-step. |
| `on_epoch=True` | Call `metric.compute()` at epoch end and log; then `reset()`. |
| `sync_dist=True` | Tell the metric to all_gather state before computing. |
| `prog_bar=True` | Show in tqdm. |
| `reduce_fx` | Ignored for metrics — they have their own DDP reduction. |

**Important**: don't call `self.log("acc", metric.compute())`. That eagerly computes and logs the *current* state, then leaves it un-reset — a classic mistake. Pass the metric itself.

---

## Step-level vs. epoch-level logging

```python
self.log("train/acc", self.train_acc, on_step=True, on_epoch=True)
```

- **`on_step=True`** logs `metric.forward(preds, target)` — the per-batch value (state-aware, not just accuracy on this batch).
- **`on_epoch=True`** logs `metric.compute()` at the end of the epoch.

Setting both is fine; you'll see two separate series in TensorBoard / W&B (`train/acc_step` and `train/acc_epoch`).

For validation / test, prefer `on_step=False, on_epoch=True` — you only care about the epoch number.

---

## DDP and `sync_dist`

Two layers of synchronization can confuse people:

1. **TorchMetrics' own `_sync_dist`** — the metric's *state* is `all_gathered` and reduced.
2. **Lightning's `sync_dist=True`** — would also gather a *scalar* across ranks if the value were a tensor.

When you log a `Metric` instance, you do **not** need `sync_dist=True`. The metric handles it. Setting `sync_dist=True` on a metric is at best harmless, at worst (depending on Lightning version) it duplicates work or warns.

For non-metric scalars (e.g. `self.log("loss", loss)`), you *do* want `sync_dist=True` if you want a global mean.

---

## `MetricCollection.clone(prefix=...)`

A common bug: defining one shared `MetricCollection` and using it in both training and validation. Each phase `update`s the same state — your validation Accuracy is contaminated by the training set.

`metrics.clone(prefix="val/")` creates an independent copy, with each metric prefixed in the log so the two series don't collide.

---

## Common mistakes (from real Lightning bug reports)

1. **Forgot to register as attribute** — metric stays on CPU even after `.to(device)`. Fix: assign with `self.<name> = metric`.
2. **Stored in a dict** — same problem. Use `MetricCollection` (which is a `ModuleDict`) instead.
3. **`compute()` outside an epoch hook** — works, but the value reflects whatever state was there, possibly across phases.
4. **Manual `update` plus `self.log(metric)`** — `self.log` then calls `forward`, which `update`s *again*, double-counting. Pick one or the other.
5. **`sync_dist=True` on a metric** — redundant; can mask the metric's own sync logic.
6. **Resetting in `validation_step`** — leaves state for one batch only, returns nonsense at epoch end.

---

## Manual control: when Lightning's defaults aren't enough

Sometimes you want behavior Lightning's `self.log` doesn't expose — e.g. logging metric *per dataloader* in a multi-dataset eval. Use the explicit lifecycle:

```python
def validation_step(self, batch, batch_idx, dataloader_idx=0):
    x, y = batch
    self.val_metrics[dataloader_idx].update(self(x), y)

def on_validation_epoch_end(self):
    for i, metrics in enumerate(self.val_metrics):
        for name, value in metrics.compute().items():
            self.log(f"val/loader_{i}/{name}", value)
        metrics.reset()
```

---

## Lightning Fabric

If you use Fabric instead of Trainer, the contract is simpler — you call `update`/`compute`/`reset` yourself, and TorchMetrics' built-in DDP sync still works because Fabric initializes `torch.distributed` for you. Lightning Fabric does not auto-move metrics; you must do `metric = fabric.setup_module(metric)` or `metric.to(fabric.device)` yourself.

---

## Interview Drill-Down (multi-level follow-ups)

### Q1. Why pass the metric object to `self.log()` instead of `metric.compute()`?

> Lightning treats a `Metric` instance specially — it understands the `forward → compute → reset` lifecycle and aligns with `on_step` / `on_epoch` hooks. Passing `metric.compute()` collapses the metric to a scalar, and Lightning has no idea it should reset, sync, or accumulate.

  **F1.** What goes wrong if you do `self.log("acc", metric.compute())` inside training_step?

  > Lightning logs whatever scalar `compute()` returned — likely the cached value from the last sync, possibly stale. State isn't reset, so next epoch's `compute()` includes this epoch's data. The metric "value" trends look correct for ~1 epoch and then go monotonic forever.

    **F1.1.** Why doesn't Lightning warn when you make this mistake?

    > It can't tell — the argument is a Tensor, indistinguishable from any other scalar log. Recent Lightning versions warn for some patterns, but in general the only reliable signal is your weird-looking metric trace.

      **F1.1.1.** How do you guard against this in code review?

      > Add a lint rule (or a custom regex pre-commit hook) that flags `.compute()` calls inside `training_step` / `validation_step`. Pair with a unit test that asserts metric `_update_count == 0` at the start of each epoch.

### Q2. What's the actual difference between `on_step=True, on_epoch=True` and only `on_epoch=True`?

> `on_step=True` calls `metric.forward(...)` per batch and logs the per-batch value. `on_epoch=True` calls `metric.compute()` at the appropriate epoch hook and logs that. Setting both means you get two TensorBoard series: `acc_step` and `acc_epoch`.

  **F1.** Doesn't `forward()` and `update()` together double-count state?

  > No — `forward()` calls `update()` exactly once internally. The two-pass behavior of `_forward_full_state_update` resets and re-updates *to compute the per-batch value*, but the global state increments only once. (See Metric Class Internals for the full trace.)

    **F1.1.** What if I want only the epoch number for training (no per-step series)?

    > `on_step=False, on_epoch=True`. Lightning will only log at epoch end. This is cheaper to log (one number per epoch) but you lose the mid-epoch stability signal.

      **F1.1.1.** Is per-step logging really cheap in DDP?

      > Yes, *if* the metric doesn't sync per step (the default). Logging the local-rank `forward()` value is roughly free. The cost spike comes from `dist_sync_on_step=True`, which is unrelated to whether you log.

### Q3. You define metrics in `__init__` — why is that critical?

> So Lightning's `.to(device)` traversal moves them to the right device. Storing a metric in a dict (`self.metrics = {"acc": Accuracy()}`) bypasses `nn.Module.__setattr__` and Lightning never sees it — device-mismatch error follows.

  **F1.** Why does `MetricCollection` (which is a dict-like) work, then?

  > Because `MetricCollection` extends `nn.ModuleDict`, which *does* register children. Setting `self.metrics = MetricCollection(...)` is correct; setting `self.metrics = {...}` is broken.

    **F1.1.** What if you have a list of metrics (one per dataloader)?

    > Use `nn.ModuleList(metrics)` so they're properly registered. Plain Python lists fail the same way as dicts.

      **F1.1.1.** Why doesn't TorchMetrics ship a `MetricList` class?

      > `nn.ModuleList` does the job; it's polymorphic with `Metric` already. No need for a wrapper.

### Q4. `metrics.clone(prefix="train/")` — what does it actually do?

> Deep-copies the entire `MetricCollection` and prepends `"train/"` to every key. Each cloned metric has its own state. Lightning's auto-namespacing then produces `"train/acc"`, `"train/f1"`, etc., in the log.

  **F1.** Why deep-copy and not just rename keys?

  > Different states. If you used the same instance for train and val, every `update()` you do in `validation_step` adds to the *same* state as `training_step`. Validation accuracy would silently include training data.

    **F1.1.** How much memory does cloning cost?

    > As much as the original metric state. Tensor states: trivial. List states: doubles. For 50k-sample AUROC list states, clone before fitting — and consider `compute_on_cpu=True`.

      **F1.1.1.** When is it OK to *not* clone?

      > Test-time-only metrics (no training-time use). Or read-only inspection metrics where you accept they'll see all phases. Default to clone; deviate only with intent.
