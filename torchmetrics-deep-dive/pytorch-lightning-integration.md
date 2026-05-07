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
