---
title: Getting Started
nav_order: 2
---

# Getting Started

This page gets you from zero to a working metric in under five minutes.

---

## Installation

```bash
pip install torchmetrics            # core metrics only
pip install "torchmetrics[image]"   # adds FID, IS, KID, LPIPS, …
pip install "torchmetrics[text]"    # adds BLEU, ROUGE, BERTScore, …
pip install "torchmetrics[audio]"   # adds PESQ, SI-SDR, STOI, …
pip install "torchmetrics[all]"     # everything
```

Optional extras pull in heavy dependencies (e.g. `transformers`, `pesq`, `nltk`) only when you actually need them.

---

## The two APIs

TorchMetrics ships every metric in two flavors. Pick whichever fits.

### Functional API — fire and forget

```python
import torch
from torchmetrics.functional import accuracy

preds  = torch.tensor([0, 1, 1, 0])
target = torch.tensor([0, 1, 0, 0])

acc = accuracy(preds, target, task="multiclass", num_classes=2)
print(acc)   # tensor(0.7500)
```

Use it when:

- You're computing a one-shot metric on a fixed set of predictions.
- You're inside `torch.no_grad()` evaluating a frozen model.
- You don't need to accumulate across batches.

### Modular API — accumulate across batches and devices

```python
import torch
from torchmetrics import Accuracy

metric = Accuracy(task="multiclass", num_classes=10)

for x, y in val_loader:
    logits = model(x)
    metric.update(logits, y)         # accumulate state, no return

acc = metric.compute()               # one final number
metric.reset()                       # clear state for next epoch
```

Use it when:

- You're streaming many batches and want the global metric.
- You're on multi-GPU (DDP) and need `all_gather`-correct aggregation.
- You're using PyTorch Lightning (`self.log("acc", self.acc)` will Just Work).

---

## A minimal training loop

```python
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, MetricCollection

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Linear(20, 3).to(device)

# Group metrics so you only pay one forward through the data once.
val_metrics = MetricCollection({
    "acc": Accuracy(task="multiclass", num_classes=3),
    "f1":  F1Score(task="multiclass", num_classes=3, average="macro"),
}).to(device)

opt = torch.optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    # ---- train ----
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()

    # ---- validate ----
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val_metrics.update(model(x), y)

    print(f"epoch {epoch}: {val_metrics.compute()}")
    val_metrics.reset()
```

**What to notice**

- `MetricCollection` accepts a dict and keys the output by your names.
- `.to(device)` moves *all* internal state (running counts) onto the GPU.
- `compute()` returns a plain `dict[str, Tensor]`.
- You **must** call `reset()` between epochs, or last epoch's state leaks forward.

---

## "Use it like a function" — `forward()`

Calling a `Metric` like a function (e.g. `metric(preds, target)`) is equivalent to *both*:

1. Updating the running state with this batch.
2. Returning the metric value *for this batch only*.

```python
metric = Accuracy(task="binary")
batch_acc = metric(preds, target)   # value on this batch
total_acc = metric.compute()        # value across every batch seen so far
```

This is what you want when you also need a per-step value (e.g. to log a step-level training accuracy curve).

---

## Common pitfalls (read this before debugging)

1. **Forgot `reset()`** — your "epoch 2" number includes data from epoch 1.
2. **Forgot `.to(device)`** — you'll see `Expected all tensors to be on the same device`. The base class actually rewrites this error message for you with a clearer hint.
3. **Wrong `task=` argument** — many classification metrics now require `task="binary" | "multiclass" | "multilabel"`. Old code that passed integer `num_classes` only will fail at construction.
4. **Mixing functional and modular** — calling `f1(preds, target)` (functional) inside a loop and averaging the result is **not** the same as `F1Score().update(...)` then `compute()`. The latter is correct.
5. **Calling `compute()` before any `update()`** — raises and warns. Always update at least once.

---

## What's next?

- [Core Concepts](./core-concepts.md) — the mental model.
- [Metric Class Internals](./metric-class-internals.md) — what `add_state` and `forward` actually do.
- [Custom Metrics](./custom-metrics.md) — when you need to build your own.
