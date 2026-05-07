---
title: Custom Metrics
nav_order: 11
---

# Custom Metrics

Sooner or later, you need a metric TorchMetrics doesn't ship — a domain-specific KPI, a weighted loss, a fairness ratio. This page is the recipe for writing one *correctly* the first time.

---

## The five-step recipe

1. **Inherit `Metric`.**
2. **Set the metadata flags** (`is_differentiable`, `higher_is_better`, `full_state_update`).
3. **Declare state with `add_state` in `__init__`.**
4. **Implement `update(*args, **kwargs)`** — mutate state, return nothing.
5. **Implement `compute()`** — pure function from state to value.

That's it. The base class handles device placement, DDP sync, caching, JIT-safety, and `reset()`.

---

## Example 1 — a simple custom metric

A weighted average regression error: `Σ wᵢ |yᵢ − ŷᵢ| / Σ wᵢ`.

```python
import torch
from torch import Tensor
from torchmetrics import Metric

class WeightedMAE(Metric):
    is_differentiable = True
    higher_is_better  = False
    full_state_update = False     # state is summable, so use the fast path

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_weights",   default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, weights: Tensor) -> None:
        if preds.shape != target.shape != weights.shape:
            raise ValueError("preds, target, and weights must share shape")
        abs_err = (preds - target).abs()
        self.sum_abs_error += (abs_err * weights).sum()
        self.sum_weights   += weights.sum()

    def compute(self) -> Tensor:
        return self.sum_abs_error / self.sum_weights.clamp(min=1e-12)
```

Why this is correct out of the box:

- Both states are scalar tensors with `sum` reduction → DDP-correct.
- `update()` does not return anything; `forward()` will still work.
- `compute()` is a pure function. No globals, no mutation.
- We protect against divide-by-zero in `compute`, not in `update` — that way state stays "raw."

---

## Example 2 — a metric that needs the full distribution

Median absolute error needs all residuals (no streaming algorithm gives an exact median in O(1) memory).

```python
class MedianAbsoluteError(Metric):
    is_differentiable = False
    higher_is_better  = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("residuals", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.residuals.append((preds - target).abs())

    def compute(self) -> Tensor:
        residuals = torch.cat(self.residuals, dim=0) if isinstance(self.residuals, list) \
                    else self.residuals
        return residuals.median()
```

Notes:

- `default=[]` and `dist_reduce_fx="cat"` — list state, concatenated across ranks.
- After sync, the state is already a tensor; before sync, it's a list. Handle both.
- Consider `compute_on_cpu=True` if your eval set is huge — the residuals tensor will live on CPU between updates.

---

## Example 3 — a per-class custom metric

Wrap any per-class metric you want with `ClasswiseWrapper`:

```python
from torchmetrics import Recall
from torchmetrics.wrappers import ClasswiseWrapper

per_class_recall = ClasswiseWrapper(
    Recall(task="multiclass", num_classes=10, average=None),
    labels=class_names,        # optional list of names
)
```

`per_class_recall.compute()` returns `{"recall_dog": 0.9, "recall_cat": 0.8, ...}`, ready to log.

---

## Example 4 — a "running window" metric

Sometimes you want the metric *over the last 100 batches*, not the entire epoch. Use `Running`:

```python
from torchmetrics.wrappers import Running
from torchmetrics import MeanSquaredError

running_mse = Running(MeanSquaredError(), window=100)
```

It maintains a circular buffer of state snapshots — useful for streaming dashboards.

---

## Example 5 — a bootstrap CI

```python
from torchmetrics.wrappers import BootStrapper
from torchmetrics import F1Score

bootstrapped_f1 = BootStrapper(
    F1Score(task="multiclass", num_classes=10),
    num_bootstraps=1000,
    quantile=torch.tensor([0.05, 0.95]),
)
```

`compute()` returns mean F1 plus the 5/95 percentiles. Standard for reporting confidence intervals on benchmarks.

---

## When to set `full_state_update=False`

The fast `forward()` path requires that batch state can be merged into the global state via the registered reduction. That's true when *all* states reduce by `sum`, `mean`, `min`, `max`, or `cat`.

Set `full_state_update=False` when:

- You only have tensor states with sum/mean/min/max reductions, **or**
- You only have list states with `cat`, **or**
- Both of the above.

Leave it `True` (or `None`) when:

- Your `update()` reads state in a non-trivial way (e.g. updates a running quantile estimator).
- You override `merge_state` because the standard reductions don't fit.

When in doubt, leave it `True` and benchmark — `_forward_full_state_update` is correct for everything, just twice as expensive.

---

## Naming and metadata conventions

To match the rest of the library, your metric should:

- Use `CamelCase` class names (`WeightedMAE`, not `weighted_mae`).
- Document `preds` and `target` shapes in the docstring with a "As input to ``forward`` and ``update``" section.
- Provide a docstring with a runnable doctest — that doubles as a sanity test.
- Pair with a functional version in `<package>/functional.py`.
- Set `is_differentiable`, `higher_is_better`, `full_state_update`.
- Optionally implement `plot(...)` returning a Matplotlib axis.

---

## Don'ts (the gotchas)

1. **Don't store Python lists / floats** as state. They won't sync. Use tensors.
2. **Don't call `torch.distributed`** inside `update()`. Let the base class do sync at `compute()` time.
3. **Don't read `self._defaults`** at runtime — those are deep-copied templates, not your live state.
4. **Don't return values from `update()`** — they're ignored; users will be confused.
5. **Don't compute** in `update()` — keep math in `compute()` so the metric is restartable.
6. **Don't mutate inputs** — copy if you need to.
7. **Don't depend on insertion order** — DDP `cat` does not guarantee order across ranks.
