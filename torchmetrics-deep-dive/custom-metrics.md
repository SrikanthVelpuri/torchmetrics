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

---

## Interview Drill-Down (multi-level follow-ups)

### Q1. You wrote a custom metric and it works on 1 GPU, fails silently on 4. What do you check first?

> Three things, in order: (1) every state is a `Tensor` or list of `Tensor` (not Python list of floats); (2) every state has an explicit `dist_reduce_fx`; (3) reductions are associative. Most "works on 1 GPU" failures are missing reductions — state stacks across ranks but doesn't reduce, then `compute()` does the wrong math.

  **F1.** What does "doesn't reduce" actually look like in state?

  > After sync, your `tp` tensor has shape `(world_size, ...)` instead of `(...)` — `compute()` treats the rank dimension as data. The bug is in `add_state(..., dist_reduce_fx=None)`. Specify a reduction.

    **F1.1.** A state can't be both a tensor and a list. How do you decide?

    > Tensor for sufficient statistics (counts, sums) — cheap, summable, DDP-friendly. List for raw observations (preds, targets) when the metric needs them all. Almost every metric you write should be tensor-state; only fall back to list-state when the math truly demands it.

      **F1.1.1.** A senior asks you to convert a list-state metric to tensor-state. How?

      > Find the sufficient statistic. For AUROC, it's binned counts of positives and totals — bin predictions, accumulate per-bin totals. The metric becomes approximate but tensor-state, dramatically more memory-efficient.

### Q2. You set `full_state_update=False` but `forward()` returns wrong batch values. Why?

> `_forward_reduce_state_update` assumes batch state can be merged into the global state via the registered reduction. If your reduction is non-trivial (e.g. you need the full population to compute), the merge will be wrong.

  **F1.** When is the merge wrong even with `sum` reduction?

  > When `compute()` divides by something that depends on global state. Example: a "running mean" metric where compute is `sum / count`; both are sum-reduced. The fast-path forward computes `batch_sum / batch_count` and that's the correct *batch value*. But if you accidentally read `self.global_count` inside compute, the batch value is wrong.

    **F1.1.** How do you tell?

    > Run a unit test: do one big update, vs. two half-size updates, both via `forward()`. The two batch values must be `forward(batch1) → m1, forward(batch2) → m2`, with `compute()` matching the single-update result. If the batch values are wrong but `compute` is right, your update is fine but the fast-path forward is broken — set `full_state_update=True`.

      **F1.1.1.** What's the cost of `full_state_update=True`?

      > Two `update()` calls per `forward()`. For most metrics this is a minor slowdown. For metrics where `update` is expensive (encoding text via BERT, etc.), it's substantial — invest in fixing the fast-path conditions instead.

### Q3. How do you make a custom metric that depends on **per-sample weights**?

> Add `weights` as a third argument to `update`. Inside, multiply `(preds, target)` element-wise by the weights before accumulating. State stays the same shape; only `update`'s body changes.

  **F1.** What if weights aren't always provided (back-compat)?

  > Make `weights` default to `None`. Inside `update`, branch: `weights = weights if weights is not None else torch.ones_like(target)`. The metric still works for un-weighted callers.

    **F1.1.** Could you do this in `forward()` rather than asking callers to pass weights every time?

    > Yes — but `forward(*args, **kwargs)` and `update(*args, **kwargs)` share signatures. Adding optional `weights` to both is the clean approach. If the metric should *always* be weighted, make `weights` required.

      **F1.1.1.** What if some callers pass weights, others don't, and your `MetricCollection` mixes them?

      > Compute groups will fail (different update signatures). Either (a) make weighted and unweighted versions distinct metrics in the collection, or (b) make all metrics accept the unified signature with optional weights. Option (b) is cleaner long-term.

### Q4. How do you write a custom metric that does **bootstrap CI** internally?

> Maintain N parallel copies of your states (e.g. `sum_i_j` for j in 1..N). For each `update`, sample which copies the batch goes into via Poisson(1) (the standard bootstrap with-replacement). At `compute`, you get N metric values; report mean ± quantile range.

  **F1.** Why Poisson(1) and not "pick K samples uniformly"?

  > Poisson(1) is the asymptotic limit of with-replacement sampling — each item appears in each bootstrap copy with the right multinomial expectation. It's also memory-efficient: you don't need to store sample IDs, just sample-by-sample inclusion counts.

    **F1.1.** Isn't this just `BootStrapper`?

    > Yes — and you should use `BootStrapper` instead of writing your own unless you need a custom resampling strategy (e.g. paired bootstrap, stratified bootstrap). Most use cases are covered by the wrapper.

      **F1.1.1.** When would you write a custom paired bootstrap?

      > A/B comparison of two models on the same data. The right CI is over the *paired difference*, not over each model independently. Custom wrapper that maintains synced sample indices for both models.
