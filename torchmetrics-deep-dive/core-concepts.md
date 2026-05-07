---
title: Core Concepts
nav_order: 3
---

# Core Concepts

This page explains the mental model behind TorchMetrics. Once you internalize the **state → update → compute → reset** lifecycle, almost everything else (DDP, Lightning, custom metrics) is "just" a consequence.

---

## The four primitives

Every modular metric in TorchMetrics is built from four primitives:

| Primitive | Purpose |
|---|---|
| **`add_state(...)`** | Declare a piece of state (a tensor or a list) that lives across batches. |
| **`update(...)`** | Mutate state from one batch of inputs (no return value). |
| **`compute()`** | Pure function from state → final metric value. |
| **`reset()`** | Restore state back to its declared default. |

> Source: `src/torchmetrics/metric.py`. The whole base class is ~1300 lines, but ~80 % of it is plumbing around these four ideas.

---

## Why split state from compute?

Many ML metrics are **non-decomposable** — you cannot average per-batch values to get the correct global value.

Concrete example. Suppose batch 1 has 90 correct out of 100 (acc = 0.90) and batch 2 has 1 correct out of 5 (acc = 0.20).

- "Average of batch accuracies" = (0.90 + 0.20) / 2 = **0.55** ❌
- "Correct global accuracy"     = 91 / 105            = **0.867** ✅

The fix is the same one used in databases and MapReduce: don't aggregate the *output*, aggregate the *sufficient statistics*. Here, that's `correct` and `total`. Across-batch (and across-device) reduction of those is just `sum`. Then `compute()` does the final division.

This is exactly what TorchMetrics does for every metric — F1 keeps `(tp, fp, fn)`, AUROC keeps `(preds, target)` lists, MSE keeps `(sum_squared_error, n)`, and so on.

---

## The lifecycle

```text
                            ┌────────────────────────┐
   __init__()  ──> _defaults │  add_state("tp", ...)  │
                            │  add_state("fp", ...)   │
                            └────────────────────────┘
                                     │
                                     ▼
       ┌──────────────┐   for each batch
       │   update()    │ ◄──────────────────────────┐
       └──────────────┘                              │
                                     │               │
                                     ▼               │
                            ┌────────────────────┐    │
                            │  compute()         │    │
                            │  (optional: sync)  │    │
                            └────────────────────┘    │
                                     │               │
                                     ▼               │
                            ┌────────────────────┐    │
                            │  reset()            │ ──┘
                            └────────────────────┘
```

- `__init__` registers state via `add_state`. The defaults are `deepcopy`'d so `reset()` can restore them later.
- `update(*args, **kwargs)` is wrapped by the base class to (a) bump an `_update_count`, (b) clear the cached `_computed` value, and (c) optionally enable autograd if you call the metric as `forward(...)`.
- `compute()` is wrapped to (a) sync state across processes if running under DDP, (b) cache the result so subsequent `compute()` calls are free, (c) restore unsynced state when you exit the sync context.
- `reset()` overwrites state with the deep-copied defaults — also crucial because we want the storage *device* and *dtype* to match where the metric currently lives.

---

## Two reduction modes

Look at the comment block in `metric.py` around `forward()` and you'll see the library distinguishes two modes:

### `_forward_full_state_update` (the safe default for some metrics)

For each batch, the base class:

1. Calls `update(...)` to add this batch into global state.
2. Saves the global state.
3. Calls `reset()`, then `update(...)` again with just this batch.
4. Calls `compute()` → that's the *batch value* returned to you.
5. Restores the saved global state.

This is the most general path. Used when `full_state_update=True` (or unset). Safe for any metric, but does ~2× the work per call.

### `_forward_reduce_state_update` (the fast path)

When you set `full_state_update=False` on your metric class, you're declaring "my batch state can be merged into the global state by the same reduction function I use across processes." The base class can then:

1. Snapshot global state, reset to default.
2. `update(...)` on this batch only → call `compute()` for the batch value.
3. Re-merge the saved global state into the new (batch-only) state using the registered `dist_reduce_fx`.

That avoids the second `update()` call. F1 / Accuracy / Precision all set `full_state_update=False` because tp/fp/fn just *sum*.

> **Heuristic**: if your metric's state reduces with `sum` / `mean` / `cat`, you can set `full_state_update=False` for a 2× `forward()` speedup. If it requires custom logic at update time (rare — e.g. a streaming median), leave it `True`.

---

## State types — tensor vs. list

`add_state` accepts two shapes of default:

| Default | Behavior | DDP reduction |
|---|---|---|
| `torch.tensor(0.0)` (or any tensor) | Mutated in place / overwritten in `update`. | `sum`, `mean`, `min`, `max`, `cat`, or any callable. |
| `[]` (empty list) | Append tensors during `update`. Concatenated across processes. | Almost always `cat`. |

**Rule of thumb**: prefer tensor states. Use list states only when the metric mathematically *requires* the full prediction/target arrays (AUROC, mAP, BLEU, NDCG). List states are expensive — they hold the whole epoch's data in memory and `all_gather` it across the cluster.

> Pro tip: pass `compute_on_cpu=True` to a metric that uses list states. The base class will move list elements to host memory after each `update`, which can save a *lot* of GPU RAM at the cost of a small PCIe transfer.

---

## What the base class gives you for free

If you write a tiny custom metric, the parent `Metric` class automatically:

1. Registers state on the right device when you `.to(...)`.
2. Concatenates / sums / averages state across DDP ranks via `_sync_dist`.
3. Caches the `compute()` result so calling it twice in a row doesn't recompute (`compute_with_cache=True`).
4. Validates types — passing `compute_on_cpu="yes"` raises a clear `ValueError` instead of silently breaking later.
5. Rewrites cryptic device-mismatch errors with a hint to call `.to(device)`.
6. Exposes `metric.metric_state` so you can inspect raw state for debugging.
7. Lets you `merge_state(other_metric)` if state is reducible.

---

## Performance flags every senior engineer should know

Inherited keyword arguments accepted by every metric:

| Flag | Default | What it controls |
|---|---|---|
| `compute_on_cpu` | `False` | Move list states to CPU after each update — saves GPU RAM. |
| `dist_sync_on_step` | `False` | If `True`, sync states inside `forward()` every batch. **Almost always leave `False`**: per-step sync is a global barrier and tanks throughput. |
| `process_group` | `None` (= world) | Restrict sync to a subgroup. |
| `dist_sync_fn` | `None` | Replace the default `all_gather` with your own (e.g. for non-NCCL backends or testing). |
| `sync_on_compute` | `True` | If `False`, `compute()` returns local rank's value only — useful for per-rank diagnostics. |
| `compute_with_cache` | `True` | If `False`, every `compute()` call recomputes. Turn off when the metric depends on state that mutates between updates. |

These knobs are surfaced because real-world setups often need them — e.g. on huge eval sets you want `compute_on_cpu=True` and `sync_on_compute=True` (default), while during noisy debug runs you might want `sync_on_compute=False` to see per-rank values.

---

## `MetricCollection` and **compute groups**

`MetricCollection` lets you bundle metrics. Its killer feature is **compute groups**: metrics whose internal states are the same can share state across the collection — so you compute the confusion matrix *once* and derive Accuracy, Precision, Recall, F1 all from it.

The collection introspects each metric's `update` signature and runtime state shape. Metrics that match get bucketed into a single compute group; only the bucket's representative actually runs `update()`, and the rest read from it. You can disable this with `compute_groups=False` if you want strict isolation (e.g. for debugging).

This optimization can be **3–10×** faster on collections of stat-score-derived metrics.

---

## Summary: what to remember

1. State, not values, is what gets aggregated.
2. `update` mutates state; `compute` is pure; `reset` restores defaults.
3. Tensor state is cheap; list state is expensive but sometimes necessary.
4. `forward()` does **both** an update and a per-batch compute.
5. Set `full_state_update=False` whenever your state is summable for a free speedup.
6. `MetricCollection` + compute groups can dramatically reduce wasted work.

Move on to [Metric Class Internals](./metric-class-internals.md) for the line-by-line view.
