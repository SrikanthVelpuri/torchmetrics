---
title: Distributed Training
nav_order: 9
---

# Distributed Training

This page is about what *actually* happens when you call `metric.compute()` on rank 0 of an 8-GPU DDP job. Understanding it is the difference between a trustable benchmark and a silently wrong one.

---

## The problem (in one example)

You're computing F1 on 8 GPUs. Each rank sees 1/8 of the validation set. Each rank can compute its own per-shard F1 — but those numbers can't be averaged.

| Rank | TP | FP | FN | local F1 |
|---|---|---|---|---|
| 0 | 100 | 5  | 5  | 0.952 |
| 1 |  10 | 0  | 0  | 1.000 |
| 2 |  50 | 50 | 50 | 0.500 |
| ... | ... | ... | ... | ... |

Mean of local F1 ≠ global F1. You need to sum TP, FP, FN *first*, then compute. That's what TorchMetrics does, transparently.

---

## How `_sync_dist` works

Every state declared with `add_state(name, default, dist_reduce_fx=...)` participates in DDP synchronization:

1. On the rank entering `compute()` (which is every rank — DDP keeps them in lockstep), the wrapper calls `sync()`.
2. `sync()` saves a copy of local state into `self._cache`, then calls `_sync_dist`.
3. `_sync_dist`:
   - For each state attribute, pre-concatenates list states locally (one big tensor per rank).
   - Calls `gather_all_tensors(tensor, group=process_group)` — wraps `dist.all_gather` and pads tensors to a common shape if needed.
   - Applies the registered reduction (`sum`, `mean`, `cat`, `min`, `max`, or callable) on the result.
4. `compute()` runs on the merged state.
5. On exit, `unsync()` restores the local state from cache. Without this, calling `compute()` would corrupt subsequent `update()` calls.

Read this in `metric.py`:
- `Metric.sync` / `Metric.unsync`
- `Metric._sync_dist`
- `Metric._wrap_compute`
- `torchmetrics.utilities.distributed.gather_all_tensors`

---

## Padding for ragged tensors

`gather_all_tensors` handles the common case where each rank has a *different size* tensor (e.g. uneven last batch). It:

1. `all_gather`s shape information first.
2. Pads each rank's tensor to the max shape with zeros.
3. `all_gather`s the padded tensors.
4. Slices each rank's contribution back to its real shape.

If you replace `dist_sync_fn` with your own, your function must handle this case (or you must guarantee all ranks have identical-shape state).

---

## When does sync happen?

You usually have three knobs:

| Setting | Sync happens when |
|---|---|
| Default | At the beginning of every `compute()` call. Local restore on exit. |
| `dist_sync_on_step=True` | At every `forward()` call too (i.e. every batch). **Slow.** |
| `sync_on_compute=False` | Never; `compute()` returns *local* (per-rank) value. |

The default is correct for almost every workflow:

- During training, you log batch-level metrics with `forward()` — no sync, no global barrier.
- At validation epoch end, you call `compute()` — single global all_gather, single answer.

`dist_sync_on_step=True` is a footgun. Reach for it only when you're truly trying to compute a fully-synced batch number every step (e.g. some research papers report it). Even then, prefer logging the local value and reducing it via your training framework (Lightning's `sync_dist=True`).

---

## DDP-correctness checklist

A metric is **DDP-correct** if and only if:

1. Every state variable has a sensible `dist_reduce_fx`.
2. Reductions are *associative and commutative* (so the order ranks merge in doesn't matter).
3. `compute()` is a pure function of the merged state (no rank-specific globals).

Violations:

- Storing a Python list of Python floats: works locally, breaks under DDP because `gather_all_tensors` won't catch it.
- Calling `torch.distributed` directly inside `update()`: deadlocks if some ranks have empty inputs.
- Mutating `self._defaults`: breaks `reset()` invariants.

---

## Common DDP bugs and what they mean

| Symptom | Cause | Fix |
|---|---|---|
| `compute()` hangs forever on N-1 ranks | One rank threw inside `update()` and stopped participating. | Wrap eval in try/except *before* the metric or fix the bug; never silently skip update on one rank only. |
| Final number changes between runs | Reduction is non-associative (e.g. you wrote a custom mean of means). | Reduce sufficient statistics, not derived values. |
| OOM during `compute()` | List-state `cat` ships the whole eval set to every rank. | Use `compute_on_cpu=True`; consider `BinningMetric` patterns; set `sync_on_compute=True` and read the result on rank 0 only. |
| Different number per rank | You forgot `sync_on_compute=True` (or set it to False) and printed from each rank. | This is intentional with `sync_on_compute=False` — print only on rank 0. |
| Wrong number with `MetricCollection` and DDP | Rare: an unsynchronized custom metric inside the collection. | Audit each metric's `dist_reduce_fx`. |

---

## Multi-node specifics

Nothing special — TorchMetrics relies on `torch.distributed`, so it works over any backend (`nccl`, `gloo`, `mpi`). NCCL is fastest on GPU; gloo is needed if you sync on CPU tensors.

Subgroups (`process_group=...`) work too. Useful when you want different sets of metrics on different model replicas (e.g. expert-parallel models).

---

## FSDP / sharded data parallel

TorchMetrics is agnostic to the *model* parallelism strategy. As long as your `update()` is called once per (preds, target) pair on each rank, and as long as `torch.distributed` is initialized, sync works the same way.

For tensor-parallel cases where the same sample is "split" across ranks (e.g. you pass logits-shard, target-shard), reduce on the model side first to get a complete `preds`/`target` per rank, then call `update()`.

---

## A defensive pattern for production

```python
metric = AUROC(task="binary", compute_on_cpu=True)

with torch.no_grad():
    for batch in loader:
        metric.update(model(batch.x), batch.y)

# Optional: explicit sync so we can log on every rank if we want.
metric.sync()
try:
    if local_rank == 0:
        log(metric.compute())
finally:
    metric.unsync()
```

This is more verbose than the default, but it makes the sync boundary explicit, which helps when debugging exotic distributed configs.
