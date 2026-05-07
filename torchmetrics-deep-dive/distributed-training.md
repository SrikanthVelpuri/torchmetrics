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

---

## Interview Drill-Down (multi-level follow-ups)

### Q1. Walk through, line-by-line, what `compute()` does on rank 3 of an 8-GPU DDP job.

> 1. The wrapper checks `_update_count > 0` (warns if not). 2. Checks `compute_with_cache` — if cached, returns immediately. 3. Calls `sync()`: copies current state to `_cache`, marks `_is_synced=True`, then `_sync_dist`. 4. `_sync_dist` pre-concats list states locally, then `gather_all_tensors` (`all_gather` with ragged-shape padding) across the world. 5. Applies the registered reduction (`sum`/`cat`/etc.). 6. Runs the user's `compute()` on the merged state. 7. Calls `unsync()`: state is restored from `_cache`. 8. Caches and returns.

  **F1.** What if rank 3 has fewer batches than the others?

  > Two cases. (a) Tensor states with sum reduction — fine, rank 3 contributes its smaller value. (b) List states — `_sync_dist` adds an empty `torch.tensor([], device=..., dtype=...)` to make `all_gather` shape-compatible.

    **F1.1.** What if rank 3 actually had **zero** updates (e.g. dataloader exhausted)?

    > `_update_count` is 0 → warning fires, but `compute()` still runs. The synced state from other ranks dominates. The metric returns a value, not an error. Whether that's correct depends on the metric: for an `Accuracy` (sum-reduced), it's correct. For `FrechetInceptionDistance` (which divides by sample counts), zero contribution from rank 3 is fine because `n=0` on that rank.

      **F1.1.1.** What if rank 3 throws inside `update()`?

      > Now you have a real problem. Rank 3 stops calling collectives; rank 0-2 + 4-7 hang at `_sync_dist`'s `all_gather`. NCCL eventually times out (default 30 min). Fix: wrap eval in a try/except that re-raises, but make the eval-loop transactional so all ranks see the same exception window.

### Q2. Why does TorchMetrics keep a local cache of state during sync?

> So that calling `compute()` is non-destructive. After `_sync_dist` mutates state to the merged version, `compute()` reads it, then `unsync()` restores it from cache. This means subsequent `update()` calls keep accumulating local state correctly, and you can call `compute()` multiple times without re-syncing.

  **F1.** Does this mean each rank has the *same* local state after compute() returns?

  > No — each rank has the same *cached* local state restored, which is its **per-rank-only** state, not the merged one. Crucial subtlety: after `compute()`, ranks have *different* state (each rank's own contribution); only the value returned by `compute()` was synced.

    **F1.1.** What if I want every rank to keep the merged state after compute?

    > Pass `should_unsync=False` to `sync()` (manual API), or set `_should_unsync=False`. Now every rank has the global state — useful if you want to do further compute steps that need it. Downside: `update()` after that point double-counts because it adds to already-merged state.

      **F1.1.1.** Concrete situation where `should_unsync=False` is right?

      > Computing a derived metric whose input is the synced metric itself — e.g. computing class-balanced accuracy from a globally-merged confusion matrix in a custom downstream step. You take the merged state, run derived math, then `reset()` before the next eval.

### Q3. Why is `dist_sync_on_step=True` almost always wrong?

> It triggers a global `all_gather` inside `forward()`. Every batch becomes a global barrier — the slowest rank gates the whole world. Throughput tanks; gradient sync (which DDP also does) compounds the cost.

  **F1.** When *is* it right?

  > Only when the per-step value must be globally consistent — e.g. a custom learning-rate schedule that reads a synced metric per step. Even then, prefer logging local values and reducing them via your training framework's `sync_dist` for scalars.

    **F1.1.** What's the alternative when you really want a step-level synced number?

    > Maintain two metric instances. Use the un-synced one for fast `forward()` per-step logs (local rank, no barrier); call `compute()` on the synced one only at lower frequency (every N steps). Two metrics, two log streams, no global barrier per step.

      **F1.1.1.** Doesn't that double the GPU memory?

      > Yes, by the size of the metric state. For tensor-state metrics that's negligible. For list-state metrics it can matter — push the per-step (un-synced) one to `compute_on_cpu=True` and keep only the periodic synced metric on GPU.

### Q4. Your DDP run produces different metric values across two identical runs. Why?

> A few candidates:
> - **Floating-point non-associativity**: `(a + b) + c ≠ a + (b + c)` at machine precision. `all_gather` order can vary across runs.
> - **NCCL determinism**: `nccl` reductions can be non-deterministic by default; set `CUDA_LAUNCH_BLOCKING=1` and `TORCH_DETERMINISTIC=1` for repro.
> - **Data sharding**: if your sampler isn't seeded the same way, different ranks see different data.

  **F1.** Which is most common?

  > Data sharding 90 % of the time. Sampler seeding is the silent killer.

    **F1.1.** How do you fix sampler seeding correctly?

    > Use `DistributedSampler(dataset, seed=...)` and call `sampler.set_epoch(epoch)` every epoch. Forgetting `set_epoch` means rank-0 sees the same shard every epoch — silent train-set leak.

      **F1.1.1.** Could that bias your metric values?

      > Yes — eval metrics over a small fixed shard are higher-variance and biased toward whatever's in that shard. The fix is to also seed the val sampler deterministically and ensure each rank sees a different shard.

### Q5. How does `_sync_dist` handle list states with very different sizes per rank?

> `gather_all_tensors` first does an `all_gather` of *shapes*, then pads each rank's tensor to the max shape with zeros, then `all_gather`s the padded tensors, then slices each contribution back to its real shape.

  **F1.** Doesn't padding waste memory and bandwidth?

  > Yes, proportional to the variance in shape. For most metrics this is negligible (per-batch shape variance is small). For pathological cases (one rank with 10× the data of others), it costs you. Mitigation: better data sharding so per-rank counts are roughly equal.

    **F1.1.** What if you can't balance shards?

    > Use a metric that has tensor state (sum/mean/min/max reductions) instead of list state. The padding problem disappears because tensor states are fixed-shape.

      **F1.1.1.** What if the metric is canonically list-state (AUROC, mAP)?

      > Switch to a binned variant — bucket predictions into K bins; track `(positives_per_bin, count_per_bin)`. AUROC becomes a sum over bins. Tensor state, no padding, slightly approximate. Trade-off worth it for very large evals.
