---
title: Follow-Up Questions
nav_order: 15
---

# Follow-Up Questions

The questions interviewers ask *after* the surface answer, to find out whether you've actually used the library at scale.

---

### F1. "OK, but show me the line in `metric.py` where states are aggregated across DDP ranks."

Open `Metric._sync_dist`. Two important moves:

```python
# 1. Local pre-concat for list states — one all_gather per state, not one per chunk.
for attr, fn in self._reductions.items():
    if fn == dim_zero_cat and isinstance(input_dict[attr], list) and len(input_dict[attr]) > 1:
        input_dict[attr] = [dim_zero_cat(input_dict[attr])]

# 2. The actual cross-rank gather.
output_dict = apply_to_collection(input_dict, Tensor, dist_sync_fn,
                                  group=process_group or self.process_group)
```

The `apply_to_collection` walks the dict, finds tensors, and calls `gather_all_tensors` (Lightning's `all_gather` wrapper that handles ragged shapes).

---

### F2. "Why is calling `compute()` cheap on the second call?"

The wrapper around `compute` caches the last result in `self._computed`. The cache is invalidated whenever `update()` runs (the wrapper sets `self._computed = None`). You can disable this with `compute_with_cache=False` if your metric's compute depends on something other than its declared state.

---

### F3. "Walk me through what `_forward_reduce_state_update` does in pseudo-code."

```text
saved = deepcopy(state)        # current global state
reset()                         # state ← defaults
update(batch)                   # state ← batch only
batch_value = compute()         # value for this batch
state ← merge(saved, state) using dist_reduce_fx
return batch_value
```

This is the "fast" forward path because it does only one update per call. It's only safe when `_reduce_states` knows how to merge — i.e. when every state's reduction is one of the standard ones.

---

### F4. "What's the difference between `sync_on_compute` and `dist_sync_on_step`?"

`sync_on_compute` (default `True`) controls whether `compute()` triggers all_gather. `dist_sync_on_step` (default `False`) controls whether `forward()` *also* triggers all_gather every step. The first is essentially mandatory for a global metric; the second is a global barrier per step and almost always wrong.

---

### F5. "Why are state defaults `deepcopy`'d in `add_state`?"

So `reset()` can restore them without you accidentally sharing state across instances. Without deepcopy, `default = torch.tensor(0)` would be a shared singleton — mutating one instance's state would leak to the next. Same reason `nn.Module.register_buffer` clones.

---

### F6. "Why are list states expensive in DDP, and what fixes it?"

They scale with eval-set size: every rank holds its own predictions until `compute()`, then `all_gather` ships everything everywhere. Two mitigations:

1. `compute_on_cpu=True` — keep list state on CPU between updates.
2. `sync_on_compute=True` (default) and read `compute()` only on rank 0 to avoid materializing the merged tensor on every rank — though TorchMetrics' default is to materialize on all ranks for correctness.

For very large evals, you may want a *sketch-based* metric (e.g. `BinningMetric`-style) rather than a list-state metric.

---

### F7. "If I write a custom metric and forget `dist_reduce_fx=...`, what happens?"

The `_reductions[name]` is `None`. In `_sync_dist`, the gathered tensor is stacked but no reduction is applied — your state is now a `(world_size, ...)`-shaped tensor. `compute()` will likely break or silently produce wrong numbers. Always set the reduction explicitly.

---

### F8. "How does `MetricCollection` decide two metrics are in the same compute group?"

It runs each metric's `update()` once at warm-up time, then introspects the resulting `metric_state` dict — same keys, same shapes/dtypes ⇒ same group. The first metric in each group is "the leader"; followers don't run their own `update`. You can opt out by passing `compute_groups=False` (or a list of explicit group names).

---

### F9. "What's the difference between `MultioutputWrapper` and `MultitaskWrapper`?"

`MultioutputWrapper`: same metric, multiple output dimensions (e.g. multi-output regression — N parallel MAE).

`MultitaskWrapper`: different metrics for different tasks (e.g. one F1 for fraud + one MAE for LTV in a multi-task model). Inputs are dicts keyed by task.

---

### F10. "What bug would you suspect if a metric value drifts every time you call `compute()` repeatedly without updating?"

Either `compute_with_cache=False` *and* a non-deterministic compute (rare), or — much more likely — `unsync()` failed, leaving merged DDP state in place; the next `compute` syncs again on already-synced state. The base class guards against the latter via `_is_synced` and raises `TorchMetricsUserError`. If you see drift, look for a `.sync()` you called manually and forgot to `unsync()`.

---

### F11. "In Lightning, what's the difference between `on_step=True` and `on_epoch=True` for a metric?"

`on_step=True` calls `metric.forward(...)` per batch and logs that value (per-batch state-aware metric value).

`on_epoch=True` calls `metric.compute()` at the appropriate epoch hook and logs that — then `reset()`s.

Setting both is fine; you'll get two separate series. For *validation*, prefer `on_step=False, on_epoch=True`.

---

### F12. "Why shouldn't I call `self.log("acc", metric.compute())`?"

Because `compute()` returns a tensor, and Lightning has no idea it came from a metric. It will treat it like any other scalar — caching it for the epoch buffer, possibly applying `sync_dist` on top of an already-synced value. You also lose the auto-reset Lightning would have triggered.

---

### F13. "What if my `update()` raises on a single rank?"

That rank stops calling subsequent collectives; the others hang at the next `all_gather`. Wrap eval in try/except above the metric layer, or fix the bug. Never silently skip `update()` on one rank — the world-size mismatch will deadlock at `compute()`.

---

### F14. "Why is `Metric.__init__` so heavy on validating kwargs?"

Misconfiguration here corrupts every later run silently. The validation block (`isinstance(self.compute_on_cpu, bool)` etc.) is intentionally loud and early — fail at `__init__`, not at the 99th batch of training.

---

### F15. "How does TorchMetrics handle half precision and `torch.compile`?"

Tensor states are auto-cast when `.to(dtype=...)` is called, the same as buffers. Most metrics work with `bfloat16` and `float16` because their math is well-conditioned. `torch.compile` compatibility is tested in CI for many metrics — caveats exist for metrics that branch on Python state (e.g. `compute_with_cache`).

---

### F16. "What does `merge_state` do, and what does it require?"

It folds another metric's (or dict's) state into the current metric using the registered reductions. Requires `full_state_update=False` (because the merge must use the same reductions as DDP would). Custom metrics that don't fit the pattern must override `merge_state` themselves.

---

### F17. "Why does `_sync_dist` add an empty tensor for ranks with no data?"

`all_gather` requires every rank to participate with a same-typed tensor. If a rank received no examples (uneven last batch), its list state is empty. We send `torch.tensor([], device=..., dtype=...)` so the dimension semantics are preserved and the reducer can still concatenate.

---

### F18. "What's the right way to compute a metric only on a subset of classes?"

Use `MulticlassF1Score(num_classes=K, average=None)` to get the per-class array, then index. Or wrap with `ClasswiseWrapper(metric, labels=[...])` for nicely-labeled output. Don't pre-filter inputs in `update()` — you'll throw off accumulation if the filter changes per batch.

---

### F19. "Why do retrieval metrics need an `indexes` argument and how is it used in `compute`?"

Because retrieval metrics evaluate per-query lists, not per-sample independently. `indexes` groups items by query. The metric stores all `(preds, target, indexes)` until `compute()`, then groups by index, sorts each group by predicted score, computes the per-query metric, and aggregates with the chosen `aggregation` function (mean by default).

---

### F20. "How would you debug a metric whose value depends on batch order?"

Three checks:

1. Reductions are summative? `sum`, `mean`, `cat`, `min`, `max` are order-invariant up to floating-point noise. Custom reductions might not be.
2. Tensor states only — never Python lists of floats.
3. Numerical noise — large floats summed in different orders give slightly different totals. Use Kahan summation or accumulate in `float64`.

---

### F21. "Why is `is_differentiable` an explicit class attribute?"

So users who want to use the metric *as a loss* know what to expect — and so the trainer can sanity-check (e.g. don't allow `.backward()` through a metric whose author marked it non-differentiable). Many "metrics" (BLEU, AUROC, mAP) are inherently non-differentiable; some (MAE, MSE, SI-SDR) are.

---

### F22. "What happens to a metric's state when you call `model.state_dict()`?"

By default, nothing — metric states are non-persistent. Pass `persistent=True` per state (or in `add_state(..., persistent=True)`) to include them in the model state dict. This is a deliberate choice: most users don't want to bloat checkpoints with throwaway eval state.

---

### F23. "If you saw a metric whose `compute()` dominates GPU memory, what would you change?"

Try in order: `compute_on_cpu=True` to push list state to host RAM; switch to a tensor-state algorithm if one exists (e.g. binning AUROC); reduce eval batch size; chunk eval into shards and use `merge_state` to combine.

---

### F24. "What's a metric that's not in TorchMetrics but probably should be, given your domain?"

(Bring your own answer here — interviewers love a candidate who knows the gaps in their tools.) Common examples: domain-specific KPIs in your industry, calibration on multilabel data, segmented fairness over continuous protected attributes, novelty/diversity metrics for recommenders.

---

### F25. "Can `Metric` be used outside of training, e.g. on a Spark cluster?"

Yes, via `merge_state`. Each Spark task instantiates a `Metric`, updates it on its shard, and returns `metric.metric_state` (a dict of tensors). The driver creates a fresh metric and folds in each shard's state with `merge_state`. The catch: every shard must use a metric whose reductions are summable / catable. Stateful metrics like running quantiles need custom merging.
