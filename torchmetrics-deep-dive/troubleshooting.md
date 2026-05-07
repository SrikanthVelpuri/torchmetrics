---
title: Troubleshooting
nav_order: 17
---

# Troubleshooting

A flat list of bugs you will hit and how to fix them. Sorted by symptom.

---

## "Expected all tensors to be on the same device, but found at least two devices"

**Cause.** Metric state is on CPU; `preds` / `target` are on GPU (or vice versa).

**Fix.** Move the metric: `metric = metric.to(preds.device)` (or in Lightning, register it as a module attribute via `self.<name> = metric` so the framework moves it for you).

The library actually rewrites this error message to point you at this exact fix — see `_wrap_update` in `metric.py`.

---

## "compute() returned a number; later I called update() and compute() returned the *same* number"

**Cause.** `compute_with_cache=True` (default) and you didn't call `update()` between the two `compute()` calls — or you did, but the wrapper failed to clear the cache because the call raised.

**Fix.** Make sure `update()` actually ran. If you really need to bypass the cache, pass `compute_with_cache=False` at construction.

---

## My metric is silently wrong only on multi-GPU (DDP)

**Almost always one of three things.**

1. State is a Python list of floats / scalars instead of `Tensor` / list-of-`Tensor`. Won't all_gather.
2. `dist_reduce_fx` is missing or wrong. A `cat` where you wanted `sum` doubles the state.
3. You called `metric.compute()` on rank 0 only and other ranks didn't enter. Causes hangs, not wrong values — but if you bail with a timeout, you'll see truncated state.

**Diagnostic.** Run a tiny `world_size=2` test where each rank has half the data. The DDP value must equal the single-process value over the full data.

---

## `compute()` raises "The Metric has already been synced"

**Cause.** You called `metric.sync(...)` manually and never `metric.unsync()`. The next `compute()` tried to sync again and refused.

**Fix.** Always pair `sync()` / `unsync()`, or use the `with metric.sync_context(...)` form. Inside Lightning, never call `sync()` yourself.

---

## "RuntimeError: One of the differentiated Tensors does not require grad"

**Cause.** You're trying to backprop through a metric whose `is_differentiable = False` (e.g. AUROC, F1). Or you're inside a `torch.no_grad()` context.

**Fix.** Don't use a non-differentiable metric as a loss. Use a differentiable surrogate (e.g. soft F1, or a logistic loss).

---

## Memory blows up during validation

**Cause.** Metric uses list state (AUROC / mAP / BLEU / NDCG). It holds all preds + targets until `compute()`.

**Fix.** Pass `compute_on_cpu=True` at construction. Consider chunking eval into shards and merging via `merge_state`.

---

## My metric value is wrong by ~50 % (suspiciously close to a half)

**Cause.** Common `MetricCollection` mistake: same metric instance used in both training and validation. Each phase `update`s, so `compute()` reflects both phases.

**Fix.** Use `metrics.clone(prefix="train/")` and `metrics.clone(prefix="val/")` to make independent copies.

---

## Lightning logs `acc` as a scalar but it doesn't update each epoch

**Cause.** You did `self.log("acc", metric.compute())` instead of `self.log("acc", metric)`. Lightning never sees the `Metric`, so it never resets.

**Fix.** Pass the metric instance directly: `self.log("acc", self.acc, on_epoch=True)`.

---

## `compute()` warns "compute called before update"

**Cause.** `_update_count == 0`. Either you really didn't update, or something raised inside `update()` before incrementing.

**Fix.** Verify `update()` runs. Check exception handlers around it — some catch silently.

---

## My custom metric works on one rank but DDP run hangs

**Cause.** Likely `update()` raised on one rank. That rank stops calling subsequent collectives. The others wait at the next `all_gather` forever.

**Fix.** Reproduce on `world_size=2` with `NCCL_DEBUG=INFO`. Wrap your `update()` in a try/except that re-raises after logging — *every* rank must reach `compute()` or none.

---

## Per-class metrics return NaN for some classes

**Cause.** Class has zero support in either `preds` or `target`. The denominator (TP+FN or TP+FP) is zero.

**Fix.** Most metrics expose a `zero_division=` knob; the new API uses `nan_handler` or returns NaN by design. Decide what NaN should mean: "this class wasn't present, ignore" → handle in the consumer, not by silently coercing to 0/1.

---

## FID values are wildly inconsistent across machines

**Cause.** Different Inception V3 weights, different image preprocessing, different sample counts.

**Fix.** Pin TorchMetrics version; pin sample count; pin image preprocessing; cache real-dataset statistics.

---

## "RuntimeError: Tensors must be CUDA and dense" inside `_sync_dist`

**Cause.** Some state lives on CPU because you called `.cpu()` on it manually, or the metric was never moved to GPU.

**Fix.** Always `.to(device)` *after* construction; never set state attributes manually — use `add_state(...)`.

---

## `torch.compile(metric)` works but compiles slowly / fails on first call

**Known.** `_wrap_update` and `_wrap_compute` introduce dynamic Python branches that don't compile cleanly. Compile the *user model* instead, not the metric. Recent PRs have improved this — check the current TorchMetrics release notes if compile is required.

---

## My epoch metric value drifts every epoch despite calling `reset()`

**Cause.** You're holding a reference to a list state and modifying it after `reset()` overwrote `self.<name>`. Reset doesn't truncate the old list — it replaces the attribute with a fresh empty list.

**Fix.** Don't keep references to mutable state across reset boundaries. If you need a copy, `copy.deepcopy(metric.metric_state)` before `reset()`.

---

## "TypeError: forward() got an unexpected keyword argument 'reduce_fx'"

**Cause.** Lightning's `self.log` is forwarding aggregation kwargs through to the metric. Older Lightning + newer TorchMetrics combos can clash.

**Fix.** Don't pass `reduce_fx` when logging a `Metric` (it doesn't apply). Update both libraries to current versions.

---

## Multiclass with `average="micro"` returns the same value as Accuracy

**That's correct.** Micro-averaged precision == micro-averaged recall == micro-averaged F1 == accuracy when every sample has exactly one true label. If you want a different number, use `"macro"` or `"weighted"`.

---

## `BootStrapper(metric).compute()` returns a single tensor, not a tuple

**Cause.** You didn't pass `quantile=`. With no quantiles requested, it just returns the mean across resamples.

**Fix.** Pass `quantile=torch.tensor([0.025, 0.975])` (or whatever interval you want) to get `(mean, lower, upper)`.

---

## Quick triage table

| Symptom | First place to look |
|---|---|
| Wrong number, single GPU | `update` math, then `compute` math. |
| Wrong number, multi-GPU only | `dist_reduce_fx` for every state. |
| Hangs in DDP | Different ranks took different paths in `update`. |
| OOM in eval | List-state metric → `compute_on_cpu=True`. |
| Nondeterministic | Unset seed; `BootStrapper` without `generator=`. |
| Lightning `self.log` weirdness | You probably passed `metric.compute()` instead of `metric`. |
| Drifting values across epochs | Forgot `reset()` or kept references to old state. |
