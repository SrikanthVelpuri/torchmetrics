---
title: Interview Questions
nav_order: 14
---

# Interview Questions

A curated list of questions an ML-engineering interviewer might ask about TorchMetrics specifically — and the metric / evaluation engineering they probe more broadly. Each entry has a model answer; the **next page**, [Follow-Up Questions]({{ "./follow-up-questions.md" | relative_url }}), drills further.

> Use these as a self-test. If you can answer 18/25 confidently, you are well above the bar.

---

### Q1. What is TorchMetrics and why does it exist?

A PyTorch-native library of 100+ rigorously-tested ML metrics with a small base class (`Metric`) for building custom ones. It exists because (a) metrics are subtle (non-decomposable, distributed-unsafe when handwritten); (b) it was originally inside PyTorch Lightning and was extracted so plain-PyTorch users could benefit; (c) every research codebase used to ship a slightly-wrong copy of `f1()`, and consolidating fixed that.

---

### Q2. Why can't we just average per-batch accuracy across batches?

Because the metric is non-decomposable when batches differ in size or class balance. Averaging batch values weighs every batch equally regardless of its sample count. The correct aggregation is to sum *sufficient statistics* (TP/FP/FN/TN, or correct/total) across batches and compute once. F1, AUROC, mAP, BLEU are even more obviously non-decomposable.

---

### Q3. Walk me through what happens when I call `metric(preds, target)`.

It's a `forward()` call. `forward()` chooses one of two paths:

- `_forward_full_state_update`: update global state, snapshot, reset, update with batch only, compute (this is the per-batch return value), restore snapshot.
- `_forward_reduce_state_update`: snapshot global state, reset, update with batch only, compute, then merge the snapshot back via the registered `dist_reduce_fx`.

The fast path requires `full_state_update=False` and works when state reduces by sum/mean/min/max/cat. Both paths leave you with one batch return value and a correctly-accumulated global state.

---

### Q4. Functional vs. modular API — when do you use each?

Functional for one-shot computation on a fixed `(preds, target)` pair. Modular when you need to accumulate across batches, run under DDP, or use Lightning's `self.log` integration. Modular metrics handle device placement, sync, caching, and reset; functional ones don't.

---

### Q5. How does TorchMetrics handle DDP?

Each metric declares state with `add_state(name, default, dist_reduce_fx)`. On `compute()`, the wrapper calls `_sync_dist`, which pre-concatenates list states locally, `all_gather`s every state across ranks, and applies the registered reduction (`sum`, `mean`, `cat`, `min`, `max`, or callable). After compute, local state is restored from a cache so subsequent updates aren't corrupted.

---

### Q6. What's the difference between `dist_sync_on_step=True` and the default?

`dist_sync_on_step=True` forces a global all_gather every batch (inside `forward`). It tanks throughput because every step becomes a global barrier. The default syncs only inside `compute()` (typically once per epoch), which is correct for almost every workload.

---

### Q7. What's a "compute group" in `MetricCollection` and why does it matter?

When multiple metrics share the same internal state (e.g. all classification metrics derived from `StatScores`), `MetricCollection` detects it and runs `update()` once for the whole group, then derives each metric in `compute()`. It can give a 3–10× speedup for large collections. You can disable it via `compute_groups=False` if you want strict isolation.

---

### Q8. Why does `add_state` distinguish tensor states from list states?

Tensor states are O(1) memory and reduce by sum/mean/min/max — fast and DDP-friendly. List states keep the entire prediction/target population (needed for AUROC, mAP, NDCG, BLEU) and reduce by `cat` — they grow with eval-set size and require care under DDP. Some metrics offer `compute_on_cpu=True` to keep list state off the GPU.

---

### Q9. How are list states made DDP-correct?

`_sync_dist` first concatenates each rank's list locally to one tensor, then `all_gather`s. It also handles edge cases (empty rank, ragged shapes) by emitting an empty tensor of the right device/dtype and by padding.

---

### Q10. What's the lifecycle of a metric across an epoch?

`__init__` registers state. For each batch, either `update(...)` (no return) or `forward(...)` (which both updates and returns a per-batch value). At epoch end, `compute()` syncs across DDP and produces the global value. `reset()` restores defaults, ready for the next epoch.

---

### Q11. Why must I call `reset()` between epochs?

Because state persists across `compute()` calls (compute doesn't reset). If you skip reset, your epoch-2 number includes epoch-1 data. PyTorch Lightning calls `reset()` for you at the right hooks; in raw PyTorch, you do it yourself.

---

### Q12. What's `compute_on_cpu` and when do you use it?

After every `update`, list states are moved to CPU memory. Useful when the eval set is huge and the predictions don't fit on GPU. The cost is host-device PCIe traffic per update. Recommended for AUROC/mAP/BLEU on million-sample eval sets.

---

### Q13. How do I make a custom metric DDP-correct?

(1) Inherit `Metric`. (2) Declare every persistent value via `add_state` with an appropriate `dist_reduce_fx`. (3) Keep state as tensors (or lists of tensors). (4) Don't call `torch.distributed` yourself. (5) Set `full_state_update=False` if your state reduces by sum/mean/cat/min/max — you get a free 2× speedup on `forward`.

---

### Q14. What is `MetricTracker` and why use it?

It's a wrapper that keeps a history of `compute()` results across epochs and exposes "best so far" / "best epoch" using the metric's `higher_is_better` flag. It removes the need for handwritten "best val acc" bookkeeping and works with any `Metric`.

---

### Q15. How does `BootStrapper` work?

It maintains *N* internal copies of the wrapped metric, each getting a Poisson-bootstrapped subsample of every batch. At `compute()`, it returns mean and quantiles across the N replicas — the standard non-parametric CI for a metric.

---

### Q16. How does TorchMetrics integrate with Lightning's `self.log`?

`self.log("acc", self.train_acc, on_step=True, on_epoch=True)` accepts a `Metric` instance. Lightning internally calls `forward` for the step value and `compute` (then `reset`) for the epoch value, all DDP-aware. You should *not* pass `metric.compute()` — that eagerly computes and short-circuits the lifecycle.

---

### Q17. Why is logging `metric.compute()` directly (instead of the metric) bad?

Because Lightning won't manage the lifecycle. You'll either log a stale cached value (because of `compute_with_cache`), forget to reset, or accidentally double-update. Pass the metric module; Lightning knows what to do with it.

---

### Q18. What's the right metric for an imbalanced binary classification problem?

Don't rely on accuracy. Look at AUROC for a threshold-free view, F1 / Precision / Recall at the operating point you care about, and Average Precision (PR-AUC) when positives are rare. Add a calibration metric (ECE) if your output is a probability that's consumed downstream.

---

### Q19. AUROC vs. Average Precision — when to prefer which?

AUROC integrates over FPR; AP integrates over recall. On heavily imbalanced data, AUROC can stay high even when the classifier ranks negatives ahead of positives in absolute terms. AP is the more honest signal for rare-positive problems (search, fraud, anomaly).

---

### Q20. How does TorchMetrics handle multilabel vs. multiclass classification?

Multiclass: one ground-truth class per sample; `argmax` over the K class dim. Multilabel: K independent binary tasks per sample, each thresholded separately. Most classification metrics in the new API are explicitly task-prefixed (`BinaryX`, `MulticlassX`, `MultilabelX`); the legacy `X(task=...)` wrapper picks one for you.

---

### Q21. What is NDCG and why does it need `indexes=`?

NDCG is the normalized version of DCG, the position-discounted gain summed over a *ranked list per query*. The `indexes` tensor tags which query each item belongs to so the metric can group, sort within a group, then average across groups.

---

### Q22. How do you compute confidence intervals on F1?

Wrap with `BootStrapper(F1Score(...), num_bootstraps=1000, quantile=torch.tensor([0.025, 0.975]))`. For a paired comparison of two models, prefer a paired bootstrap (resample sample indices, not predictions) to keep the comparison statistically valid.

---

### Q23. How do you monitor metrics in production?

Use `Running(metric, window=...)` for per-segment rolling windows; emit `compute()` to your time-series system on a fixed cadence (e.g. every 30 s). Complement scalar values with per-segment breakdowns (region, device, model version) — aggregate metrics hide the failures that matter.

---

### Q24. What is `merge_state` for?

It merges another metric's state (or a state dict) into the current metric, using the registered reductions. This lets you persist metric states from different machines / pipelines and reduce them later — useful for offline distributed evaluation.

---

### Q25. Suppose you build a metric that doesn't sync correctly under DDP. What's the minimal failing test?

A two-rank DDP test where each rank `update`s a disjoint shard of the data, calls `compute()`, and asserts equality with the single-process result on the full data. This will fail if any state isn't all_gather'd or if the reduction is wrong.

---

Continue to [Follow-Up Questions]({{ "./follow-up-questions.md" | relative_url }}) for the harder drill-downs.
