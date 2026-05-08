---
title: Wrappers — interview deep dive
---

# Wrappers — interview deep dive

[← Family page](./index.md) · [← Home](../index.md)

---

## Q1. "When do you bootstrap a metric vs run multiple seeds?"

**Answer.** Different sources of variance:
- **Bootstrap**: variance from the *test set sampling*. "If I drew a different test set, would the metric be different?" Answer with `BootStrapper(num_bootstraps=1000)`.
- **Multiple seeds**: variance from *training run*. "If I retrained with different init / data shuffles, would the model be different?" Answer with N retrained models.

For paper-grade reporting, do both: per-seed metric + bootstrap-CI per seed. For ship/no-ship comparisons against a baseline, bootstrap on the held-out set is usually sufficient.

> **F1.1** "How many bootstrap iterations are enough?"
>
> **Answer.** Variance of the bootstrap CI scales as `1/sqrt(B)`. 1000 is the standard default — gives roughly 2-decimal stability. For expensive metrics (FID), 100 with documented limitations is acceptable.

---

## Q2. "Why use `MetricTracker` instead of logging the metric every epoch?"

**Answer.** `MetricTracker` keeps a running history and exposes `best_metric()` directly. Without it: you log to TensorBoard, then write boilerplate to find the best epoch — which is fragile because higher/lower-better is metric-specific. `MetricTracker(maximize=True/False)` makes the direction explicit and integrates cleanly with checkpointing logic.

---

## Q3. "What's `compute_groups` in `MetricCollection` and when does it bite?"

**Answer.** Default `compute_groups=True` — TorchMetrics detects metrics with the same state schema and update method (e.g., Precision and Recall both compute `(TP, FP, FN, TN)` from the same inputs) and shares state. Saves compute. Bites when:
- Custom metrics fool the heuristic.
- Two metrics have *seemingly* identical updates but compute differently (e.g., one ignores a class).
- A metric in the collection has internal randomness — the shared compute may de-randomise.

When in doubt, set `compute_groups=False`. Trade compute for correctness.

---

## Q4. "MultioutputWrapper vs MultitaskWrapper — which when?"

**Answer.**
- **Multioutput**: same metric, multiple outputs (5-target regression all measured by MAE).
- **Multitask**: different metrics, different outputs (regression head with MAE + classification head with F1).

The distinction is whether the metrics are homogeneous or heterogeneous.

---

## Q5. "FeatureShare — explain the savings."

**Answer.** Generative model evaluation runs FID, KID, IS — all of which compute Inception features. Without FeatureShare, each metric runs Inception forward separately on the same images. With FeatureShare, one Inception pass, three metrics consume the same features.

Compute saving: `~3×` for the feature-heavy phase. For 50k images at Inception cost, this is hours.

---

## Q6. "Running window metric — when?"

**Answer.** Online production monitoring. The model is live; you want the *current* AUROC, not the all-time AUROC. `Running(AUROC, window=10000)` reports AUROC over the last 10k requests. Catches drift in real-time without resetting.

---

## Q7. "Bootstrap CI is non-overlapping. Significant?"

**Answer.** Non-overlapping bootstrap CIs ⇒ p < 0.05 (roughly, for symmetric distributions). For rigorous testing on paired data (same test set, two models), use a *paired* bootstrap: bootstrap the *difference* of metrics, check if 0 is in the CI. Stricter and more powerful than independent CIs.

---

## Q8. "Custom metric — should I write a class or use Abstract wrapper?"

**Answer.** Subclass `tm.Metric` directly for new state and update logic. Use `Abstract` wrapper to compose existing metrics without re-implementing them. Rule of thumb: if the math is novel, write a `Metric`; if you're combining or post-processing existing metrics, write a wrapper.

---

## Q9. "ClasswiseWrapper logs N values. How does this interact with `self.log` in Lightning?"

**Answer.** `ClasswiseWrapper` returns a dict with prefix-keyed entries (`acc/class_0: 0.84, acc/class_1: 0.92, ...`). Use `self.log_dict(...)` not `self.log(...)`. Each key becomes a separate row in the dashboard. Lightning auto-handles `on_step` / `on_epoch` aggregation per-key.

---

## Q10. "How do wrappers behave under DDP?"

**Answer.** They delegate to the wrapped metric. `BootStrapper` calls `update`/`compute` on the inner metric — DDP sync happens inside the inner metric, then the bootstrap aggregates already-synced values. So DDP correctness is the inner metric's responsibility; wrappers don't add new sync logic.

---

[← Back to family page](./index.md)
