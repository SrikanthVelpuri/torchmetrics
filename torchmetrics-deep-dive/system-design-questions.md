---
title: System Design Questions
nav_order: 16
---

# System Design Questions

These are open-ended questions where you're expected to design *a metrics system*, not just answer trivia. Each comes with a worked solution that uses TorchMetrics where it fits and explains the trade-offs explicitly.

---

## SD1. Design a real-time evaluation service for a recommender

**Brief**: Score a recommender's online behavior. Tens of thousands of QPS. Need MRR / NDCG / Recall@k over rolling windows, segmented by region and model version.

**High-level design**

```text
+------------+   impressions+clicks    +----------------+  metric updates  +-------------+
|  Backend   | ───────────────────────►|  Metric Stream ├─────────────────►|  Metrics    |
|  Service   |                         |  Worker (per   │                  |  Store      |
+------------+                         |  shard/region) │                  | (Prometheus,|
                                       +───────┬────────+                  |  TSDB)      |
                                               │                            +─────────────┘
                                               ▼
                                        TorchMetrics
                                        Running(NDCG)
                                        Running(MRR)
```

**Where TorchMetrics fits**

- Each Metric Stream Worker holds one or more `Running(metric, window=...)` per (region, model_version) tuple.
- Updates batch incoming events for ~50 ms or 1k items, whichever comes first; call `update(preds, target, indexes)` once per batch.
- Worker emits `metric.compute()` to the TSDB on a 30 s tick.

**Trade-offs and tricky decisions**

| Decision | Why |
|---|---|
| Per-shard rather than global metric | A single global metric is a single hot lock; sharding lets you scale horizontally and aggregate offline. |
| `Running` instead of `MetricCollection` | Running windows fit "the last hour" semantics that monitoring teams want. |
| `compute_on_cpu=True` | NDCG is list-state. RAM matters under high QPS. |
| Don't sync across workers in real time | Per-worker is enough for monitoring; a nightly batch job can compute the global number for reporting. |
| Cardinality cap on segments | A new dimension every day blows up your TSDB. Limit and drop tail. |

**Failure modes to call out**

- A worker restart loses its rolling window. Accept it (you'll re-fill in `window` events) or persist the metric state to disk via `metric.metric_state`.
- Schema drift in the event stream → silent metric breakage. Version the event schema and gate metric workers on it.

---

## SD2. Design an offline evaluation pipeline for a foundation-model finetune

**Brief**: 10B-token pretraining run, periodic eval on 50+ benchmark suites. Each suite has its own metric (perplexity, exact match, BLEU, BERTScore). Eval must be reproducible and fast.

**Design**

```text
                        ┌────────────────────────────┐
       checkpoint  ───► │  Eval Orchestrator (Ray /  │  ───► dashboard
                        │   Lightning Fabric)         │
                        └─────────────┬───────────────┘
                                      │
              ┌───────────────────────┼─────────────────────────┐
              ▼                       ▼                         ▼
       ┌────────────┐          ┌────────────┐            ┌────────────┐
       │ MMLU eval   │          │ HumanEval   │            │ XSum eval   │
       │ (Accuracy)  │          │ (pass@k)    │            │ (ROUGE,     │
       │             │          │             │            │   BERTScore)│
       └─────┬──────┘          └─────┬──────┘            └─────┬──────┘
             │                       │                          │
             └───────────────► metric_state dump  ───────────────┘
                                       │
                                       ▼
                              `merge_state` aggregator → JSON / dashboard
```

**Where TorchMetrics fits**

- One `MetricCollection` per benchmark, instantiated on each eval worker.
- After the worker finishes its shard, dump `metric.metric_state` to S3 (it's just a dict of tensors).
- The aggregator constructs a fresh metric, calls `merge_state(state_dict)` for each shard, then `compute()`.

**Trade-offs**

| Decision | Reason |
|---|---|
| `merge_state` instead of all-rank `compute()` | Workers run independently with different latencies; this decouples them. |
| Pinned versions of TorchMetrics + tokenizer + (BERTScore backbone) | Generative metrics are sensitive to subtle backbone changes; pinning is the only way to get reproducible numbers. |
| Use functional API for one-shot benchmarks | Cleaner for a stateless "compute on this fixed eval" job; less moving parts than the modular API. |
| Persist `metric.metric_state` to checkpoint | Lets eval be resumed mid-flight if a worker crashes. |

---

## SD3. Design a fairness-gated CI for model releases

**Brief**: Every PR that changes the model must pass overall accuracy *and* per-group fairness gates. The gates are configurable per model.

**Design**

1. PR triggers eval CI.
2. CI builds a `MetricCollection`:
   - Top-line: `Accuracy`, `F1`, `AUROC`.
   - Per-segment: `MetricCollection({"acc_group_X": Accuracy(...) for X in groups})` *or* a custom `BinaryFairness` metric.
3. CI compares to the production model's snapshot via a paired bootstrap (`BootStrapper`).
4. CI fails the PR if:
   - Top-line drops by more than τ.
   - Any group's metric drops by more than τ_group.
   - Fairness ratios fall outside `[0.8, 1.25]` (the four-fifths rule).

**TorchMetrics-specific tips**

- Cache the reference model's predictions to S3 — re-running them every PR is wasteful.
- Use `BootStrapper(metric, num_bootstraps=1000)` to get CI; a "drop" within the CI is not a real regression.
- Persist eval inputs/outputs alongside metrics so you can re-derive any new metric retrospectively.

---

## SD4. Design a metric for "evaluating an LLM judge"

**Brief**: You're using GPT-4 to grade outputs of your own model on a 7-point scale. You need to monitor the *judge's* reliability over time.

**Approach**

- Maintain a small "gold" set with human-labeled scores. Compare judge scores to humans:
  - `PearsonCorrCoef` / `SpearmanCorrCoef` → calibration.
  - `MeanAbsoluteError` → magnitude error.
  - `CohenKappa` after binning to ordinal categories → agreement above chance.
- Build a `MetricCollection` of all four; track over time with `MetricTracker`.
- Drift signal = correlation drops by Δ relative to last week → page on-call.

**Custom metric**

If you want a single composite score, write a custom `Metric` that combines the above into one weighted formula. Keep state as `(judge_scores, human_scores)` lists with `dist_reduce_fx="cat"` so DDP / multi-worker still work.

---

## SD5. Design metrics for an object-detection inference pipeline at the edge

**Brief**: Detection model runs on edge devices. You need eval metrics on a labeled validation set you sync from the cloud, plus a privacy-preserving aggregate signal back to the cloud.

**Design**

- On-device: `MeanAveragePrecision(box_format="xyxy")` per device.
- Periodically, the device dumps `metric.metric_state` (a list-state dict) to the cloud — but with raw boxes/scores stripped, only the **counts** of TP/FP/FN at standard IoU thresholds. (You'd implement this as a derived metric so the state is summable and small.)
- Cloud aggregator constructs a fresh "TP-count-only" metric and uses `merge_state` per device.

**Why a derived metric matters here**

`MeanAveragePrecision` keeps every prediction in memory until `compute()`. That's both a privacy and a bandwidth nightmare for edge → cloud. By computing TP/FP/FN per IoU threshold *on device* and only sending those summable counts up, you trade a small amount of accuracy (no IoU sweep finer than your bin) for huge gains in privacy and cost.

---

## SD6. Design an A/B-testable metric system

**Brief**: Two model variants are live; we route traffic between them. Statistically rigorous comparison is the goal.

**Design**

- Two independent metric instances per variant: `Accuracy`, `F1`, `RetrievalNDCG`, etc.
- For each instance, wrap with `BootStrapper`.
- For paired metrics where samples line up across variants (rare in production unless you mirror traffic), implement a paired-bootstrap custom metric: maintains two state lists, samples joint indices.
- Emit `compute()` to your A/B platform; let the platform's stats engine handle Mann-Whitney / paired bootstrap CIs.

**Trade-offs**

| Decision | Reason |
|---|---|
| Per-variant metrics, separate datasets | Avoids interference; easy to reason about. |
| Paired bootstrap only when traffic is mirrored | Otherwise you're estimating a difference of independent means; the unpaired CI is the right one. |
| Don't compare metrics across model versions of *different* eval sets | A/B compares decisions; offline benchmarks compare models. They're different jobs. |

---

## SD7. Design a service to compute large-scale FID

**Brief**: Researchers spawn many image-generation runs; each needs FID against the same reference set. Reference set is 50k images, generation set is 10k–100k.

**Design**

- Pre-compute the *reference statistics* (Inception activations' mean and covariance) **once** and cache them on shared storage. FID's reference statistics are independent of the candidate set.
- Each run:
  1. Loads the cached reference stats.
  2. Streams its generated images, calling `metric.update(images, real=False)` (TorchMetrics' `FrechetInceptionDistance` accepts a `real` flag and accumulates separate statistics for real/fake).
  3. Calls `compute()` at the end.

**Why this works**

`FrechetInceptionDistance` keeps running mean and covariance for both real and fake — these are tensor states, summable, finite-size. A fresh metric loaded with cached real stats is the same as one updated with the full real set.

**Pitfalls**

- Don't cache *predictions*; cache the running statistics. They're tiny (one mean vector and one cov matrix per feature dim).
- Pin the Inception backbone — different versions yield silently different FID numbers.

---

## SD8. Design a metric ledger for compliance auditing

**Brief**: Regulators want to see which model version made which prediction at which time, plus the running performance metrics around that period.

**Approach**

- Append-only event log of `(timestamp, model_version, prediction, eventual_label, segment)`.
- A separate replay service replays the log into TorchMetrics with a per-(model_version, segment) `Running` metric.
- For audit, the log + the replay code are sufficient to recompute any metric retrospectively. **This is the killer property** — TorchMetrics' deterministic, stateful `update`/`compute` makes audit trivial.

**Implications**

- Keep the metric library version in the audit metadata. A library update could change a metric's value (rare, but happens for things like ECE binning).
- Use `seed=` everywhere stochastic (`BootStrapper`, etc.) for reproducibility.

---

## How to use these in an interview

Pick **one** that's closest to the company's product (recommender, LLM, generative, edge), and walk through:

1. **Requirements clarification** — what's the SLO, the cardinality, the latency budget?
2. **Where TorchMetrics fits and where it doesn't** — be honest about list-state limits and per-rank assumptions.
3. **Failure modes** — drift in inputs, schema changes, worker restarts.
4. **Trade-offs**, not "the right answer." Senior interviewers want to see the *space* of decisions, not a single solution.
