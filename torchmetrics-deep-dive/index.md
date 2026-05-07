---
title: TorchMetrics Deep Dive
nav_order: 1
---

# TorchMetrics Deep Dive

A complete, beginner-to-advanced learning site that explains **TorchMetrics** — the official metrics library for PyTorch and PyTorch Lightning — from first principles through production-grade design trade-offs and ML interview preparation.

> Companion guide to the source repository at
> [Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics).
> Code references in this site point to files inside `src/torchmetrics/`.

---

## 1. Repository Understanding (TL;DR)

### What TorchMetrics is

**TorchMetrics is a PyTorch-native library of 100+ pre-built, rigorously tested machine-learning metrics, plus a small but powerful base class (`Metric`) that lets you build your own custom metrics that "just work" across CPUs, GPUs, multi-GPU, and multi-node distributed training.**

It exposes two parallel APIs:

1. A **functional API** — pure functions like `torchmetrics.functional.accuracy(preds, target)` that compute a metric on a single batch.
2. A **modular API** — `torch.nn.Module` subclasses like `torchmetrics.Accuracy()` that *accumulate* state across batches, synchronize across devices, and produce one final number at the end of an epoch.

### Why it exists

Originally, metrics lived inside **PyTorch Lightning**. As the metric library grew, the maintainers split it out so that:

- Users of *plain* PyTorch could use the same battle-tested metrics without installing Lightning.
- Lightning users would still get tight integration (auto device placement, auto logging).
- A single, canonical implementation could replace the dozens of slightly-wrong copies of `accuracy()` floating around the ecosystem.

### What problem it solves

Hand-writing metrics during training is deceptively hard. The common pitfalls are:

| Pitfall | What goes wrong | How TorchMetrics fixes it |
|---|---|---|
| **Batch averaging** | Averaging per-batch accuracy is *not* the same as overall accuracy when batches differ in size or class balance. | `update()` accumulates raw counts; `compute()` produces the correct global number. |
| **Distributed aggregation** | On 8 GPUs, every rank computes its own number. Naïvely averaging them is wrong for non-linear metrics like F1 or AUROC. | `Metric.sync()` `all_gathers` *state* (TP/FP/TN/FN, predictions, targets), then computes once on rank 0. |
| **Device placement** | `target` is on `cuda:0`, the running sum is on `cpu`. Crash. | `Metric` is a `nn.Module`; `metric.to(device)` moves all state at once. |
| **Numerical stability** | Edge cases like all-negative predictions, zero-support classes, tied scores. | Each metric has dedicated tests, including reference comparison vs. scikit-learn / NumPy. |
| **API drift** | Every researcher writes a slightly different `f1()`. | One canonical interface: `update`, `compute`, `reset`. |

### How it differs from writing metrics manually

A manual implementation usually looks like:

```python
total, correct = 0, 0
for x, y in loader:
    pred = model(x).argmax(-1)
    correct += (pred == y).sum().item()
    total   += y.numel()
print(correct / total)
```

This works for **accuracy on one GPU**. It silently breaks for:

- F1 / AUROC / mAP (non-linear in batch size)
- Multi-GPU DDP (each rank only sees its own shard)
- Mixed-precision inference (dtype mismatches)
- Streaming metrics across many epochs (you forgot to reset)
- Logging from within Lightning hooks (no auto-device-move)

TorchMetrics handles all of these in the base class, so the metric author only writes the math.

### How it integrates with PyTorch and PyTorch Lightning

- **Plain PyTorch**: import a metric, `.update()` it inside your training loop, `.compute()` at the end of an epoch, `.reset()` before the next.
- **PyTorch Lightning**: register the metric as a module attribute (`self.acc = torchmetrics.Accuracy(...)`). Lightning then:
  - moves it to the correct device automatically,
  - logs it with `self.log("acc", self.acc)` (which calls `forward → compute → reset` correctly),
  - syncs it across DDP workers when you set `sync_dist=True`.

### Who uses it and when

- **Researchers** — for reproducible benchmark numbers (the same Accuracy implementation everyone agrees on).
- **Production ML teams** — for monitoring drift via metric trackers (`MinMaxMetric`, `RunningMean`, `MetricTracker`).
- **Lightning users** — as the default metrics layer.
- **Kaggle / competition users** — for fast, GPU-correct AUROC / mAP / BLEU.

### Key folders and files (in `src/torchmetrics/`)

| Path | Role |
|---|---|
| `metric.py` | The base `Metric` class — the heart of the library. |
| `collections.py` | `MetricCollection` — group many metrics with shared state. |
| `aggregation.py` | Generic aggregators: `SumMetric`, `MeanMetric`, `MaxMetric`, `CatMetric`. |
| `classification/` | `Accuracy`, `F1`, `AUROC`, `Precision`, `Recall`, `ConfusionMatrix`, … |
| `regression/` | `MAE`, `MSE`, `R2Score`, `PearsonCorrCoef`, … |
| `retrieval/` | `RetrievalMRR`, `RetrievalNDCG`, `RetrievalMAP`, … |
| `text/`, `audio/`, `image/`, `multimodal/`, `detection/`, `segmentation/`, `nominal/`, `clustering/`, `video/`, `shape/` | Domain-specific metrics. |
| `wrappers/` | `MetricTracker`, `BootStrapper`, `MultioutputWrapper`, `ClasswiseWrapper`, `Running`, `MinMaxMetric`. |
| `functional/` | Stateless function versions of every metric. |
| `utilities/` | `distributed.py` (gather), `data.py` (reductions), `checks.py`, `imports.py`, `plot.py`. |
| `tests/` | Mirror of the source tree; every metric is unit-tested against a reference (sklearn / numpy / scipy). |
| `docs/` | Sphinx documentation source; this site complements (does not replace) it. |

### Important classes and abstractions

- **`Metric`** (`metric.py`) — abstract base; child classes implement `update()` and `compute()` and call `add_state()` in `__init__`.
- **`MetricCollection`** — bundles many metrics; can share intermediate computation via *compute groups*.
- **Wrappers** — turn any `Metric` into a richer one (running window, bootstrap CI, per-class breakdown, multi-output, max-tracking).
- **Functional vs. modular** — every modular metric has a stateless functional sibling under `torchmetrics.functional`.

### How metrics are registered, implemented, tested, and documented

- **Registration**: each subpackage's `__init__.py` re-exports its public metrics. The top-level `torchmetrics/__init__.py` re-exports the most common ones so `torchmetrics.Accuracy` works.
- **Implementation**: every metric's `update()` only mutates state via `add_state()` variables; `compute()` is pure math from those states.
- **Testing**: each metric has a parity test in `tests/unittests/` against a trusted reference (e.g. sklearn). DDP behavior is tested too.
- **Documentation**: each metric has a Sphinx `.rst` page under `docs/source/<domain>/`, and a docstring with mathematical formulae, shape tables, and runnable doctests.

---

## 2. How this site is organized

| Page | What you'll learn |
|---|---|
| [Getting Started]({{ "./getting-started.md" | relative_url }}) | Install, first metric, functional vs. modular. |
| [Core Concepts]({{ "./core-concepts.md" | relative_url }}) | `update`/`compute`/`reset`, state, the metric lifecycle. |
| [Metric Class Internals]({{ "./metric-class-internals.md" | relative_url }}) | What actually happens inside `Metric.__init__`, `forward`, `_sync_dist`. |
| [Classification Metrics]({{ "./classification-metrics.md" | relative_url }}) | Binary / multiclass / multilabel, threshold handling, AUROC math. |
| [Regression Metrics]({{ "./regression-metrics.md" | relative_url }}) | MAE, MSE, R2, correlations, quantile losses. |
| [Retrieval Metrics]({{ "./retrieval-metrics.md" | relative_url }}) | MRR, MAP, NDCG, the `indexes=` argument. |
| [Text, Audio, Image]({{ "./text-audio-image-metrics.md" | relative_url }}) | BLEU, ROUGE, BERTScore, FID, SSIM, PESQ, SI-SDR. |
| [Distributed Training]({{ "./distributed-training.md" | relative_url }}) | How `all_gather` aggregation actually works under DDP. |
| [PyTorch Lightning Integration]({{ "./pytorch-lightning-integration.md" | relative_url }}) | `self.log`, sync_dist, on_step vs. on_epoch. |
| [Custom Metrics]({{ "./custom-metrics.md" | relative_url }}) | Building your own metric the right way. |
| [Production Scenarios]({{ "./production-scenarios.md" | relative_url }}) | Drift monitoring, A/B tests, online evaluation. |
| [Testing & Validation]({{ "./testing-and-validation.md" | relative_url }}) | How TorchMetrics tests itself; how to test yours. |
| [Interview Questions]({{ "./interview-questions.md" | relative_url }}) | 25+ ML-engineering interview questions with model answers. |
| [Follow-Up Questions]({{ "./follow-up-questions.md" | relative_url }}) | The "drill-down" questions interviewers actually ask. |
| [System Design Questions]({{ "./system-design-questions.md" | relative_url }}) | Designing a metrics service for a recommender, an LLM, a CV pipeline. |
| [ML ↔ Business Metrics]({{ "./ml-business-metrics.md" | relative_url }}) | Connecting ML metrics to business KPIs at American Airlines and Amazon. |
| [Scenario Setups]({{ "./scenario-setups.md" | relative_url }}) | How metrics are wired in real ML systems (CV, NLP, speech, recsys, time-series, anomaly). |
| [Troubleshooting]({{ "./troubleshooting.md" | relative_url }}) | The bugs you will hit and how to fix them. |

---

## How to read this site

1. Skim **Getting Started** and run the snippets locally.
2. Then **Core Concepts** and **Metric Class Internals** — these unlock everything else.
3. Pick the **domain page** (classification / regression / …) you actually need.
4. Use the **interview / follow-up / system-design** pages as a self-test before an interview.
5. For staff-level prep, read **ML ↔ Business Metrics** and **Scenario Setups** — these tie the library to real product / company decisions.

Each domain page now also includes an **Interview Drill-Down** section with multi-level follow-ups (Q → F1 → F1.1 → F1.1.1) so you can practice the way real interviewers escalate.

---

## 🎓 [Mastery Hub]({{ "./mastery-hub.md" | relative_url }}) — start here for interview prep

A standalone landing page that organizes everything below into a 7-tier learning path, with a 30-day roadmap, visual cheat sheets, mnemonics, decision trees, code challenges, debug exercises, and 10 company scenarios.

If you only have time to bookmark one page, bookmark the Mastery Hub.

---

## 🎯 Interactive Mastery Dashboard

A static, self-contained HTML app with **six modes**:

[**▶ Open the dashboard →**](./dashboard/index.html)

- **📖 Revise** — concise summaries of every topic.
- **🎯 Quiz** — multi-level follow-ups (Q → F1 → F1.1 → F1.1.1).
- **🃏 Flashcards** — flip cards with 1–5★ confidence rating.
- **🎲 Random Mix** — random question from anywhere.
- **🃏 Flashcards (All)** — weak-first spaced repetition queue.
- **📊 Mastery Map** — heat-map of confidence across all questions.

Confidence ratings persist in `localStorage`. Color-coded sidebar pips show your weak topics at a glance.

---

## New mastery pages

| Page | Purpose |
|---|---|
| [🎓 Mastery Hub]({{ "./mastery-hub.md" | relative_url }}) | The master entry — everything organized into a 7-tier path. |
| [📅 Mastery Roadmap]({{ "./mastery-roadmap.md" | relative_url }}) | 30 days, 45 min/day, structured. |
| [📑 Cheat Sheets]({{ "./cheat-sheets.md" | relative_url }}) | 18 visual one-pagers — print and pin. |
| [🧠 Mnemonics]({{ "./mnemonics.md" | relative_url }}) | 21 memory hooks (RUUCC, MEPP, PRINCE, …). |
| [🌳 Decision Trees]({{ "./metric-decision-tree.md" | relative_url }}) | "Which metric for X?" flowcharts. |
| [📦 Wrappers Deep Dive]({{ "./wrappers-deep-dive.md" | relative_url }}) | All 11 wrappers with composition recipes. |
| [➕ Aggregation, Clustering, Pairwise, Nominal]({{ "./aggregation-clustering-pairwise.md" | relative_url }}) | The under-documented metric families. |
| [💻 Code Challenges]({{ "./code-challenges.md" | relative_url }}) | 12 implement-from-scratch problems. |
| [🐛 Spot the Bug]({{ "./spot-the-bug.md" | relative_url }}) | 15 buggy snippets — debug, then check. |
| [🔢 Numerical Pitfalls]({{ "./numerical-pitfalls.md" | relative_url }}) | NaN, precision, edge cases. |
| [🏢 Extended Company Scenarios]({{ "./extended-company-scenarios.md" | relative_url }}) | Netflix, Uber, Stripe, Meta, Tesla, healthcare, finance, more. |
