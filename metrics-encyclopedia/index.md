---
title: Metrics Encyclopedia
---

# Metrics Encyclopedia — TorchMetrics Deep Dive

> A per-metric reference. For every metric across every domain in TorchMetrics:
> **what it computes · when to use it · the real-world scenario it solves · code · pitfalls**.
> Each domain also ships a parallel **interview deep-dive** with multi-level Q&A.

This site is a sibling to the high-level [`torchmetrics-deep-dive`](../torchmetrics-deep-dive/) site.
That one teaches the framework. This one is the **per-metric encyclopedia** — open the page for the metric you have to ship/defend tomorrow.

🎯 **Interactive dashboard:** [`dashboard/index.html`](./dashboard/index.html) — five modes (Revise, Quiz, Flashcards, Scenarios, Search), keyboard-driven, no server.

📦 **Deployment guide:** [`DEPLOYMENT.md`](./DEPLOYMENT.md) — three paths (read on GitHub / open dashboard locally / publish via Pages) with troubleshooting.

---

## How to use this site

Three reading modes:

1. **"I have a problem, which metric?"** → start at [Cross-family scenarios](./scenarios/index.md), then drill into the family page.
2. **"Tell me everything about *this* metric."** → jump straight to the family page and `Ctrl-F` the metric name. Each entry is self-contained.
3. **"I have an interview tomorrow."** → read the family `index.md`, then the family `interview-deep-dive.md`. Then [Cross-family scenarios](./scenarios/index.md). That's a 4-hour cram if you've already used these metrics.

Every entry on every family page follows the same template:

```
### MetricName
What it computes  (1-line plain English + formula in math)
Intuition         (why the formula has that shape)
Range / direction (range and whether higher = better)
When to use       (scenarios where this is the right pick)
When NOT to use   (scenarios where it lies to you)
Real-world scenario (one concrete production example)
Code              (functional + class API)
Pitfalls          (the 2-3 traps people hit)
```

You can read just the **bold lines** of any entry to skim a family in 5 minutes.

---

## Domain map

### Supervised learning

| Family | Metrics | Page | Interview |
|---|---|---|---|
| Classification | 28 (accuracy, F1, AUROC, AP, MCC, calibration, fairness, …) | [classification/](./classification/index.md) | [interview](./classification/interview-deep-dive.md) |
| Regression | 22 (MAE/MSE/RMSE/MAPE/wMAPE, R², Pearson, Spearman, CRPS, Tweedie, …) | [regression/](./regression/index.md) | [interview](./regression/interview-deep-dive.md) |
| Retrieval | 10 (NDCG, MAP, MRR, Hit@k, R-Precision, Fall-out, …) | [retrieval/](./retrieval/index.md) | [interview](./retrieval/interview-deep-dive.md) |

### Computer vision

| Family | Metrics | Page | Interview |
|---|---|---|---|
| Detection | 8 (mAP, IoU, GIoU, DIoU, CIoU, panoptic quality, …) | [detection/](./detection/index.md) | [interview](./detection/interview-deep-dive.md) |
| Segmentation | 4 (mean IoU, Dice, generalized Dice, Hausdorff) | [segmentation/](./segmentation/index.md) | [interview](./segmentation/interview-deep-dive.md) |
| Image quality | 23 (FID, KID, IS, MIFID, LPIPS, DISTS, ARNIQA, SSIM, MS-SSIM, PSNR, SAM, ERGAS, RASE, UQI, VIF, TV, SCC, D-λ, D_S, QNR, PSNRB, RMSE-SW, perceptual path length) | [image/](./image/index.md) | [interview](./image/interview-deep-dive.md) |
| Multimodal | 3 (CLIP score, CLIP-IQA, LVE) | [multimodal/](./multimodal/index.md) | [interview](./multimodal/interview-deep-dive.md) |
| Video | 1 (VMAF) | [video/](./video/index.md) | (combined) |
| Shape | 1 (Procrustes) | (in [video/](./video/index.md)) | — |

### Language & speech

| Family | Metrics | Page | Interview |
|---|---|---|---|
| Text / NLP | 16 (BLEU, SacreBLEU, ROUGE, METEOR-like, chrF, TER, EED, BERTScore, InfoLM, perplexity, edit distance, CER, WER, MER, WIL, WIP, SQuAD F1/EM) | [text/](./text/index.md) | [interview](./text/interview-deep-dive.md) |
| Audio | 9 (PESQ, STOI, SI-SDR, SI-SNR, SDR, SNR, C-SI-SNR, PIT, SRMR, DNSMOS, NISQA) | [audio/](./audio/index.md) | [interview](./audio/interview-deep-dive.md) |

### Unsupervised, fairness, helpers

| Family | Metrics | Page | Interview |
|---|---|---|---|
| Clustering | 11 (Rand, Adjusted Rand, MI, NMI, AMI, Fowlkes-Mallows, Cluster Accuracy, Calinski-Harabasz, Davies-Bouldin, Dunn, Homogeneity-Completeness-V) | [clustering/](./clustering/index.md) | [interview](./clustering/interview-deep-dive.md) |
| Nominal / categorical | 5 (Cramér's V, Tschuprow's T, Pearson's contingency, Theil's U, Fleiss kappa) | [nominal/](./nominal/index.md) | [interview](./nominal/interview-deep-dive.md) |
| Wrappers | 10 (BootStrapper, ClasswiseWrapper, MetricTracker, MultioutputWrapper, MultitaskWrapper, MinMax, Running, FeatureShare, Transform, Abstract) | [wrappers/](./wrappers/index.md) | [interview](./wrappers/interview-deep-dive.md) |
| Aggregation | 8 (Sum, Mean, Min, Max, Cat, RunningSum, RunningMean) | [aggregation/](./aggregation/index.md) | (combined with wrappers) |

### Cross-family

| Page | Why it's separate |
|---|---|
| [Cross-family scenarios](./scenarios/index.md) | Real systems mix metrics from 3-4 families (recsys = retrieval + classification + regression + fairness). One page mapping 25 systems → metric stacks. |

---

## What each interview deep-dive page contains

Every `interview-deep-dive.md` follows the same Q → F1 → F1.1 drill-down format:

```
Q1 (asked in the screen)
  F1.1 (asked once you answer Q1)
    F1.1.1 (asked once you answer F1.1 — only senior loops go this deep)
  F1.2 (alternative drill-down branch)
```

Each leaf has a model answer plus a "what NOT to say" line where there's a common bad answer.

The drill-down format is what you actually face in real loops — interviewers don't move to Q2 until they've squeezed every layer out of Q1. Practising flat lists of questions is *not* the same shape.

---

## Conventions used everywhere

- **`B` = batch size, `C` = classes, `K` = top-K, `N` = total samples**.
- **`y` = ground truth, `ŷ` = prediction, `p` = predicted probability**.
- **"Higher is better" / "Lower is better"** is in every entry — get this wrong and your model ranking flips.
- **Range** is given as `[lo, hi]` (closed) or `[lo, ∞)` (open above) — important for `MetricTracker(maximize=...)` and for plotting.
- **Functional vs class** — every entry shows both, because Lightning's `self.log` wants the *class* form (it owns state), while one-shot eval wants the functional.
- **`task=` argument** — most classification/regression metrics in TorchMetrics 1.x take `task="binary"|"multiclass"|"multilabel"`. Always specify it; the default behaviour can change between versions.

---

## License

Same as parent repo (Apache-2.0).
