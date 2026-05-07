---
title: Aggregation, Clustering, Pairwise & Nominal
nav_order: 25
---

# The Underdocumented Metric Families

Most TorchMetrics tutorials stop at classification and regression. But you'll be asked about these in any senior interview because they show up in **real production**: clustering for unsupervised learning, pairwise for retrieval embeddings, aggregation for streaming KPIs, nominal for survey/social data.

---

## 1. Aggregation metrics — the building blocks

These look trivial but they're the chassis many custom metrics are built on.

| Metric | What it accumulates | Reduction |
|---|---|---|
| `SumMetric` | Sum of all updates | sum |
| `MeanMetric` | Running mean | sum (sum + count) |
| `MaxMetric` | Running max | max |
| `MinMetric` | Running min | min |
| `CatMetric` | Concatenation of all updates | cat |
| `RunningMean` / `RunningSum` | Rolling-window variant | window-based |

```python
from torchmetrics.aggregation import SumMetric, MeanMetric, MaxMetric

# Track total dollars saved by a model
dollars = SumMetric()
dollars.update(action_value_per_batch)
dollars.compute()   # cumulative total
```

**When to use them.** Whenever you have a custom KPI that's "just sum / mean / max of these values across the eval set." Don't reimplement — use these.

**Interview hook.** "What's the simplest custom metric in TorchMetrics?"

> **A.** `SumMetric`. Two states (`sum_value`, `weight`), sum-reduced, DDP-correct. ~30 lines. Reading it teaches you the entire `Metric` API in 10 minutes.

---

## 2. Clustering metrics — when you don't have ground-truth labels

Located under `torchmetrics.clustering`. Two flavors:

### Internal indices (no ground truth)

| Metric | Class | What it measures |
|---|---|---|
| Silhouette | `silhouette_score` (functional) | Cluster cohesion vs separation; ∈ [-1, 1] |
| Calinski-Harabasz | `CalinskiHarabaszScore` | Variance ratio (between/within); higher better |
| Davies-Bouldin | `DaviesBouldinScore` | Avg similarity to nearest cluster; lower better |
| Dunn index | `DunnIndex` | Min inter-cluster / max intra-cluster |

### External indices (you do have labels)

| Metric | Class | What it measures |
|---|---|---|
| Mutual Info | `MutualInfoScore` | Information shared between predicted and true clusters |
| Normalized MI | `NormalizedMutualInfoScore` | MI scaled to [0, 1] |
| Adjusted MI | `AdjustedMutualInfoScore` | MI corrected for chance |
| Rand index | `RandScore` | Pair-agreement of clusterings |
| Adjusted Rand | `AdjustedRandScore` | Rand corrected for chance |
| Homogeneity | `HomogeneityScore` | Each cluster contains only one class |
| Completeness | `CompletenessScore` | Each class is in exactly one cluster |
| V-measure | `VMeasureScore` | Harmonic mean of homogeneity + completeness |
| Fowlkes-Mallows | `FowlkesMallowsIndex` | Geometric mean of pair-precision and pair-recall |

**Decision tree:**

```text
                CLUSTERING EVAL
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
   Have labels?               No labels?
        │                           │
        ▼                           ▼
   Adjusted Rand Score        Silhouette
   + Adjusted MI              + Calinski-Harabasz
   (chance-corrected)
        │
        ▼
   Want soft summary?
   →  V-measure
   (homogeneity + completeness)
```

**Production trap.** Internal indices like Silhouette are *expensive* (O(n²) pairwise distances). For large clusterings, sample or use approximate variants.

**Interview drill-down:**

> **Q.** Why "adjusted" Rand and not plain Rand?

> **A.** Plain Rand is biased upward — random labelings score above zero. Adjusted Rand subtracts the expected value under random labeling, so 0 = chance and 1 = perfect.

>> **F1.** What if the true clustering is unknown but you suspect it?

>> **A.** Run multiple clusterings with different K, plot Adjusted Rand against assumed K. The "elbow" gives an estimate. Combine with internal indices (Silhouette) for a sanity check.

---

## 3. Pairwise distance metrics

`torchmetrics.functional.pairwise.*` — pairwise distances, used in:

- Embedding-based retrieval ("how close are these vectors?").
- Nearest-neighbor evaluation.
- Custom contrastive losses' eval (cosine on positive vs negative pairs).
- Clustering distance pre-compute.

| Function | Distance |
|---|---|
| `pairwise_cosine_similarity` | 1 − (a·b)/(\|a\|\|b\|) |
| `pairwise_euclidean_distance` | √Σ(a_i − b_i)² |
| `pairwise_manhattan_distance` | Σ\|a_i − b_i\| |
| `pairwise_minkowski_distance` | (Σ\|a_i − b_i\|^p)^(1/p) |
| `pairwise_linear_similarity` | a · b |

**These are functional only** — no modular state-tracking metric. Use them inside your own custom metric or as raw building blocks.

**Performance note**: pairwise distances on N×N grow quadratically. For N=10⁴ vectors of dim 768, that's 10⁸ × 768 floats = ~300 GB. **Always chunk** for production.

```python
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
def chunked_pairwise(a, b, chunk_size=1024):
    out = []
    for i in range(0, a.size(0), chunk_size):
        out.append(pairwise_cosine_similarity(a[i:i+chunk_size], b))
    return torch.cat(out, dim=0)
```

---

## 4. Nominal metrics — for categorical / survey data

`torchmetrics.nominal` — measures of association between categorical variables.

| Metric | Class | Use |
|---|---|---|
| Cramér's V | `CramersV` | Symmetric association between two categorical vars |
| Theil's U | `TheilsU` | Asymmetric: how much one var explains the other |
| Tschuprow's T | `TschuprowsT` | Like Cramér's but corrects for table shape |
| Pearson contingency | `PearsonsContingencyCoefficient` | Older, less preferred |

**When you'd use these:**

- Survey analysis ("does region predict survey response?").
- Feature analysis on categorical inputs.
- Bias detection (correlation between protected attribute and prediction).

```python
from torchmetrics.nominal import CramersV

# Are predictions correlated with a protected categorical attribute?
cv = CramersV(num_classes=4, nan_strategy="replace")
cv.update(predictions, protected_group)
print(cv.compute())   # 0 = no association, 1 = perfect
```

**Interview hook.** "How would you measure if a model's predictions are independent of a categorical protected attribute?"

> **A.** Cramér's V between predictions and the protected group. Below ~0.1 = essentially independent. Above ~0.3 = meaningful association — investigate.

---

## 5. Shape metrics

`torchmetrics.shape` — for evaluating shape/correspondence problems (medical, robotics).

- **Procrustes Disparity** — minimum sum of squared distances after optimal rotation/scaling alignment. Used in shape analysis.

Niche but worth knowing exists; you'll never get a primary interview question but it can show up in CV/biomedical roles.

---

## 6. Multimodal metrics

`torchmetrics.multimodal`:

| Metric | Class | Use |
|---|---|---|
| CLIP Score | `CLIPScore` | Text–image alignment via CLIP cosine |
| CLIP IQA | `CLIPImageQualityAssessment` | Reference-free image quality via CLIP-prompted comparisons |

**The thing to know about CLIPScore.** It's the *standard* metric for evaluating text-to-image generation today. FID measures fidelity; CLIPScore measures whether the image matches the prompt. They're orthogonal — report both.

```python
from torchmetrics.multimodal import CLIPScore

clip = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
clip.update(images, prompts)
print(clip.compute())   # higher = better text-image alignment
```

**Caveat**: CLIPScore inherits CLIP's biases. Don't use it as the *only* metric — pair with FID/KID and human eval.

---

## 7. Detection metrics — the canonical mAP

`torchmetrics.detection.MeanAveragePrecision` is the COCO-style implementation. Already covered in the [Scenario Setups]({{ "./scenario-setups.md" | relative_url }}). Two things worth re-emphasizing:

- **Always pass `compute_on_cpu=True`** — predictions and targets accumulate; large eval runs OOM otherwise.
- **`class_metrics=True`** to get per-class AP for diagnosis. Log only worst-K class APs.

**Interview hook**: "Why doesn't TorchMetrics give you GPU mAP?"

> **A.** The COCO evaluator is in NumPy/Cython on CPU. Sorting and Hungarian-style box matching is awkward to fuse on GPU. TM ships the canonical implementation for trust over speed; it's faster to optimize the *training* loop.

---

## 8. Segmentation metrics — the often-confused

`torchmetrics.segmentation`:

| Metric | What it measures |
|---|---|
| `MeanIoU` | Mean Jaccard over classes |
| `GeneralizedDiceScore` | Class-weighted Dice (square / inverse / linear weighting) |
| `HausdorffDistance` | Worst-case boundary distance |

**Dice vs IoU**: monotonic but not affine.

```text
IoU = TP / (TP + FP + FN)
Dice = 2 TP / (2 TP + FP + FN) = 2 IoU / (1 + IoU)
```

Always reportable together — reviewers expect it. **For boundary quality**, use Hausdorff but prefer **HD95** (95th percentile, not max) in clinical reporting because the worst-case is dominated by single rogue voxels.

---

## 9. Video metrics

`torchmetrics.video`:

- **VMAF** (`VideoMultiMethodAssessmentFusion`) — Netflix's perceptual video quality metric.

Wraps `libvmaf`. Useful for video codec evaluation, streaming quality, video-generation eval.

---

## 10. Cheat-sheet table — which family for what

```text
+----------------------+----------------------------------+
| Family               | When you'd use it                |
+----------------------+----------------------------------+
| Aggregation          | Custom KPIs from sums/means/maxes|
| Classification       | Class labels (binary/multi/multi)|
| Regression           | Continuous targets               |
| Retrieval            | Per-query ranked lists           |
| Text                 | NLP outputs                      |
| Audio                | Speech / source separation       |
| Image                | Pixel/feature image comparison   |
| Detection            | Object detection (mAP)           |
| Segmentation         | Mask quality (Dice/IoU/Hausdorff)|
| Clustering           | Unsupervised grouping eval       |
| Pairwise             | Embedding distance pre-compute   |
| Nominal              | Categorical association / bias   |
| Multimodal           | Text-image / cross-modal         |
| Shape                | Geometric shape correspondence   |
| Video                | Video quality (VMAF)             |
+----------------------+----------------------------------+
```

---

## Hidden gems checklist

These are useful and almost no one mentions them in interviews:

- **`SumMetric` and `MeanMetric`** — perfect base classes for custom $-loss / dollar-weighted metrics.
- **`silhouette_score` (functional)** — quick clustering eval without a class instance.
- **`pairwise_cosine_similarity` chunked** — embedding-similarity infra.
- **`CramersV`** — bias / fairness diagnostic on categorical attributes.
- **`CLIPScore`** — the t2i prompt-fidelity standard.
- **`CalibrationError` with adaptive bins** — better than equal-width on skewed predictions.

If you can name three of these correctly, you're already above the bar for "TorchMetrics knowledge" in most interviews.
