---
title: Clustering metrics — deep dive
---

# Clustering metrics — deep dive

> Clustering metrics split into:
> - **External** (you have ground-truth cluster labels): Rand, ARI, MI, NMI, AMI, Fowlkes-Mallows, V-measure, Cluster Accuracy.
> - **Internal** (no ground truth, geometric): Calinski-Harabasz, Davies-Bouldin, Dunn.

[← Home](../index.md) · [Interview deep-dive →](./interview-deep-dive.md)

## External (with ground truth)

These compare predicted clustering against true labels. Predicted labels can be in any order — they're just partition IDs.

### RandScore (`tm.clustering.RandScore`)

**What it computes.** Fraction of pairs of samples that are *concordant*: either in the same cluster in both partitions or in different clusters in both. `(TP + TN) / (TP + TN + FP + FN)` over pairs.

**Range / direction.** `[0, 1]`. Higher better.

**Pitfalls.**
- Not adjusted for chance: random labeling can score high.

---

### AdjustedRandScore / ARI (`tm.clustering.AdjustedRandScore`)

**What it computes.** Rand index corrected for chance: `(RI − E[RI]) / (max(RI) − E[RI])`.

**Range / direction.** `[-1, 1]` (typically `[0, 1]`). Higher better. 0 = random.

**Real-world scenario.** Comparing two clusterings of the same dataset (e.g., a new algorithm vs k-means baseline). ARI is the standard.

---

### MutualInfoScore (`tm.clustering.MutualInfoScore`)

**What it computes.** `I(X, Y) = Σ p(x, y) log(p(x, y) / (p(x) p(y)))` between cluster assignments.

**Range / direction.** `[0, ∞)`. Higher better.

---

### NormalizedMutualInfoScore / NMI (`tm.clustering.NormalizedMutualInfoScore`)

**What it computes.** MI normalised by the entropies of X and Y.

**Range / direction.** `[0, 1]`. Higher better.

**Pitfalls.**
- NMI doesn't correct for chance — random clusterings of many small clusters score artificially high. Use **AMI** instead.

---

### AdjustedMutualInfoScore / AMI (`tm.clustering.AdjustedMutualInfoScore`)

**What it computes.** MI corrected for chance, similar to ARI for the Rand index.

**Real-world scenario.** Default external clustering metric when number of clusters varies between partitions — AMI is the chance-corrected, balanced choice.

---

### FowlkesMallowsIndex (`tm.clustering.FowlkesMallowsIndex`)

**What it computes.** Geometric mean of pairwise precision and recall: `sqrt(P · R)` over pairs.

**Range / direction.** `[0, 1]`. Higher better.

---

### Homogeneity, Completeness, V-measure (`tm.clustering.HomogeneityScore`, `CompletenessScore`, `VMeasureScore`)

**Homogeneity**: each cluster contains only one true class (no mixing).
**Completeness**: each true class is in only one cluster (no splitting).
**V-measure**: harmonic mean of the two.

**Real-world scenario.** Diagnosing clustering failures: low homogeneity = clusters are mixed; low completeness = classes are split. V-measure is the single number; the two components are the diagnostic.

---

### ClusterAccuracy (`tm.clustering.ClusterAccuracy`)

**What it computes.** Accuracy after solving the optimal cluster-to-class assignment via Hungarian algorithm.

**Real-world scenario.** Unsupervised classification: you cluster into k groups, then map each group to a class via Hungarian, report accuracy. Used in self-supervised representation evaluation.

**Pitfalls.**
- Requires `num_clusters == num_classes`. Otherwise, decide whether to merge or split.

---

## Internal (no ground truth)

These score a clustering by its geometric properties.

### CalinskiHarabaszScore (`tm.clustering.CalinskiHarabaszScore`)

**What it computes.** Ratio of between-cluster dispersion to within-cluster dispersion.

**Range / direction.** `[0, ∞)`. Higher better.

**Real-world scenario.** k-selection — sweep `k`, plot Calinski-Harabasz, pick the `k` at the elbow / max.

**Pitfalls.**
- Biased toward higher k (more clusters often increase between-dispersion). Combine with elbow rule.

---

### DaviesBouldinScore (`tm.clustering.DaviesBouldinScore`)

**What it computes.** Average ratio of within-cluster scatter to between-cluster separation, taking max over closest cluster.

**Range / direction.** `[0, ∞)`. **Lower better.**

---

### DunnIndex (`tm.clustering.DunnIndex`)

**What it computes.** Minimum inter-cluster distance / maximum intra-cluster diameter.

**Range / direction.** `[0, ∞)`. Higher better.

**Pitfalls.**
- Sensitive to outliers — a single outlier pumps the maximum diameter.

---

## Quick-reference: which clustering metric for which scenario?

| Scenario | Primary | Secondary |
|---|---|---|
| Have ground truth, comparing 2 partitions | ARI | AMI |
| Mixed cluster diagnosis | V-measure | Homogeneity, Completeness separately |
| Unsupervised classification (k = num_classes) | Cluster Accuracy | ARI |
| Choosing k (no ground truth) | Calinski-Harabasz | Silhouette (external) |
| Internal compactness | Davies-Bouldin | Dunn |

---

[Continue → interview deep-dive](./interview-deep-dive.md)
