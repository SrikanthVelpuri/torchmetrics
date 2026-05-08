---
title: Clustering metrics — interview deep dive
---

# Clustering metrics — interview deep dive

[← Family page](./index.md) · [← Home](../index.md)

---

## Q1. "Why is NMI biased and what corrects it?"

**Answer.** Normalised Mutual Information doesn't account for the *expected* MI under random labelling. With many small clusters, random partitions can achieve substantial NMI just by accident. **AMI** (Adjusted MI) subtracts the expected NMI from the score and rescales — random labelling gets AMI ≈ 0. Always prefer AMI over NMI for comparing partitions with different numbers of clusters.

---

## Q2. "Homogeneity, completeness, V-measure — when does each fail?"

**Answer.**
- **Homogeneity** = 1 if every cluster has only one class. Fails: a model that puts every point in its own cluster scores homogeneity = 1 (each singleton has one class) but is useless.
- **Completeness** = 1 if every class is in one cluster. Fails: putting all points in one giant cluster scores completeness = 1.
- **V-measure** = harmonic mean: punishes either failure. **But** still doesn't correct for chance.

For chance-correction, use **AMI** as the headline; report homogeneity/completeness as the diagnostic.

---

## Q3. "Calinski-Harabasz keeps going up as k increases. Bug?"

**Answer.** No, expected. CH = `(between dispersion) / (within dispersion) × (n − k) / (k − 1)`. Scaling factor `(n−k)/(k−1)` decreases with k, but for typical data the dispersion ratio increases faster. So **CH alone doesn't pick k** — you need an elbow heuristic on the CH-vs-k curve. For automatic k-selection, use the **gap statistic** or stability-based methods.

---

## Q4. "ARI vs AMI — when do they disagree?"

**Answer.** ARI works on pairs, AMI on contingency-table information. They diverge when:
- **Cluster sizes are very unequal**: ARI is more affected by mismatches in big clusters; AMI is balanced.
- **Number of clusters differs between true and predicted**: AMI handles this gracefully; ARI can be pulled around.

In practice they agree on which model is better; they can disagree on absolute magnitude. Pick one and stick with it. AMI is the more conservative default.

---

## Q5. "What's the right metric when you have N clusters but only ground truth for a sub-cluster structure?"

**Answer.** Use the **homogeneity** of the predicted clustering w.r.t. the true sub-structure: each predicted cluster should be pure even if the granularity differs. Completeness will be low (because the true classes are split), and that's fine — your model is *over-segmenting*. Report homogeneity as primary; completeness as the diagnostic.

---

## Q6. "Internal vs external metrics — when do you use which?"

**Answer.**
- **External** (ARI, AMI, V-measure): need ground-truth labels. Use to compare clustering algorithms on labelled data.
- **Internal** (Davies-Bouldin, Calinski-Harabasz, Dunn, Silhouette): geometric scores, no labels. Use for unlabelled data or to pick `k`.

Real workflow: tune `k` on internal metrics, validate the chosen clustering against any partial ground truth via external metrics on a held-out subset.

---

## Q7. "Cluster accuracy with Hungarian assignment — what's the trick?"

**Answer.** Predicted cluster IDs are arbitrary; `cluster_4` could correspond to any class. Hungarian algorithm finds the optimal one-to-one assignment between predicted clusters and true classes that maximises diagonal entries of the contingency matrix. Then apply standard accuracy. Used in self-supervised learning evaluations (e.g., DeepCluster, SeLa, IIC).

> **F7.1** "What if `num_clusters > num_classes`?"
>
> **Answer.** Hungarian leaves extra clusters unmatched (counted as 0 correct). Either merge clusters before assignment, or use `cluster_acc(over_cluster_factor=k)` with k > 1, common in over-clustering papers.

---

## Q8. "Davies-Bouldin score is 0.4 — is that good?"

**Answer.** Lower is better; 0.4 is *better than 1.0* but the absolute is dataset-dependent. Compare across algorithms on the same dataset. DB depends on within-cluster scatter vs between-cluster distance — datasets with naturally well-separated clusters get low DB regardless of algorithm.

---

## Q9. "How does TorchMetrics handle DDP for clustering metrics?"

**Answer.** State is per-rank lists of `(cluster_id, ground_truth_id)` for external metrics, and feature embeddings for internal ones. `_sync_dist` gathers; cost O(N). For internal metrics with large feature dim, use sub-sampling — Calinski-Harabasz on 100k randomly-sampled points is unbiased for the population value.

---

## Q10. "Fowlkes-Mallows vs ARI — same thing?"

**Answer.** Both pair-based, both compare clusterings. Different math: FMI = `sqrt(P · R)` over pairs. ARI = chance-adjusted Rand. ARI is preferred because it has a chance correction; FMI doesn't. They typically rank algorithms similarly, so reporting both is redundant.

---

[← Back to family page](./index.md)
