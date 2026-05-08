---
title: Retrieval metrics — deep dive
---

# Retrieval metrics — deep dive

> All retrieval metrics evaluate **ranked lists**: given a query, the system returns items in some order; we score how well the *true relevant items* are placed at the top. Every metric here is a different way to weight rank position.

[← Home](../index.md) · [Interview deep-dive →](./interview-deep-dive.md)

## The retrieval setup

Retrieval metrics in TorchMetrics need **three** tensors, not two:
```python
preds    # (N,)  — model scores
target   # (N,)  — binary relevance (0 or 1) per item
indexes  # (N,)  — which query each item belongs to
```
Items with the *same `indexes` value* form one ranked list. The metric is computed *per query* then averaged. **Forgetting `indexes` is the #1 retrieval bug** — the metric will pool all items across queries into one giant list and report an inflated number.

For graded relevance (NDCG), `target` can be a non-negative float (e.g., 0/1/2/3 grades).

---

## Top-K vs full-list metrics

Some metrics take `top_k=` and only consider the first k positions; others use the full list. The user-facing system shows top-k, so you usually want the metric pinned to that k.

---

### RetrievalNormalizedDCG / NDCG (`tm.retrieval.RetrievalNormalizedDCG`)

**What it computes.** `DCG@k = Σ_{i=1..k} (2^{rel_i} − 1) / log2(i+1)`, then `NDCG@k = DCG@k / IDCG@k` (normalised by the ideal ranking).

**Intuition.** Discounts gain by log of position — being at rank 1 is much more valuable than at rank 10. Normalised so 1.0 = perfect ranking, regardless of how many relevant items there are.

**Range / direction.** `[0, 1]`. Higher better.

**When to use.** Anything with **graded relevance** (e-commerce search where some results are very relevant, some moderately). Or binary relevance with deep ranked lists.

**Real-world scenario.** Search ranker leaderboard at an e-commerce site: relevance grades are `{0=irrelevant, 1=substitute, 2=related, 3=exact match}`. NDCG@10 is the offline KPI; A/B tests confirm that 0.01 NDCG ↔ ~1% revenue.

**Code.**
```python
from torchmetrics.retrieval import RetrievalNormalizedDCG
ndcg = RetrievalNormalizedDCG(top_k=10)
ndcg(preds, target, indexes=indexes)
```

**Pitfalls.**
- Two NDCG formulas exist — Jarvelin-Kekalainen 2002 (`(2^rel − 1)`) vs Burges 2005 (`rel`). TorchMetrics uses the first (the Kaggle / RecSys standard). Be explicit when comparing across libs.
- `top_k=None` uses all items; for very long lists this dilutes the metric — set `top_k` to the user-facing depth.

---

### RetrievalMAP / MAP (`tm.retrieval.RetrievalMAP`)

**What it computes.** Mean Average Precision: per query, AP = `(1/R) Σ_k P@k · 1[item_k is relevant]`, where R = total relevants. Then mean over queries.

**Intuition.** AP rewards getting *all* relevants high in the list — discount per position is 1/k. Sums precision-at-each-relevant-hit; normalised by R so each query weighs equally.

**Range / direction.** `[0, 1]`. Higher better.

**When to use.** Binary relevance, deep retrieval, multi-relevant-per-query (e.g., document retrieval where many documents are relevant).

**Real-world scenario.** TREC-style document retrieval evaluation: per query, dozens of relevant docs hidden in a corpus of millions. MAP is the gold standard — researchers/ books report MAP and you should match.

**Pitfalls.**
- AP for queries with **zero** relevants is undefined. TorchMetrics' `empty_target_action="neg"` (default) treats them as 0 — pulls down your MAP. Use `"skip"` to ignore those queries when reporting.

---

### RetrievalMRR / Mean Reciprocal Rank (`tm.retrieval.RetrievalMRR`)

**What it computes.** Per query, MRR = `1 / rank_of_first_relevant`. Mean over queries.

**Intuition.** A single number that rewards getting *something* relevant at the top — but ignores subsequent relevants. Steeper discount than NDCG.

**Range / direction.** `[0, 1]`. Higher better. 0 if no relevant in the list.

**When to use.** Question-answering, navigational search ("find the official Apple support page") — only the first hit matters.

**Real-world scenario.** Voice-assistant search where the user wants *one* answer ("when is my package arriving?"). Top-1 success is the only thing that matters — MRR (especially MRR@1, equivalent to top-1 recall) is the deployment KPI.

**Pitfalls.**
- MRR is brutal: if the only relevant item is at rank 11 and you set `top_k=10`, the metric is 0. Make sure top_k matches user-facing depth.

---

### RetrievalHitRate (`tm.retrieval.RetrievalHitRate`)

**What it computes.** Per query, 1 if any of top-k is relevant, else 0. Mean over queries.

**Intuition.** "Did we surface anything useful?" Loosest binary signal.

**Range / direction.** `[0, 1]`. Higher better.

**Real-world scenario.** Recommender system top-10 carousel: did the user click *any* item? Hit-rate@10 maps directly to "engagement at all" — coarse but interpretable.

**Pitfalls.**
- Doesn't distinguish ranking quality within top-k. A model that puts the relevant at rank 1 and one at rank 10 both score 1.0. Use NDCG/MRR for finer signal.

---

### RetrievalRPrecision (`tm.retrieval.RetrievalRPrecision`)

**What it computes.** Precision at k = R, where R is the number of true relevants for the query.

**Intuition.** Per query, you have R relevants; how many of the top-R results are actually relevant? An *adaptive* precision-at-k.

**Real-world scenario.** TREC-style evaluation when query difficulty varies — easy queries have many relevants (R=50), hard queries have few (R=2). R-precision auto-adjusts.

---

### RetrievalPrecision / RetrievalRecall (`tm.retrieval.RetrievalPrecision`, `RetrievalRecall`)

**What they compute.** Standard precision and recall, but per query and only over the top-k.

**Real-world scenario.** Top-10 search: precision@10 = "fraction of the 10 shown that are relevant." Recall@10 = "fraction of all relevants that appeared in top 10." Both averaged over queries.

**Pitfalls.**
- Recall@k is bounded by `min(1, k/R)` — for queries with R=100 and k=10, max recall is 0.1. Don't compare recall@k across queries with very different R.

---

### RetrievalFallOut (`tm.retrieval.RetrievalFallOut`)

**What it computes.** False-positive rate over the top-k: `non_relevant_in_topk / total_non_relevant`.

**Intuition.** "How polluted is the top-k with irrelevant items, relative to the irrelevant universe?" Lower better.

**Real-world scenario.** Information-retrieval research where the corpus is huge — even a 1% fall-out rate floods the top with irrelevants.

---

### RetrievalAUROC (`tm.retrieval.RetrievalAUROC`)

**What it computes.** AUROC computed *per query* and averaged.

**Intuition.** Standard AUROC except per-query, so each query weighs equally regardless of size. Threshold-free ranking quality at the query level.

**Real-world scenario.** Per-user product ranker where each user has very different number of impressions; per-query (per-user) AUROC normalises.

---

### RetrievalPrecisionRecallCurve (`tm.retrieval.RetrievalPrecisionRecallCurve`)

**What it computes.** Per query, the full PR curve, averaged across queries.

**Real-world scenario.** Depth-vs-precision study for picking the user-facing top-k. Plot the curve, find the elbow.

---

## Quick-reference: which retrieval metric for which scenario?

| Scenario | Primary | Secondary |
|---|---|---|
| E-commerce search (graded relevance) | NDCG@10 | MAP@10 |
| QA / navigational search | MRR | Hit@1 |
| Recsys carousel (binary clicks) | Hit@k | NDCG@k |
| TREC/document retrieval | MAP | NDCG |
| Per-query AUROC ranking | RetrievalAUROC | NDCG |
| Variable-difficulty queries | R-Precision | MAP |
| Coarse "did we get *anything*" | Hit@k | Recall@k |

---

[Continue → interview deep-dive](./interview-deep-dive.md)
