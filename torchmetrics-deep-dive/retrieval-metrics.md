---
title: Retrieval Metrics
nav_order: 7
---

# Retrieval Metrics

Information-retrieval and ranking metrics live under `torchmetrics.retrieval`. They differ from classification metrics in one big way: a sample isn't a single `(pred, target)` pair — it's a **query** with a *list of items* and a *relevance per item*. The `indexes=` argument is what tells the metric "these items belong to query 7."

---

## The retrieval contract

Every retrieval metric expects three tensors per `update`:

```python
metric.update(preds, target, indexes)
```

| Argument | Shape | Meaning |
|---|---|---|
| `preds`   | `(N,)` float | Score / probability for each item. |
| `target`  | `(N,)` int   | Relevance (binary 0/1 or graded for NDCG). |
| `indexes` | `(N,)` int   | The query id each item belongs to. |

The metric will:

1. Group rows by `indexes` (this is why list state is required).
2. Sort each group by `preds` descending.
3. Compute the per-query metric.
4. Aggregate (mean by default).

---

## Available metrics

| Metric | What it measures |
|---|---|
| `RetrievalMRR` | Mean Reciprocal Rank — `1 / rank_of_first_relevant`. |
| `RetrievalMAP` | Mean Average Precision — area under the per-query PR curve. |
| `RetrievalNDCG` | Normalized DCG — graded relevance, position-discounted. |
| `RetrievalPrecision` | Precision @ k. |
| `RetrievalRecall` | Recall @ k. |
| `RetrievalRPrecision` | Precision at rank R (R = total relevant). |
| `RetrievalHitRate` | Did *any* relevant item appear in top-k? |
| `RetrievalFallOut` | False-positive rate among non-relevant items. |
| `RetrievalAUROC` | AUROC computed per-query, then averaged. |
| `RetrievalNormalizedDCG` | NDCG @ k. |

All of them accept a `top_k=` parameter so you can report `MRR@10` etc.

---

## Why is the `indexes=` argument needed?

In classification, every prediction is independent. In retrieval, the *position* of an item within its query is what matters. NDCG @ 10 for query A says nothing about NDCG @ 10 for query B — they're computed independently and averaged.

Without `indexes`, the metric would treat the whole batch as one giant query, which is almost never what you want.

```python
preds   = torch.tensor([0.9, 0.1, 0.8, 0.7])
target  = torch.tensor([1,   0,   0,   1  ])
indexes = torch.tensor([0,   0,   1,   1  ])

# Query 0: predicted [0.9, 0.1] for [1, 0]   → MRR = 1.0
# Query 1: predicted [0.8, 0.7] for [0, 1]   → MRR = 0.5
# Mean MRR = 0.75
```

---

## How aggregation across queries works

By default, retrieval metrics return the **mean across queries**. You can change this with `aggregation="mean" | "median" | "min" | "max"`. For long-tail evaluation sets, the *mean* MRR is misleading — most platforms also track per-query histograms.

If a query has **no relevant items**, what should the metric do? Behavior is configurable via `empty_target_action="neg" | "pos" | "skip" | "error"`:

- `"neg"` — count as zero (the metric is "0" for that query).
- `"pos"` — count as one.
- `"skip"` — drop the query from the average. (Most fair when ground truth is sparse.)
- `"error"` — raise.

Pick deliberately. A silent default can change your reported number by a few percent.

---

## NDCG specifics

NDCG @ k = DCG @ k / IDCG @ k, where:

- DCG = Σ (2^rel_i − 1) / log2(i + 1)
- IDCG = same on the *ideal* (best-possible) ordering.

`RetrievalNormalizedDCG` supports **graded relevance** — `target` doesn't have to be 0/1. Pass relevance scores like `[0, 1, 2, 3]` and the metric uses the gain formula above. For binary relevance use `RetrievalNDCG` (special case).

---

## When to use which

| Scenario | Recommended metric |
|---|---|
| Web search relevance | NDCG @ 10 (graded) + MRR. |
| Recommender homepage | Recall @ 10, NDCG @ 10. |
| Question answering | MRR. |
| Image retrieval | Recall @ 1 / 5 / 10 + MAP. |
| Ad ranking | Precision @ k weighted by revenue (custom). |
| RAG passage retrieval | MRR + Recall @ 5; you'll need both. |

---

## Production traps

1. **Indexes must be sorted contiguously per query** *or* the metric must support unsorted input. Recent versions handle both, but the cost differs. Check your version.
2. **Eval-set imbalance** — if 90 % of queries have 1 relevant item and 10 % have 50, a *mean* MRR is dominated by the easy ones. Always inspect per-bucket histograms.
3. **Cold-start queries** — queries with no relevant items should be handled with `empty_target_action="skip"` if your goal is "performance on answerable queries."
4. **DDP** — retrieval metrics use list states (must `cat` preds/targets/indexes across ranks). Memory grows linearly with eval set. Use `compute_on_cpu=True` if you have many small queries.

---

## Reading the source

Retrieval metrics share machinery via `torchmetrics.retrieval.base.RetrievalMetric`. Their `update()` simply appends `(preds, target, indexes)` triples to list states; the heavy lifting happens in `compute()`, which:

1. Sorts `indexes`, then groups.
2. For each group, calls a per-query function from `torchmetrics.functional.retrieval` (e.g. `retrieval_reciprocal_rank`).
3. Stacks the per-query values and aggregates.

This pattern — **list state + per-group reduce in `compute`** — is the canonical recipe for any retrieval-style metric, including any custom one you might write.
