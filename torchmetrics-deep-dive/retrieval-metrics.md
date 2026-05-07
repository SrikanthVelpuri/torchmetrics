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

---

## Interview Drill-Down (multi-level follow-ups)

### Q1. NDCG vs MAP — when do you pick which?

> Both are "averaged over queries" metrics, but they handle the relevance signal differently. **MAP** assumes binary relevance and integrates over precision-at-each-relevant-rank. **NDCG** handles graded relevance via the discount factor and the gain formula. **Graded ⇒ NDCG. Binary ⇒ either, but NDCG is more flexible.**

  **F1.** Why is the NDCG discount factor `1/log2(rank+1)` and not something else?

  > It's empirical, calibrated against user click curves. `log2` discount falls fast at the top (rank 1 → discount 1, rank 2 → 0.63, rank 3 → 0.5) and slowly at the tail. You can change it for your domain — the math still works for any monotonically decreasing discount.

    **F1.1.** Could you use a different discount based on actual click data?

    > Yes — fit a click-position curve to your platform's logs and use that as the discount. Custom metric: subclass `RetrievalNDCG` and override the gain accumulator. The metric is still well-defined; the literature value just won't be comparable.

      **F1.1.1.** What about position bias in the click data you use to fit the discount?

      > Click data has *positional* bias (top results are clicked because they're top). Correct via inverse-propensity-scoring before fitting, or use interleaving experiments to estimate true relevance independently of position.

### Q2. Recall@1000 vs Recall@10 — what's the architectural significance?

> They evaluate **different stages** of a retrieval pipeline. Recall@1000 evaluates **first-stage candidate generation** — does the (cheap) ANN/BM25 retriever bring relevant items to the reranker's attention? Recall@10 evaluates the **reranker** — given the candidate set, is the top-10 right?

  **F1.** What's a good first-stage recall target?

  > For a 1k-candidate reranker, you want recall@1000 ≥ 95-98 %. Below 90 %, the reranker is bottlenecked by what got recalled. The reranker can't fix items it never sees.

    **F1.1.** What if the first stage hits 99 % recall@1000 but the reranker only achieves 60 % NDCG@10?

    > Reranker is the bottleneck. Invest in reranker capacity / features. The metric layer correctly identified the layer to fix.

      **F1.1.1.** How would you decide the right candidate-set size K?

      > Sweep K = {100, 500, 1000, 5000}; compute recall@K and reranker latency. Pick the smallest K where recall plateaus *and* latency fits the budget. Often the recall curve has a knee at K ≈ 1000.

### Q3. The `empty_target_action` argument — what's the right default?

> No universal default. `"skip"` is the most defensible (don't penalize for queries with no ground-truth-positive). `"neg"` makes the metric stricter (a query with no relevant item gets metric = 0). `"pos"` is rarely correct.

  **F1.** When is `"skip"` wrong?

  > When "no relevant items" is a valid signal. Example: spam detection retrieval — a query with no spam should result in zero spam pulled. Reporting that as "skipped" hides the system's correct behavior.

    **F1.1.** How do you communicate the choice in your dashboards?

    > Always include the *empty-target rate* (fraction of queries skipped) alongside the metric. A jump in skip rate is a leading indicator of distribution shift.

      **F1.1.1.** Implement that as a metric?

      > Custom metric. Two states: `n_skipped` and `n_total`, both tensor sum-reduced. `compute = n_skipped / n_total`. Add it to your retrieval `MetricCollection`.

### Q4. Why is per-query aggregation `mean` by default? When isn't it right?

> Mean is fair when queries are statistically similar. It's misleading on a long-tail distribution where a few high-traffic queries dominate. Then prefer **traffic-weighted mean** or **per-bucket histograms**.

  **F1.** How do you implement traffic-weighted retrieval metrics?

  > Pass query-level weights into a custom metric. State: `sum_weighted_metric` and `sum_weights`. `update()` weights the per-query metric by its traffic. `compute()` divides. DDP-correct via sum reduction.

    **F1.1.** What if you want both un-weighted and weighted versions on the same dashboard?

    > `MetricCollection({"ndcg_unweighted": RetrievalNDCG(), "ndcg_weighted": WeightedRetrievalNDCG(...)})`. Both consume the same input; collection's compute-groups won't share state because their states differ — that's fine, the cost is just slightly more memory.

      **F1.1.1.** Could compute groups share state if you were clever?

      > Yes — make both metrics inherit from the same base that stores `(preds, target, indexes, weights)` lists, and override only `compute`. Then `MetricCollection` will detect identical state and compute groups will kick in.
