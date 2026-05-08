---
title: Retrieval metrics — interview deep dive
---

# Retrieval metrics — interview deep dive

[← Family page](./index.md) · [← Home](../index.md)

---

## Q1. "NDCG vs MAP — when to use which?"

**Answer.** NDCG handles **graded relevance** (0,1,2,3) — items are not just relevant or irrelevant. MAP is for **binary** relevance. Both reward putting relevants high; NDCG's discount is logarithmic (`1/log2(i+1)`), MAP's is linear (`1/k` at each relevant). NDCG is the e-commerce / web-search standard because grades are natural; MAP is the IR / TREC standard because relevance judgements there are binary.

> **F1.1** "Why use NDCG even when relevance is binary?"
>
> **Answer.** Two reasons: (1) the log discount in NDCG closely tracks user-attention curves on web pages — empirically tuned to user behaviour. (2) Normalisation by IDCG removes per-query scale; in MAP, queries with R=2 vs R=50 contribute differently. NDCG's per-query bound is always 1.0.

> **F1.2** "If model A is +0.01 NDCG over model B, is it better?"
>
> **Answer.** Need a CI. NDCG variance across queries is large; per-query NDCG difference + paired Wilcoxon is the rigorous test. Production rule of thumb at major engines: ≥ 0.005 NDCG with significance is shippable; smaller, A/B-test it before relying on offline.

---

## Q2. "What's the most common bug with retrieval metrics in TorchMetrics?"

**Answer.** **Forgetting the `indexes=` argument.** Without it, the metric pools all items across queries into one giant ranked list and reports a number that's neither per-query nor pooled correctly. The first symptom is "my NDCG is suspiciously high" — because all relevants from all queries cluster at high scores, and the metric thinks one giant query is well-ranked.

> **F2.1** "How do you `indexes=` if your data has variable items per query?"
>
> **Answer.** Construct it as a flat tensor where `indexes[i]` = the query ID of `preds[i]`. Repeated values are fine and expected. Example: query 0 has 5 items, query 1 has 3 → `indexes = [0,0,0,0,0,1,1,1]`.

---

## Q3. "Recsys ranks 1M items per user. Computing NDCG over all 1M is slow. What do you do?"

**Answer.** Two strategies:
1. **NDCG@k**, with k matching the user-facing depth. NDCG@10 is computed in O(k log k) per query.
2. **Sub-sample negatives**: per query, keep all positives + N random negatives (N=100). NDCG over the sample is unbiased estimator of recall@k, with massively reduced compute.

The interview signal here is recognising that retrieval at scale needs sampling, and articulating the bias/variance trade-off.

---

## Q4. "MRR is brutal — explain."

**Answer.** MRR = `1/rank_of_first_relevant`. If the only relevant item is rank 11 and `top_k=10`, MRR = 0. So MRR is only meaningful when (a) you set `top_k` to match the user-facing depth, and (b) the system is *expected* to find the relevant in the top-k. For QA / navigational search, this is the right metric. For broad search, it's too narrow — use NDCG.

> **F4.1** "When does MRR rank models the same as Recall@1?"
>
> **Answer.** When the only relevant per query is rank 1, MRR = 1; otherwise MRR = 0 if `top_k=1`. So MRR@1 = Recall@1 = Hit@1 when there's at most one relevant per query. They diverge when MRR has its full 1/rank discount over many positions.

---

## Q5. "Build a metric stack for an e-commerce search ranker."

**Answer.**
```python
metrics = MetricCollection({
    "ndcg@10":  RetrievalNormalizedDCG(top_k=10),
    "ndcg@30":  RetrievalNormalizedDCG(top_k=30),
    "mrr@10":   RetrievalMRR(top_k=10),
    "hit@1":    RetrievalHitRate(top_k=1),
    "hit@10":   RetrievalHitRate(top_k=10),
    "map@10":   RetrievalMAP(top_k=10),
    "p@10":     RetrievalPrecision(top_k=10),
})
```
**Why each**:
- NDCG@10 — primary KPI matching the user-facing top-10.
- NDCG@30 — depth diagnostic; if @30 improves but @10 doesn't, the model is shuffling deep ranks.
- MRR@10 — captures "did we put one good thing at the top."
- Hit@1, Hit@10 — interpretable for stakeholders.
- MAP@10 — sanity check against MRR (rewards multiple relevants).
- P@10 — the literal "what fraction of shown items were relevant."

---

## Q6. "What does R-Precision capture that P@k does not?"

**Answer.** R-Precision = P@R, where R = number of relevants for *this query*. So query-difficulty is normalised: an easy query (R=50) is judged on top-50 precision; a hard query (R=3) on top-3. P@10 is the same depth for every query, which over-rewards easy queries (more relevants = more chances to land in top-10).

---

## Q7. "How do you handle queries with zero relevants?"

**Answer.** It depends on what they mean:
- **System bug** (relevance labels missing) — drop them; they're not signal.
- **Truly no relevant exists** (model should return nothing useful) — penalise with metric=0 to reward systems that abstain or return short lists.

TorchMetrics exposes `empty_target_action`:
- `"neg"` — treat as zero (default, conservative).
- `"pos"` — treat as one (rewards conservative models).
- `"skip"` — skip the query (assume label issue).
- `"error"` — raise.

The right pick depends on the data; in interviews, articulate the trade-off rather than guessing.

---

## Q8. "Recall@k > Precision@k always — why?"

**Answer.** Not always. They're related: `Recall@k = (# relevant in top-k) / R` and `Precision@k = (# relevant in top-k) / k`. So `Precision@k = Recall@k · R / k`. If `R < k`, Precision can exceed Recall (when most of the small relevant set is at the top). For most search queries with `R >> k`, recall is bounded by `k/R` while precision can hit 1.0 — so precision tends to be higher on hard queries.

---

## Q9. "Two retrieval models — A is better at NDCG@10, B is better at NDCG@30. Pick."

**Answer.** Depends on the user-facing depth. If users only see top-10 (search results page), ship A. If users page through (deep retrieval, scrolling), ship B. The right move in an interview: ask about the user-facing experience before picking. The metric difference is informing the decision; the user is making it.

---

## Q10. "How does TorchMetrics handle DDP for retrieval metrics?"

**Answer.** State is `(preds, target, indexes)` lists per rank; `_sync_dist` uses `all_gather_object` to merge. After gather, the metric groups by `indexes` and computes per-query, then averages. Memory cost is O(N); a 100M-item global ranking gathered to one rank is expensive. For training tracking, sub-sample queries; full numbers at epoch end on rank 0.

---

## Q11. "When does NDCG ≠ NDCG (across libraries)?"

**Answer.** Three places:
1. **Gain function**: `(2^rel − 1)` (Jarvelin-Kekalainen, the standard) vs `rel` (Burges). TorchMetrics uses the first.
2. **Discount**: `1/log2(i+1)` is universal but some libs use `1/log2(i)` with `log2(1)=0` handled by `i+1` mapping — same thing.
3. **Normalisation**: TorchMetrics normalises by IDCG of the *given* relevance grades; some libs cap grades.

Always rerun with the canonical implementation if numbers don't match across teams.

---

## Q12. "Why is per-query MRR averaging better than pooled?"

**Answer.** Pooled MRR (mix all queries' first-relevant-ranks into one list) over-weights queries with many candidates. Per-query MRR averages, so each query is one data point. This is the same per-vs-pooled trade-off as macro vs micro F1.

---

## Q13. "User clicks on rank 3, ignores ranks 1 and 2 — was the ranker bad?"

**Answer.** Single click is noisy signal. The interview signal: distinguish offline metrics from online behaviour. Offline NDCG measures *judged* relevance; user clicks measure *attention × relevance × snippet quality*. Always couple offline metrics with online experiments. A model that improves NDCG by 1% but reduces clicks-on-rank-1 by 5% has issues with snippet/title — not the ranker per se.

---

## Q14. "How do you measure ranking quality when the relevance is implicit (clicks)?"

**Answer.** Two patterns:
- **Counterfactual evaluation**: use logged-bandits techniques (IPS-weighted retrieval metrics) to debias clicks for position bias.
- **Pairwise preferences**: train a model to rank `clicked > not-clicked-but-shown-above` pairs. Evaluate with pairwise concordance.

Direct Hit/MRR on click data systematically rewards models that match the existing ranker's position bias.

---

## Q15. "What does it mean for two models to have identical NDCG but different precision@1?"

**Answer.** Means they place relevants at different *depths* but the discount-weighted total is the same. Model A: rank 1 = relevant, rank 10 = irrelevant. Model B: rank 1 = irrelevant, rank 2 = relevant, rank 5 = relevant. Same NDCG (potentially), very different P@1. P@1 is what users feel first; ship A unless deeper coverage is the goal.

---

## Q16. "How do you A/B test a search ranker?"

**Answer.** Two-step:
1. **Offline gating**: NDCG@10 with significance test on held-out judged queries. Floor: must not regress beyond −0.005.
2. **Online**: split traffic, measure click-through, time-to-click, conversion, abandonment. Run for 2+ weeks to absorb day-of-week. Pre-commit metrics before launching the test.

The trap: optimising for offline NDCG without online checks. Offline often disagrees because NDCG measures judged relevance, not user behaviour.

---

## Q17. "What's the simplest baseline for a retrieval system?"

**Answer.** BM25 (exact-keyword ranking with TF-IDF normalisation). Compute NDCG/MAP/MRR for BM25, treat that as the floor. Any neural retrieval that doesn't beat BM25 by ≥ 5% NDCG offline is suspect — usually a training/serving mismatch.

---

[← Back to family page](./index.md)
