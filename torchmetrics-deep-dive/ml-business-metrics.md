---
title: ML-to-Business Metrics
nav_order: 18
---

# Connecting ML Metrics with Business Metrics

Every senior interviewer will eventually ask a version of:

> "Your model has 0.92 F1. So what?"

If you can't translate model quality into dollars, customer satisfaction, or operational outcomes, you don't get the staff role. This page is the **translation manual**: ML metric → business metric, with concrete examples for **American Airlines** (transportation / revenue management) and **Amazon** (e-commerce / recommendations / fraud / fulfillment).

Each scenario shows: the business KPI, the chosen ML metric, the TorchMetrics setup, and the multi-level interview drill-down you should expect.

---

## The translation framework

There are only four bridges from a model number to a P&L number.

| Bridge | What you compute | Example |
|---|---|---|
| **Cost-of-error matrix** | Map TP/FP/TN/FN to dollars. | Fraud: a missed fraud (FN) costs the average chargeback; a blocked legit transaction (FP) costs the gross-margin of that order. |
| **Top-k truncation** | Business shows only K items; care about quality of those K. | Recommendations, search, ranking. |
| **Threshold operating point** | Pick a threshold to hit a desired precision or recall. | Disease screening at 95 % recall; fraud at 1 % FPR. |
| **Calibration** | Use the predicted probability *as a number* downstream (pricing, bidding). | Dynamic pricing, ad bidding — a 10 % miscalibrated probability is a 10 % revenue leak. |

A good interview answer always says **which bridge** it's using and **why**.

---

## American Airlines — Transportation & Revenue Management

Airlines run thin margins (~3-5 %). Every percent of forecast error or no-show miss flows almost directly to the bottom line. Here's how each ML use-case maps.

---

### AA-1. Flight-delay prediction (binary classification → ops planning)

**The business goal.** Pre-stage crews, gate agents, and connecting passenger rebooking when a flight is *likely* to delay. Reduce missed connections (DOT-reportable, ~$300 each) and IROPS (irregular ops) cascading cost.

**ML metric** — `BinaryF1Score` *and* `BinaryAUROC` at the chosen operating point, plus per-hub segmentation.

**Why these.** The cost matrix is asymmetric:

- FN (predicted on-time, actually delayed) → unstaffed gate, missed connections, hotel vouchers.
- FP (predicted delay, actually on-time) → wasted ramp labor, inflated reserve costs.

You set the threshold to maximize **expected dollar value** = `P(delay) × cost_FN_avoided − P(on-time) × cost_FP_added`. AUROC is reported alongside so leadership can see threshold-free quality even if the operating point shifts seasonally.

**TorchMetrics setup**

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryF1Score, BinaryAUROC, BinaryAveragePrecision,
    BinaryCalibrationError, BinaryRecallAtFixedPrecision,
)

delay_metrics = MetricCollection({
    "f1":     BinaryF1Score(threshold=0.42),    # tuned operating point
    "auroc":  BinaryAUROC(),
    "ap":     BinaryAveragePrecision(),
    "ece":    BinaryCalibrationError(n_bins=15),
    "rec@p90": BinaryRecallAtFixedPrecision(min_precision=0.90),
}).to(device)

# Per-hub breakdown (DFW, CLT, ORD, PHX, MIA, JFK, LGA, DCA, LAX, PHL)
per_hub_metrics = {
    hub: delay_metrics.clone(prefix=f"hub_{hub}/")
    for hub in HUBS
}
```

**Why calibration matters here.** Gate planners look at the *probability number*, not just a binary label — "this flight has 35 % delay risk" triggers different action than "this flight has 80 %." Miscalibrated probabilities mean wasted reserves on the high end and surprised ops on the low end.

#### Interview drill-down

**Q.** Why F1 and not just accuracy?

> A. ~80 % of flights are on-time. Accuracy of 0.80 is the trivial "predict on-time always" baseline. F1 forces both precision and recall on the rare delay class.

  **F1.** Why not just optimize recall, since a missed delay (FN) is more expensive?

  > Because over-recall floods ops with false alarms, eroding their trust and causing them to ignore the model. The right answer is `RecallAtFixedPrecision(min_precision=0.85)` — keep precision above the trust threshold, then maximize recall.

    **F1.1.** What if the precision constraint cannot be met by your model?

    > Either the feature set is insufficient (add weather feeds, upstream-delay propagation features) or the threshold is too aggressive. As a stopgap, segment the model: high-confidence delay calls (P > 0.7) go to ops; medium-confidence (0.4 < P < 0.7) go to a softer "monitor" queue. Two operating points instead of one.

      **F1.1.1.** How would you measure the value of the "monitor" queue?

      > A/B test the queue against historical no-action data. The right metric is **dollar-weighted lift** = (cost_avoided_with_action − cost_avoided_no_action). It's not in TorchMetrics out of the box; you build it as a custom metric whose state is `sum_dollars_saved` and `count`, reduced by `sum`.

**Q.** A hub VP wants delay-risk for *individual* flights, not aggregated.

  **F1.** What's the per-flight metric?

  > Brier score (mean squared error on probabilities) is the gold standard for individual-flight calibration. In TorchMetrics: `MeanSquaredError()` over `(P_predicted, label_observed)` pairs.

    **F1.1.** Why not log-loss?

    > Log-loss penalizes confident wrong calls more heavily — that's actually what you *don't* want at the per-flight level, because hub VPs want a probability they can act on, not one that's been pulled toward 0.5 by penalty pressure. Brier is the more interpretable choice.

---

### AA-2. No-show prediction & overbooking (probabilistic regression → revenue)

**The business goal.** Sell more seats than the cabin holds because some passengers won't show up. Overbook by exactly the right amount: too few = empty seats (revenue lost); too many = denied boarding compensation (DBC) ~ $500-2000 per displaced passenger.

**ML metric.** This is a *probabilistic forecasting* problem. The right metrics are:

- **Quantile loss / pinball loss** at the relevant quantile (you want P95 of show-up count, not the mean).
- **CRPS (Continuous Ranked Probability Score)** for the full distribution.
- **Coverage** (does the predicted P95 actually contain truth 95 % of the time?).

**TorchMetrics setup**

```python
from torchmetrics.regression import (
    ContinuousRankedProbabilityScore,
    MeanSquaredError,
    MeanAbsoluteError,
)
# coverage is a custom metric (see custom-metrics.md)
```

**Business → ML mapping**

| Business outcome | ML signal |
|---|---|
| Empty seats (under-booked) | Predicted no-show count > actual; loss = unsold seats × yield. |
| Denied boarding (over-booked) | Predicted no-show count < actual; loss = DBC × passengers displaced. |
| Net contribution | Σ (revenue_recovered_from_overbooking − DBC_paid). |

**Operating-point logic.** You don't optimize CRPS directly for production — you pick the *quantile* whose threshold maximizes net contribution given the asymmetric cost. Typical: P85 to P95 depending on route (DBC cost is higher on long-haul transcons because rebooking is harder).

#### Interview drill-down

**Q.** Why quantile loss instead of MSE?

> A. MSE forces a model to predict the conditional mean. The mean isn't where you want to operate — you want the upper tail because the cost of being slightly wrong on the high side (DBC) far exceeds the cost on the low side (one empty seat).

  **F1.** What's the gradient flow during training look like?

  > For pinball loss at quantile τ, the gradient is `-τ` if `y > ŷ` and `(1-τ)` if `y < ŷ`. The asymmetry is the whole point — you can't get this from MSE.

    **F1.1.** TorchMetrics doesn't ship pinball loss as a metric class — how do you add it?

    > Subclass `Metric`. Two states: `sum_loss` (tensor, sum reduction) and `n` (tensor, sum). `update` computes `pinball(y, ŷ, tau)` and accumulates. `compute` divides. About 15 lines including DDP correctness.

      **F1.1.1.** How would you make this metric report multiple quantiles in one pass?

      > Make `tau` a tensor of length K. Both states become K-dim tensors. The reduction (`sum`) still works elementwise. Output is a length-K vector you can name with `ClasswiseWrapper(metric, labels=[f"q{int(t*100)}" for t in tau])`.

**Q.** A new revenue-management lead asks for one number to put in the weekly review.

  **F1.** What's that number?

  > Net overbooking contribution = revenue_recovered − DBC_paid, summed across all flights in the week. *Not* a TorchMetrics number — it's a P&L line. The TorchMetrics number that *correlates* with it is the metric tracker over CRPS, but you don't show CRPS to the executive.

    **F1.1.** What about routes with extreme variance (e.g. stadium charters, cruises)?

    > Segment. Run separate metrics per route family. A high-variance route's CRPS will look bad globally even if the model is doing what it can. `MetricCollection` per route family + `MetricTracker` per metric solves this without hand-rolled bookkeeping.

---

### AA-3. Dynamic pricing / yield management (regression with calibration → revenue)

**The business goal.** Set fares per origin-destination per fare-class per departure-day to maximize total revenue subject to capacity and competitor constraints.

**ML signal.** Demand-elasticity model: predict `P(book at price p)` for each origin-destination-day, given historical bookings, search-to-book ratios, and competitor pricing. The decision layer optimizes expected revenue.

**Critical metric.** **Calibration** of `P(book)` directly translates to revenue because the price optimizer multiplies probability by price. A 5 % calibration miss compounds across thousands of OD pairs.

```python
from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

booking_metrics = MetricCollection({
    # On the probability output:
    "ece":      BinaryCalibrationError(n_bins=20),
    "brier":    MeanSquaredError(),     # = Brier score on probabilities
    # On the demand-curve output:
    "demand_mae":  MeanAbsoluteError(),
})
```

#### Interview drill-down

**Q.** Why is calibration the right primary metric for pricing?

> A. The optimizer takes `E[revenue] = price × P(book at price)`. If `P(book)` is biased high by 10 %, expected revenue is biased high by 10 %, and the chosen price is too high → bookings drop. Calibration error directly measures the bias.

  **F1.** Will *better calibration* always lead to *more revenue*?

  > No. The mapping is monotonic only if the optimizer respects probability monotonically (it usually does, with a price elasticity prior). A calibrated but **lower-resolution** model — one that predicts the same P everywhere — would be useless even at perfect calibration. You need both calibration **and** discrimination (AUROC).

    **F1.1.** How do you measure the trade-off between calibration and discrimination?

    > Decompose the Brier score into reliability (calibration), resolution (discrimination), and uncertainty (irreducible). Brier = uncertainty − resolution + reliability. A model with low reliability and low resolution is worthless; you want low reliability **and** high resolution.

      **F1.1.1.** Implement that decomposition with TorchMetrics?

      > Custom metric with three accumulators: `bin_count`, `bin_pos_sum`, `bin_pred_sum`. After accumulation, you have per-bin observed and predicted rates → reliability. Resolution comes from the bin-mean-vs-overall-mean spread. Uncertainty is `p̄(1-p̄)`. Roughly 40 lines.

**Q.** Two pricing models are live in shadow mode. How do you pick a winner?

  **F1.** Per-OD revenue lift, not aggregate.

  > Aggregate revenue is dominated by hub-hub ODs (DFW-LAX, CLT-LGA). A model that's worse on those but better on long-tail ODs would be punished even though long-tail is where pricing flexibility lives. Per-segment metrics + a paired bootstrap on revenue gives the honest answer.

    **F1.1.** TorchMetrics support for paired bootstraps?

    > `BootStrapper(metric, num_bootstraps=1000)` does unpaired by default. For paired, you write a custom wrapper that maintains synced indices across both models. Worth 30 minutes of engineering for the clean answer.

---

### AA-4. Cancellation prediction / IROPS recovery (classification + retrieval blend)

**Business goal.** During disruptions, automatically rebook stranded passengers onto alternative itineraries. The rebooker is a **ranking model** over candidate alt-itineraries scored by `P(passenger accepts)`.

**ML metric.**

- For the cancellation classifier: same as delay (BinaryF1, AUROC, calibration, hub segmentation).
- For the rebooker: `RetrievalNDCG@5`, `RetrievalRecall@5`, plus a custom **dollar-weighted MRR** (rank of the eventually-chosen itinerary, weighted by ticket value).

```python
from torchmetrics.retrieval import RetrievalNDCG, RetrievalMRR, RetrievalRecall

rebooker_metrics = MetricCollection({
    "ndcg@5":   RetrievalNDCG(top_k=5),
    "mrr":      RetrievalMRR(empty_target_action="skip"),
    "recall@5": RetrievalRecall(top_k=5),
}).to(device)

rebooker_metrics.update(scores, labels, indexes=passenger_id)
```

#### Interview drill-down

**Q.** Why NDCG and not classification accuracy on "did we offer the right itinerary"?

> A. Multiple "right" itineraries exist (any acceptable one resolves the disruption), and being slightly off — putting the chosen itinerary at rank 2 instead of 1 — is much less bad than putting it at rank 8. NDCG captures that gradient.

  **F1.** Why discount factor `1 / log2(rank+1)` specifically?

  > Empirical: it matches user click curves on ranked lists across many domains. You can change the discount in the gain formula if your business has a stronger/weaker top-of-list preference.

    **F1.1.** What if you have **graded** preferences (e.g. business-class passenger prefers same-cabin rebook 5×)?

    > Use graded NDCG: `target` is no longer 0/1 but the relevance grade. `RetrievalNormalizedDCG` accepts non-binary targets directly. Now your "5" passes through the gain formula `(2^rel - 1)`.

      **F1.1.1.** How do you set the grades without leaking the label?

      > Grades come from *deterministic business rules* — passenger fare-class, cabin, frequent-flyer status, original arrival time. Anything derived from "would they accept this offer" is a leak.

---

### AA-5. Customer-loyalty churn (binary classification with long horizon)

**Business goal.** Identify AAdvantage members at risk of switching to United/Delta and trigger retention offers (status match, bonus miles, hotel credits).

**ML metric.** Binary classification with **time-discounted recall**: a true churn caught 6 months early is worth more than one caught 1 month early.

```python
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
# time-discounted-recall is a custom metric
```

#### Interview drill-down

**Q.** Why AUROC and AP, not accuracy?

> A. Churn is a rare event (~ 5-10 % annual). Accuracy of 0.92 is below the trivial baseline. AP and AUROC both ignore the prevalence in the trivial baseline sense.

  **F1.** AUROC vs AP for a rare event?

  > AP penalizes false positives in proportion to recall (it's the area under the PR curve). AUROC integrates over FPR, which is dominated by the negative class so it can stay high even with terrible PR. **Use AP as the primary; AUROC as the diagnostic.**

    **F1.1.** What's a good "trigger threshold" for retention offers?

    > Cost-driven. Offer cost ≈ $50 (status match expense), retained customer LTV ≈ $400. Retention offer is worth firing when `P(churn) × LTV_retained > offer_cost / lift_rate`. The actual threshold ends up around `P(churn) > 0.15` at typical lift rates of 30 %.

      **F1.1.1.** How do you estimate `lift_rate`?

      > A holdout group: randomly *don't* offer 5 % of triggered users; compare retention 6 months later. Lift rate = `(retention_offered − retention_holdout) / retention_holdout`. The holdout is the bridge from ML metric to business metric.

---

### AA-6. Demand forecasting for fleet planning (time-series regression)

**Business goal.** Decide how many seats per route per quarter, two years ahead. Wrong = either capacity-constrained or running half-full.

**ML metric.** **Weighted MAPE** (per-revenue-weighted) and **CRPS** for the probabilistic version.

```python
from torchmetrics.regression import (
    WeightedMeanAbsolutePercentageError,
    SymmetricMeanAbsolutePercentageError,
    NormalizedRootMeanSquaredError,
)

forecast_metrics = MetricCollection({
    "wmape":  WeightedMeanAbsolutePercentageError(),
    "smape":  SymmetricMeanAbsolutePercentageError(),
    "nrmse":  NormalizedRootMeanSquaredError(normalization="range"),
})
```

#### Interview drill-down

**Q.** Why wMAPE not MAPE?

> A. MAPE divides by `y`, which blows up for low-volume routes (a single tour-charter = "infinity error"). wMAPE divides total absolute error by total actual — robust on low-volume series, still scale-free across routes.

  **F1.** Why not RMSE?

  > RMSE is in passenger-units, which doesn't compare across routes. Cross-route comparability is critical for the planning team's prioritization.

    **F1.1.** A planner wants to see "how confident are we in this number"?

    > Wrap the forecast model in a quantile head and report **P10 / P50 / P90** spreads, evaluated with quantile loss / CRPS. Plot the spread on the dashboard. Coverage of the P90 interval becomes a tracking metric.

---

## Amazon — E-commerce, Recommendations, Fulfillment

Amazon's ML maps cleanly to four major workloads: **search ranking**, **recommendations**, **fraud / risk**, **demand forecasting / fulfillment**. Each has a distinct metric stack.

---

### AMZ-1. Product search ranking (retrieval)

**Business goal.** When a customer searches "bluetooth headphones," show items that maximize the joint of *relevance × revenue × Prime eligibility × seller quality × ad budget consumption*.

**ML metric stack.**

| Layer | Metric | Why |
|---|---|---|
| First-stage retrieval (recall) | `RetrievalRecall@1000` | Did we even pull the right products from the catalog? |
| Reranker (precision) | `RetrievalNDCG@10` | Quality of top-k that user sees. |
| Personalization | `RetrievalMRR` | First clicked / converted item rank. |
| Diversity / fairness | Custom (intra-list similarity) | Don't show 10 nearly-identical SKUs. |
| Revenue alignment | Revenue-weighted NDCG (custom) | Top results by margin × P(click). |

```python
from torchmetrics.retrieval import (
    RetrievalRecall, RetrievalNDCG, RetrievalMRR, RetrievalPrecision,
)

retrieval_stack = MetricCollection({
    "recall@1000": RetrievalRecall(top_k=1000),
    "ndcg@10":     RetrievalNDCG(top_k=10),
    "mrr":         RetrievalMRR(empty_target_action="skip"),
    "precision@5": RetrievalPrecision(top_k=5),
})
```

#### Interview drill-down

**Q.** Why two retrieval stages with different metrics?

> A. First-stage candidate generation has to *not lose* a relevant item — recall@K is brutal on miss. Reranker has the recalled items in hand and is judged on *ordering* of the top — NDCG@10. Different roles, different metrics.

  **F1.** Why recall@1000 and not recall@100?

  > Catalog has hundreds of millions of items. Reranker can chew through 1k. K is set by what the reranker can score in latency budget (~50 ms).

    **F1.1.** How do you build a "ground truth" for recall@1000 evaluation?

    > Two sources:
    > - Pooled human-labeled judgments (slow, expensive, definitive).
    > - Click / purchase logs as positive labels (cheap, biased — missing items the user never saw can be relevant too).
    > 
    > Production Amazon uses both: human labels for canonical eval sets, logs for online A/B.

      **F1.1.1.** How do you correct the **position bias** in click logs?

      > Inverse propensity scoring (IPS). Each click is reweighted by `1 / P(item shown at that position)`. Estimate the propensities by interleaving experiments. The metric becomes `IPS-NDCG` — same formula, weighted contributions.

**Q.** A PM says "we want to optimize for revenue, not relevance."

  **F1.** Revenue-weighted NDCG?

  > Replace `gain = 2^rel - 1` with `gain = rel × price × margin`. Custom metric subclass. The math is the same; the relevance grade is replaced.

    **F1.1.** Won't this cause "revenue bombing" (showing only high-priced items)?

    > Yes, if you don't constrain it. The fix is multi-objective optimization with a **constraint** on relevance: maximize revenue NDCG **subject to** click-through NDCG remaining within ε of baseline. The metric stack must include both, and the launch criterion must hit both.

      **F1.1.1.** How do you measure "ε within baseline" rigorously?

      > Bootstrapped confidence intervals. `BootStrapper(NDCG_click)` on each model's logs; new model passes if its NDCG_click CI overlaps the baseline's CI. That's a non-parametric version of "no statistically significant regression."

---

### AMZ-2. Recommendations (homepage carousel, "Customers who bought…")

**Business goal.** Drive incremental purchases. The lift over a no-recommendation control is the actual ground truth.

**ML metric stack.**

- Offline: NDCG, Recall@k, Hit@k, MRR.
- Online: **incremental conversion rate**, **incremental GMV**, **session length**.

```python
from torchmetrics.retrieval import RetrievalRecall, RetrievalNDCG, RetrievalHitRate
```

#### Interview drill-down

**Q.** Why are offline metrics imperfect proxies for online lift?

> A. Offline metrics evaluate against logged data, which is *itself* generated by the prior model. A "perfect" offline metric just means you reproduce the prior model's choices. Online experimentation is the only source of unbiased lift.

  **F1.** What's the right ML number to gate an online experiment on?

  > Offline NDCG must improve on a *counterfactual* eval — i.e. eval against passively logged user choices, not just the prior recommender's outputs. Building these counterfactual sets is hard; many teams use **uniform-random exploration logs** for ground truth.

    **F1.1.** How much exploration traffic is needed?

    > Enough to make the standard error of NDCG smaller than the launch-decision threshold. Typically 1-5 % of traffic, randomly assigned. The cost (some bad recommendations during exploration) is the price of unbiased eval.

      **F1.1.1.** Walk through the math of "smaller than the launch threshold."

      > If you want to call a 1 % NDCG improvement statistically significant at α=0.05 with power 0.8, and per-user NDCG variance is σ², you need roughly `n = (1.96 + 0.84)² × 2σ² / 0.01²` paired observations. For typical e-com σ², that's millions per arm — only feasible with cheap exploration logging.

---

### AMZ-3. Fraud / payment risk (binary classification, brutally asymmetric cost)

**Business goal.** Block fraudulent orders without blocking legit ones. Cost of a missed fraud (FN) ≈ chargeback ($20-300+); cost of a blocked legit (FP) ≈ lost lifetime customer (much higher) for some segments.

**ML metric.** Operating-point precision/recall, **plus a custom dollar-loss metric**.

```python
from torchmetrics.classification import (
    BinaryRecallAtFixedPrecision, BinaryAUROC, BinaryCalibrationError
)

fraud_metrics = MetricCollection({
    "rec@p99": BinaryRecallAtFixedPrecision(min_precision=0.99),
    "auroc":   BinaryAUROC(),
    "ece":     BinaryCalibrationError(n_bins=15),
})
```

The custom metric:

```python
class DollarLoss(Metric):
    higher_is_better = False
    full_state_update = False
    def __init__(self, fn_cost_fn, fp_cost_fn):
        super().__init__()
        self.fn_cost_fn = fn_cost_fn   # e.g. lambda order: order.amount
        self.fp_cost_fn = fp_cost_fn   # e.g. lambda order: order.ltv * 0.01
        self.add_state("loss", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n",    torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, preds, target, fn_costs, fp_costs):
        decided = preds > 0.5
        loss = ((~decided) & (target == 1)).float() * fn_costs \
             + (decided  & (target == 0)).float() * fp_costs
        self.loss += loss.sum()
        self.n    += target.numel()

    def compute(self):
        return self.loss / self.n
```

#### Interview drill-down

**Q.** Why precision-at-fixed-recall and not F1?

> A. Fraud has a hard precision constraint: you cannot block more than X % of legit orders without unacceptable customer-experience damage. F1 implicitly trades precision for recall at a 1:1 ratio that doesn't reflect the business cost. Operating-point metrics make the constraint explicit.

  **F1.** Why 99 % precision specifically?

  > Most retailers operate at 99-99.5 % legit-customer pass-through. Below that, customer complaints overwhelm support queues regardless of how good the fraud catch rate is.

    **F1.1.** What if your model can't hit 99 % precision at usable recall?

    > Two paths:
    > - Serial review: model predicts top X % of orders for **manual review** rather than auto-block. Recall@FPR (cost = reviewer time per order) becomes the metric.
    > - Tiered models: a high-precision model auto-blocks; a high-recall model adds friction (3DS challenge, hold for confirmation). Each tier has its own metric.

      **F1.1.1.** How do you measure the value of "friction" interventions?

      > A challenge converts a chargeback into either (a) abandoned cart (false-friction, customer leaves) or (b) authenticated purchase. Custom metric: `(abandoned_legit × cart_value × abandon_repeat_rate − catched_fraud × chargeback_value) / total_friction_events`. This is the metric the friction team actually optimizes.

**Q.** A regulator asks for proof your model isn't biased against any group.

  **F1.** Use `BinaryFairness` over protected groups.

  > It computes demographic parity, equal opportunity, predictive equality, and equalized odds. The 4/5 rule (each group's positive rate ≥ 80 % of the favored group's) is a common bar.

    **F1.1.** What if the model passes 4/5 on aggregate but fails on a *combination* of attributes?

    > "Subgroup fairness." You must test **intersectional** subgroups (e.g. age × geography × payment method). TorchMetrics doesn't ship intersectional fairness; you build it as a `MultitaskWrapper` of `BinaryFairness` instances, one per intersectional bucket.

      **F1.1.1.** How do you handle low-volume subgroups (statistical noise dominates)?

      > Threshold by minimum sample size (`n ≥ 100` per subgroup) and report only those. For lower-n groups, report the bootstrapped CI and let the auditor see the uncertainty rather than a misleading point estimate.

---

### AMZ-4. Demand forecasting at SKU × FC × day grain (time series)

**Business goal.** Decide how many units of every SKU to position in every fulfillment center every day. Wrong = either out-of-stock (lost sale, customer disappointment) or overstock (warehouse cost, eventual markdown).

**ML metric.** Quantile loss at the inventory-relevant percentile (typically P90 or P95), wMAPE, CRPS.

```python
from torchmetrics.regression import (
    WeightedMeanAbsolutePercentageError,
    ContinuousRankedProbabilityScore,
)

forecast_metrics = MetricCollection({
    "wmape": WeightedMeanAbsolutePercentageError(),
    "crps":  ContinuousRankedProbabilityScore(),
    # quantile loss: custom metric (see custom-metrics.md)
})
```

#### Interview drill-down

**Q.** Why do inventory teams want P90 and not the mean forecast?

> A. The cost asymmetry: stockouts cost more than overstock. Holding 90th-percentile demand on the shelf means stockout in only 10 % of weeks — a typical service-level target.

  **F1.** Different SKUs have different cost asymmetries. How do you reflect that?

  > Per-SKU quantile. Cheap fast-moving consumables: P95 (high service level). Expensive slow-movers (electronics, bulky goods): P75 (cost of holding outweighs service-level loss).

    **F1.1.** How do you choose the per-SKU quantile programmatically?

    > Newsvendor formula: optimal quantile = `cost_underage / (cost_underage + cost_overage)`. For a $100 SKU with $5 holding cost and $30 lost-margin per stockout: q* = 30 / (30 + 5) = 0.857 → P85.

      **F1.1.1.** How do you evaluate per-SKU quantile *coverage* in production?

      > Custom metric: for each SKU, compute fraction of weeks where `actual ≤ predicted_quantile`. Aggregate to a **coverage-error** — average absolute deviation from the target quantile. State: per-SKU `(hits, n)`. Reduce by `sum`; compute as `mean(|hits/n − target_q|)`.

---

### AMZ-5. Delivery time prediction (regression with hard SLA)

**Business goal.** Surface "Get it by Thursday" promises that the network can keep. Missing the promise destroys trust; conservative promises lose customers to faster competitors.

**ML metric.** **Asymmetric quantile loss** (over-promise much worse than under-promise) plus **on-time rate** at the predicted quantile.

#### Interview drill-down

**Q.** Why predict a delivery date and not the underlying transit-time distribution?

> A. The customer interface is a date — "by Thursday." The model must predict the **boundary** that achieves the desired on-time rate. That's a quantile prediction problem.

  **F1.** Which quantile?

  > Service-level driven. If the company's promise is "delivered on or before the date 95 % of the time," you predict the P95 of transit time.

    **F1.1.** How do you measure the metric in production?

    > For each delivery, log `(predicted_delivery_date, actual_delivery_date)`. Custom metric: `OnTimeRate` (fraction with actual ≤ predicted). Target = 0.95. State: `(hits, n)` summed across DDP.

      **F1.1.1.** What if your predictor is too conservative on edge cases?

      > Segment by transit lane (origin FC × destination zip). Per-lane on-time rate. The right metric for the dashboard is the **histogram** of per-lane on-time rates — the min lane is what kills you, not the average.

---

### AMZ-6. Ad ranking / sponsored search

**Business goal.** Maximize `bid × P(click) × P(conversion | click)` while respecting advertiser budgets and customer relevance.

**ML metric.**

- `BinaryCalibrationError` on `P(click)` — pacing depends on calibrated probability.
- `RetrievalNDCG@5` — ad slots are typically top-of-page.
- Custom **revenue-per-mille (RPM) lift**.

#### Interview drill-down

**Q.** Why is calibration mission-critical here?

> A. Auction price depends linearly on `P(click) × bid`. A 10 % miscalibrated `P(click)` is a 10 % wrong auction price → broken pacing for every advertiser. Discrimination (AUC) without calibration is worthless.

  **F1.** What kinds of bias appear in `P(click)`?

  > Position bias (top slots get clicked because they're top, not because they're relevant). Selection bias (we only saw clicks on ads we showed). Both must be debiased before calibration is meaningful.

    **F1.1.** How does calibration error compose with position-bias correction?

    > Sequential. First debias the click logs via inverse-propensity-scoring or counterfactual reweighting; then evaluate calibration on the debiased data. TorchMetrics' `BinaryCalibrationError` doesn't ship IPS, but you can compute it on the IPS-reweighted stream by weighting `update(...)` arguments — write a small custom metric or an IPS wrapper.

---

### AMZ-7. Alexa / voice (ASR + NLU + dialog)

**Business goal.** Ship the right action for what the user said. Each subsystem has its own metric.

| Subsystem | Metric |
|---|---|
| Wake-word detection | False-accept @ false-reject operating point. |
| ASR | WER + per-domain WER (music vs. shopping). |
| Intent classification | F1 macro across intents. |
| Slot filling | Slot F1 (token-level F1). |
| Dialog success | Task-completion rate — *not a TorchMetrics metric*; it's a labelled outcome. |

The hardest interview question: how does the ASR WER number translate to dialog success?

#### Interview drill-down

**Q.** Lower WER doesn't always mean higher task completion. Why?

> A. Some words matter more than others. Misrecognizing "play" as "pay" can break a music command but be invisible to overall WER. The relevant signal is **content-word WER** or **slot-impacting WER** — error rates on the words your downstream NLU actually uses.

  **F1.** How do you compute content-word WER?

  > Tag each reference word as content/function based on the NLU schema. Run WER per tag class. TorchMetrics has `WordErrorRate`; you wrap it in a per-tag `MultitaskWrapper`.

    **F1.1.** How do you tie this back to dialog success?

    > Run a paired analysis: bucket dialogs by content-word WER and measure task-completion rate per bucket. The slope `dTaskCompletion / dContentWER` is the effective business gradient — that's the number you optimize.

---

## A meta-pattern: every business pipeline needs three metric layers

For *any* of these scenarios, the architecture is the same:

1. **Model-quality metric** — the TorchMetrics number (F1, NDCG, CRPS, calibration). Fast, comprehensive, automated.
2. **Decision-quality metric** — what you'd run in shadow mode (operating-point precision/recall, dollar-loss). Computed offline on logs.
3. **Outcome metric** — measured online via experimentation (revenue lift, on-time rate, churn rate, complaint volume). The actual business KPI.

The hierarchy:

```text
Outcome metric  (slow, ground truth, business KPI)
    ↑
Decision metric (offline replay, threshold-aware)
    ↑
Model metric    (TorchMetrics, fast, threshold-free)
```

In an interview, when someone asks "what metric do you use," answer at all three levels and explain the bridges. That alone separates a junior answer from a staff-level one.

---

## Cheat sheet — domain-to-metric translation

| Domain | Primary TorchMetrics | Bridge to business |
|---|---|---|
| Delay / cancellation prediction | `BinaryF1`, `BinaryAUROC`, `BinaryCalibrationError` | Operating-point thresholding, ops cost asymmetry |
| Overbooking / no-show | Quantile loss (custom), `CRPS` | Newsvendor: cost_under / (cost_under + cost_over) |
| Dynamic pricing | `BinaryCalibrationError`, `BrierScore` (= MSE) | Calibrated `P(buy)` enters revenue optimizer directly |
| IROPS / rebooker | `RetrievalNDCG`, `RetrievalMRR` | Top-k truncation at offer slots |
| Customer churn | `BinaryAveragePrecision`, time-discounted recall | LTV-weighted threshold tuning |
| Demand forecasting | `WeightedMAPE`, `CRPS`, quantile loss | Capacity / inventory cost asymmetry |
| E-com search | `RetrievalRecall@K`, `RetrievalNDCG@K` | Two-stage system: recall + rank |
| Recommendations | `RetrievalNDCG`, `RetrievalHitRate` | Online A/B for incremental conversion |
| Fraud / payment risk | `BinaryRecallAtFixedPrecision`, dollar-loss | Per-segment cost matrix |
| Ad ranking | `BinaryCalibrationError`, `RetrievalNDCG` | Auction price ∝ `P(click) × bid` |
| ASR | `WordErrorRate`, content-WER | Dialog success ∝ slot-impacting WER |
| LLM eval | `Perplexity`, `BERTScore`, `ROUGE` | Human SBS / preference rate is ground truth |

When a question starts with "your model …", finish with "…and that translates to business via …". That's the move.
