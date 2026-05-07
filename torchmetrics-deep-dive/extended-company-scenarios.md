---
title: Extended Company Scenarios
nav_order: 29
---

# Extended Company Scenarios — Beyond AA & Amazon

The [ML ↔ Business]({{ "./ml-business-metrics.md" | relative_url }}) page covers American Airlines and Amazon. This page extends to eight more companies / industries with the same multi-level interview drill-down format. Pick the one closest to the role you're interviewing for.

---

## 1. Netflix — recommendations, video quality, churn

### N-1. Homepage row recommendations

**Business goal.** Maximize **time spent watching content the user actually likes**, weighted by completion rate (not just clicks).

**ML metrics.**

```python
from torchmetrics.retrieval import RetrievalNDCG, RetrievalRecall, RetrievalHitRate

netflix_recs = MetricCollection({
    "ndcg@10":   RetrievalNDCG(top_k=10),
    "recall@10": RetrievalRecall(top_k=10),
    "hit@5":     RetrievalHitRate(top_k=5),
})
```

Plus a custom **completion-weighted lift** metric (incremental hours watched).

#### Drill-down

> **Q.** Why graded relevance and not binary?

> **A.** Click ≠ enjoy. A 10-second click is different from a 90 % completion. Use graded relevance (0=no click, 1=brief click, 3=>50% complete, 5=full episode + next-episode click).

>> **F1.** How do you handle "playback quality" as a confounder?

>> **A.** A user who clicks but rebuffers may abandon — that's a quality signal, not a relevance signal. Either filter out low-quality sessions, or condition the metric on a quality threshold.

>>> **F1.1.** What's the right metric for the QoE side?

>>> **A.** Netflix's own **VMAF** (in `torchmetrics.video.VideoMultiMethodAssessmentFusion`) for video quality. Plus session-level rebuffer rate as a custom rolling metric.

### N-2. Video quality (encoding pipeline)

**Business goal.** Pick the right bitrate ladder per (device × content × network) such that perceived quality is maximized for a fixed bandwidth budget.

**ML metric.** **VMAF** (Video Multi-Method Assessment Fusion) — Netflix's perceptual metric, calibrated to subjective ratings.

```python
from torchmetrics.video import VideoMultiMethodAssessmentFusion

vmaf = VideoMultiMethodAssessmentFusion()
vmaf.update(reference_frames, distorted_frames)
print(vmaf.compute())   # 0–100, higher is better
```

#### Drill-down

> **Q.** Why VMAF instead of PSNR / SSIM?

> **A.** PSNR / SSIM are pixel-domain. VMAF is perceptual — calibrated against human MOS scores from real viewers. It correlates better with what users actually experience.

>> **F1.** When does VMAF disagree with humans?

>> **A.** Out-of-distribution content — animation, sports, low-light. Train custom VMAF heads per content type for accuracy. Or fall back to subjective testing for weird content.

### N-3. Churn

**Business goal.** Identify subscribers at risk of canceling; trigger retention (price hold, recommendation refresh, exclusive offer).

**ML metric.** Same as AA churn — `BinaryAveragePrecision` + time-discounted recall + LTV-weighted threshold.

---

## 2. Uber / Lyft — ETA, dispatch, surge

### U-1. ETA prediction

**Business goal.** Tell the rider when the driver will arrive. Wrong = poor experience. Too pessimistic = lost demand. Too optimistic = canceled rides.

**ML metric.** Asymmetric quantile loss. Underestimating ETA (driver arrives later than promised) is more painful than overestimating.

```python
# Predict P85 of arrival time, evaluated by pinball loss at τ=0.85
# Plus on-time-rate per (city, hour-of-day, weather) bucket
```

#### Drill-down

> **Q.** Why the asymmetry — isn't symmetric MAE OK?

> **A.** Late arrivals trigger app-rating drops, refunds, and cancellations at much higher rates than early arrivals. Cost is asymmetric; metric should be too.

>> **F1.** What about the *driver* side of the equation?

>> **A.** Drivers also need accurate ETA for their own scheduling. So the per-segment metric matters too — per-hour, per-city, per-route-type.

>>> **F1.1.** How do you reconcile rider-side and driver-side optimization?

>>> **A.** Multi-task model with `MultitaskWrapper`. Two heads: `rider_eta` (P85 quantile loss), `driver_eta` (median or mean). Different operating points; same model.

### U-2. Dispatch matching

**Business goal.** Match riders to drivers so total wait time + driver idle time is minimized, weighted by surge dynamics.

**ML metric.** Mostly an *operations research* problem, but the underlying *driver availability prediction* is an ML model evaluated by **calibrated probability** (`BinaryCalibrationError`) — the dispatcher uses `P(driver_will_accept)` directly.

### U-3. Surge pricing

**Business goal.** Set price so supply meets demand without "surge fatigue" pushing riders away long-term.

**ML metric.** Calibrated demand-elasticity model — same pattern as AA dynamic pricing. Brier score on `P(book at price)` is primary; ECE is mandatory.

---

## 3. Stripe — payment fraud / risk

### S-1. Card-present fraud

**Business goal.** Block fraudulent transactions while keeping false-positive rate low (every false block is a lost merchant + customer).

**ML metric.** Same as Amazon fraud — `BinaryRecallAtFixedPrecision(min_precision=0.99)` + dollar-loss + per-segment fairness.

```python
fraud_metrics = MetricCollection({
    "rec@p99":  BinaryRecallAtFixedPrecision(min_precision=0.99),
    "auroc":    BinaryAUROC(),
    "ap":       BinaryAveragePrecision(),
    "ece":      BinaryCalibrationError(n_bins=15),
})
# + custom DollarLoss per merchant category
# + BinaryFairness across merchant geographies
```

#### Drill-down

> **Q.** Why merchant-segmented metrics?

> **A.** Fraud patterns differ by merchant category (digital goods vs. groceries vs. travel). A model that's great on average can be terrible on a specific MCC, and that merchant churns. Segment, monitor, alert.

>> **F1.** What if a merchant complains "your model is biased against my customers"?

>> **A.** Run `BinaryFairness` over their customer demographic; check the four-fifths rule across groups. Document the audit; if there's bias, retrain with fairness constraints.

>>> **F1.1.** How do you re-train with fairness constraints in TorchMetrics' world?

>>> **A.** TorchMetrics measures; it doesn't enforce. Fairness regularization is a *training-side* concern (e.g. adversarial debiasing, or constrained optimization with a fairness penalty). Use the metric as the constraint signal during training.

### S-2. ACH return prediction

**Business goal.** Flag bank transfers likely to bounce so Stripe can hold funds or warn the merchant.

**ML metric.** Highly imbalanced binary — Average Precision primary, ECE for calibration (because the probability is consumed by an underwriting model downstream).

---

## 4. Meta — feed ranking, ads, content moderation

### M-1. Feed ranking

**Business goal.** Maximize meaningful interactions (likes, comments, time spent) while suppressing low-quality / adversarial content.

**ML metric stack.**

- `RetrievalNDCG@K` for engagement-graded ranking.
- Multi-objective: separate metrics for `P(like)`, `P(comment)`, `P(share)`, `P(report)` — each calibrated independently.
- **Calibration mandatory** — auction-style ranking multiplies probabilities.
- Per-locale, per-feed-type metric.

#### Drill-down

> **Q.** How does Meta combine multiple objectives (like, share, comment) into one ranking?

> **A.** Each is a calibrated probability; the rank score is a learned weighted sum (the **value model**). The metric layer reports per-objective NDCG/calibration so you can spot regressions in any one.

>> **F1.** What if `P(like)` improves but `P(report)` also improves (more borderline content)?

>> **A.** Multi-objective Pareto reporting. The launch criterion includes a hard ceiling on `P(report)` — you can't trade quality of conversation for click metrics.

### M-2. Ad ranking

**Business goal.** Same as Amazon ad ranking — auction price ∝ `P(click) × bid × ad-quality`. Calibration is critical because of the auction mechanism.

### M-3. Content moderation

**Business goal.** Detect policy-violating content with low FPR (false positive = wrongly removed legitimate content; long-term PR risk).

**ML metric.** Per-policy `BinaryRecallAtFixedPrecision`. Each policy (hate speech, violence, etc.) has its own metric and its own threshold tuned to that policy's cost matrix.

#### Drill-down

> **Q.** Why per-policy and not one combined metric?

> **A.** Cost matrices differ. Mistakenly hiding a political post has different consequences than mistakenly hiding spam. One number hides the imbalance.

>> **F1.** How do you handle per-locale variation?

>> **A.** Per-(policy × locale) metric grid. Production fact: hate-speech precision in language X may legitimately need a higher threshold than language Y because the linguistic ambiguity differs.

---

## 5. Google — search, YouTube, Gmail

### G-1. Web search

**Business goal.** Return the most relevant 10 results for a query.

**ML metric.** Same retrieval stack as Amazon search — `Recall@1000` (candidate gen), `NDCG@10` (reranker), `MRR` (Q&A-style queries).

But Google adds:

- **Spam filter precision** — the rerank can't surface spam.
- **Freshness signal** — for time-sensitive queries (news, sports), rank metric weighted by recency.

### G-2. YouTube watchtime

**Business goal.** Maximize meaningful watchtime per session, suppressing clickbait.

**ML metric.** Custom `WatchtimeWeightedNDCG` — gain function uses log(watchtime + 1) instead of binary clicks. Standard NDCG ignores quality of consumption.

### G-3. Gmail spam

**Business goal.** Block spam without false-positive on important mail.

**ML metric.** Same as Stripe / Amazon fraud pattern. `BinaryRecallAtFixedPrecision(min_precision=0.999)` — even higher precision constraint because losing a real email is brutal.

---

## 6. Tesla — perception, fleet learning

### T-1. Object detection (camera-only)

**Business goal.** Detect vehicles, pedestrians, lane markings, signs at appropriate range and confidence.

**ML metric.** mAP on custom dataset, but also:

- **Per-class AP** (pedestrian AP critically more important than mailbox AP).
- **Per-distance AP** (a 100m pedestrian miss is worse than a 5m parked-car miss).
- **Per-condition AP** (rain, night, glare).

```python
from torchmetrics.detection import MeanAveragePrecision

map_metric = MeanAveragePrecision(
    box_format="xyxy",
    class_metrics=True,
    compute_on_cpu=True,
)
# + per-segment custom mAP slicers (distance, condition)
```

#### Drill-down

> **Q.** Why is mAP not enough for safety-critical perception?

> **A.** mAP averages over thresholds. For safety, you need a guaranteed *recall floor* on critical classes (pedestrian) at relevant distances. **Recall@FixedPrecision** per (class × distance × condition) bucket, with a **hard recall constraint** before launch.

>> **F1.** What about fleet-learning data labeling cost?

>> **A.** Active learning loop: low-confidence predictions are flagged for human label; the model retrains. The relevant metric here is **labeling efficiency** = AP improvement / labels added. Custom metric tracking this is a senior-level signal.

### T-2. Driver behavior modeling

**Business goal.** Predict driver intent (will they brake?) for advanced driver assistance.

**ML metric.** Time-discounted recall — catching the brake intention 1 s ahead is much more valuable than 100 ms ahead. Custom metric required.

---

## 7. Healthcare / biotech — diagnostic models

### H-1. Disease screening (binary classification at low base rate)

**Business goal.** Identify patients at risk of a disease (low base rate, e.g. 0.5 %). Cost of FN (missed diagnosis) is enormous; cost of FP (unnecessary follow-up) is moderate.

**ML metric.**

- `BinaryRecallAtFixedPrecision(min_precision=0.20)` — at low base rates you can't get high precision; aim for "20 % of positives are real."
- `BinarySensitivityAtSpecificity(min_specificity=0.95)` — alternative phrasing.
- `BinaryAveragePrecision` — the main reporting number.
- ROC at multiple operating points.

#### Drill-down

> **Q.** Why precision so low (20 %)? Isn't that bad?

> **A.** At 0.5 % positive rate, even a perfect model returning the top 5 % of risk has precision = 0.5/5 = 10 %. Realistic. The cost matrix justifies it: 5 unnecessary follow-ups for 1 caught disease is a great trade.

>> **F1.** How do you communicate this to a clinician?

>> **A.** Show **decision-curve analysis** — net benefit at different threshold probabilities. Custom metric, not in TorchMetrics core. Or report **NNS** (number needed to screen) and PPV at multiple thresholds.

>>> **F1.1.** How do you ensure model fairness across subpopulations?

>>> **A.** `BinaryFairness` or per-subgroup recall@precision tracking, with at least: (a) sex, (b) age decile, (c) race/ethnicity, (d) primary language. Healthcare bias is closely scrutinized.

### H-2. Medical imaging segmentation

**Business goal.** Segment tumors / organs from CT / MRI.

**ML metric.** Same as Scenario 3 — Dice + IoU + HD95.

---

## 8. Financial services — credit risk, AML

### F-1. Credit scoring

**Business goal.** Predict probability of default. Used directly in pricing and approval.

**ML metric.** **Calibration is paramount.** A miscalibrated PD goes straight into capital requirements (regulators care).

- `BinaryCalibrationError` (Hosmer-Lemeshow style).
- `BinaryAUROC` (Gini coefficient = 2·AUROC − 1; banks report Gini).
- Per-segment by credit-bureau source, geography, applicant age.
- Backtesting metrics: time-stability of PD estimates over multiple years.

#### Drill-down

> **Q.** Why do banks report Gini and not AUROC?

> **A.** Same information, different convention. Gini = 2·AUROC − 1. So Gini = 0.6 ↔ AUROC = 0.8. Pre-1990s convention; sticky.

>> **F1.** Calibration drift over time — how do you handle it?

>> **A.** Annual recalibration. Track ECE month-over-month. Trigger recalibration when 3-month rolling ECE > threshold. Some banks do explicit Bayesian updating against new defaults observed.

### F-2. Anti-money-laundering (AML)

**Business goal.** Flag suspicious transactions for human review.

**ML metric.** Highly imbalanced; per-typology metrics; per-jurisdiction; **regulator-mandated explainability** alongside performance.

The key metric is **review-worthiness**: of the alerts produced, what fraction lead to a SAR (suspicious activity report) being filed? That's the human-in-the-loop precision metric.

---

## 9. Cybersecurity — intrusion / threat detection

**Business goal.** Detect attacks / malware while keeping analyst alert volume manageable.

**ML metric.**

- AP (rare positives).
- `Recall@FixedPrecision` to bound false-positive load on the SOC analyst.
- Time-to-detection — custom metric measuring mean lag from attack start to detection.
- Per-attack-family metric (different threat types behave differently).

---

## 10. Robotics / manufacturing — defect detection

**Business goal.** Catch defective parts on a production line before they ship.

**ML metric.**

- Per-defect-type AP.
- Recall@FixedPrecision (high, e.g. 99.5 %) — missing a defect costs much more than re-inspecting.
- Per-line, per-shift drift monitoring.

#### Drill-down

> **Q.** What metric for an unsupervised anomaly detector on a new product line?

> **A.** Initially no labels — use unsupervised score distribution (Silhouette, isolation-forest score). Once enough human-labeled samples exist, switch to supervised metrics.

>> **F1.** How do you know when to switch?

>> **A.** Track the unsupervised → supervised metric correlation as labels accumulate. When the supervised confidence interval is tight enough to make production decisions, the labels are sufficient.

---

## How to use this page in interviews

1. **Pick one or two scenarios closest to the company you're interviewing with.** Memorize the metric stack and the cost-asymmetry rationale.
2. **Practice the 30-second pitch** for each: "For Netflix VMAF I'd…" / "For Stripe fraud I'd…"
3. **Have one favorite drill-down** ready 4 levels deep — interviewers test depth, not breadth.
4. **Always close with the bridge**: ML metric → operating point → A/B-tested business KPI. Three layers, every time.
