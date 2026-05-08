---
title: Classification metrics — interview deep dive
---

# Classification metrics — interview deep dive

> 22 interview questions in **Q → F1 → F1.1 → F1.1.1** drill-down format. The leaves are where senior loops live.

[← Family page](./index.md) · [← Home](../index.md)

---

## Q1. "Walk me through accuracy, precision, recall, F1 — when does each fail?"

**Answer (model).** Accuracy fails under class imbalance — a 99/1 dataset hits 99% accuracy with a constant predictor. Precision fails as a single number because a model that predicts positive *once* has precision 1.0; you must co-report recall. Recall fails the same way mirrored — predict positive everywhere, recall = 1. F1 hides the precision-vs-recall trade-off; F2/F0.5 expose it. The right framing in an interview: *every* scalar misleads under imbalance — that's why the family is so big.

> **F1.1** "How do you choose between F1, MCC, and AUPRC for an imbalanced binary problem?"
>
> **Answer.** AUPRC is threshold-free — pick it for *ranking* quality. F1 needs a threshold; pick it for *operating-point* quality and report the threshold. MCC for *single-number model selection*; it incorporates all four cells of the matrix and isn't fooled by majority-class predictors. In practice: AUPRC for the leaderboard, F1 for the deployment dashboard, MCC for the gating decision.
>
> > **F1.1.1** "When can MCC and F1 disagree on which model is better?"
> >
> > **Answer.** When two models have similar TP+FN counts (similar recall) but very different TN counts. F1 ignores TN; MCC uses it. Model A predicting positive aggressively has more TPs but loads more FPs — F1 may favor it because of higher recall at decent precision; MCC penalizes the spent TNs and may prefer a more conservative model B. Real example: fraud detection where model A has higher F1 but model B's *MCC* is higher because B doesn't burn analyst time on FPs.

> **F1.2** "What's the difference between macro F1 and weighted F1?"
>
> **Answer.** Macro = unweighted mean of per-class F1; treats every class equally. Weighted = mean weighted by support; treats every *sample* equally. On a 99/1 dataset, weighted F1 is dominated by the majority class and looks great; macro F1 punishes the rare-class failure. Pick macro when you want fairness across classes, weighted when you want sample-weighted realism.
>
> > **F1.2.1** "Are macro and micro F1 ever the same?"
> >
> > **Answer.** Yes — for *single-label multiclass* classification, micro F1 = micro precision = micro recall = accuracy. For multilabel they diverge. For binary, "micro" can collapse to either depending on `average=` semantics.

---

## Q2. "Explain ROC AUC. Why is it not the right metric for fraud detection?"

**Answer.** ROC AUC = Pr(score(positive) > score(negative)). It's threshold-free and equals the Wilcoxon-Mann-Whitney U statistic. For fraud (0.05% base rate), AUROC is dominated by ranking the abundant negatives correctly — it can be 0.99 while the *top of the list* is mostly false-positives. Production cares about top-of-list precision; AUPRC reflects that, AUROC doesn't.

> **F2.1** "Why does AUPRC have the property that 'random' is the base rate, not 0.5?"
>
> **Answer.** A random scorer's PR curve is a horizontal line at `P = base_rate`, so its area equals the base rate. AUROC's random baseline is 0.5 because for a random scorer, TPR(t) = FPR(t) for every threshold so the curve is the diagonal `y = x`.
>
> > **F2.1.1** "What does AUPRC = 0.30 mean for a 0.05% positive rate?"
> >
> > **Answer.** Across all thresholds, average precision is 30% — vs the random baseline of 0.05%. That's a 600× lift. Phrase it that way: "AUPRC of 0.30 vs base-rate 0.0005 ≈ 600× lift over random." Gets you instant credit for understanding the scale.

> **F2.2** "Two models have the same AUROC. How do you decide which to deploy?"
>
> **Answer.** Three lenses: (1) AUPRC — different at the rare-positive end. (2) Calibration (ECE) — both can have identical ranking but only one gives usable probabilities. (3) The deployment operating point — `RecallAtFixedPrecision` or `PrecisionAtFixedRecall` reflects the *actual* threshold business has agreed to.

---

## Q3. "What is calibration and how do you measure it?"

**Answer.** A model is calibrated if `Pr(y=1 | model says p) = p`. Measure with Expected Calibration Error: bin predictions by confidence, take the absolute gap between bin accuracy and bin confidence, weight by bin size. In TorchMetrics: `CalibrationError(task="binary", n_bins=15, norm="l1")`.

> **F3.1** "AUROC is great but ECE is bad. Can you ship?"
>
> **Answer.** It depends what consumes the score. If a threshold-based system uses class IDs, calibration doesn't matter — ranking does. If a downstream system uses the *probability* (expected-cost decisions, abstention, ensembling), bad calibration breaks it. Solution: post-hoc calibration (Platt for sigmoid-shaped miscalibration, isotonic for general). Re-evaluate ECE on a held-out set after fitting the calibrator on validation.
>
> > **F3.1.1** "When is Platt scaling preferable to isotonic regression?"
> >
> > **Answer.** Platt = single sigmoid: 2 parameters. Isotonic = step function: O(n) parameters. Use Platt with small calibration sets (< 1000) to avoid overfitting. Isotonic with large sets and when miscalibration is non-sigmoidal (e.g., underconfident at both extremes — Platt can't fix that, isotonic can).

> **F3.2** "ECE bins at 15. Why not 100?"
>
> **Answer.** Variance vs bias trade-off. With 100 bins on 10k samples, each bin has ~100 points; the per-bin accuracy estimate has ~10% standard error and ECE looks artificially high (variance). With 5 bins, miscalibration in the 0.85-0.95 range gets smeared into the 0.8-1.0 bin (bias). 15 is the de-facto standard from Guo et al., 2017.

---

## Q4. "Top-1 vs top-5 accuracy — why do we report both for ImageNet?"

**Answer.** Top-1 = strict argmax correct. Top-5 = correct class is in top-5 predictions. ImageNet has many visually similar classes (e.g., breeds of dog) where ambiguity is irreducible from the image alone. Top-5 cleans that up so models can be compared on "are you in the right neighbourhood of class space" separately from "did you get the exact label."

> **F4.1** "When would top-5 mislead you?"
>
> **Answer.** When downstream uses argmax. A top-5 of 95% with top-1 of 60% is a model that's hedged across all 5; the system never sees the bottom 4. For deployment with argmax, only top-1 matters. For retrieval / re-ranking systems where downstream uses the top-k as a candidate set, top-5 (or top-k for the right k) is the right metric.

---

## Q5. "Build a binary classification metric stack from scratch — what do you log?"

**Answer.** Six things, all in one Lightning module:

```python
self.metrics = MetricCollection({
    "accuracy":  Accuracy(task="binary"),
    "precision": Precision(task="binary"),
    "recall":    Recall(task="binary"),
    "f1":        F1Score(task="binary"),
    "auroc":     AUROC(task="binary"),
    "auprc":     AveragePrecision(task="binary"),
    "ece":       CalibrationError(task="binary", n_bins=15),
    "spec_at_sens_95": SpecificityAtSensitivity(task="binary", min_sensitivity=0.95),
    "confmat":   ConfusionMatrix(task="binary"),
})
```

Why each: AUROC + AUPRC = ranking quality on both balanced and imbalanced views. P/R/F1 = operating-point quality. ECE = calibration. Spec@Sens95 = the deployment number for medical/safety contexts. Confusion matrix = the diagnostic.

> **F5.1** "Why a `MetricCollection` instead of separate metric objects?"
>
> **Answer.** Three reasons: (1) shared state — `MetricCollection` deduplicates updates if metrics share computation (e.g., `compute_groups=True` shares confusion-matrix updates across P/R/F1). (2) DDP — one `_sync_dist` call instead of N. (3) Boilerplate — one `.update()`/`.compute()`/`.reset()` instead of N.
>
> > **F5.1.1** "What is `compute_groups` in `MetricCollection` and when does it bite you?"
> >
> > **Answer.** `compute_groups=True` (default) detects metrics that share a state schema (same `_states` and same update method) and computes their states once. Bites when two metrics *look* like they share state but have different `update` semantics — TorchMetrics is conservative but custom metrics can fool it. Set `compute_groups=False` if you suspect cross-contamination.

---

## Q6. "What does macro-average AUROC mean for multiclass?"

**Answer.** TorchMetrics computes per-class one-vs-rest AUROC then averages. There's also a multinomial Hand-Till AUROC (averages over class pairs) which is theoretically nicer but rarely matches what a sklearn user expects. Always specify `average="macro"` and remember the per-class definition.

> **F6.1** "Class 7 has only 3 samples in the test set. What does its AUROC mean?"
>
> **Answer.** Almost nothing. With 3 positives, the AUROC is one of `{0, 1/3, 2/3, 1}` essentially, depending on whether they're all ranked first. Per-class AUROC is unstable for tiny support — report support next to it, or aggregate with `average="weighted"` so the number isn't pulled around by tiny classes.

---

## Q7. "Multilabel classification — what's special?"

**Answer.** Each label is independent and gets its own threshold. Implications: (1) label correlations matter (predicting all 10 labels independently can violate "≤ 5 labels per sample" constraints). (2) TN dominates → specificity-flavoured metrics are trivially ~1. (3) "Accuracy" has two meanings: per-label accuracy (loose) and subset/exact-match accuracy (strict). Choose explicitly.

> **F7.1** "What metric do you use for a multilabel system that requires the *exact* label set?"
>
> **Answer.** `ExactMatch` (subset accuracy). It's brutal — random baseline is `0.5^C` — but it matches the downstream contract. Co-report Hamming so you know how *close* you are when you're wrong.

> **F7.2** "How do you handle threshold selection for 50 labels?"
>
> **Answer.** Per-label threshold tuning: for each label, sweep thresholds, pick the one maximizing the per-label objective (F1 or fixed-precision). Cache the 50 thresholds and apply them at inference. Single global threshold is leaving precision/recall on the table.

---

## Q8. "Why does sklearn give a different F1 from TorchMetrics?"

**Answer.** Three places they can diverge: (1) `average="binary"` semantics in sklearn vs `task="binary"` in TorchMetrics — sklearn returns the positive-class score by default, TorchMetrics treats binary as a two-class case. (2) Zero-division — sklearn's `zero_division=` parameter vs TorchMetrics' default 0.0. (3) Float precision — TorchMetrics keeps state in float64 by default, sklearn varies. The first two cause real disagreements; the third is sub-epsilon.

> **F8.1** "Your unit tests pin sklearn — what gotcha do you have to handle?"
>
> **Answer.** When the test set has no positives in some class, sklearn's macro F1 with `zero_division=0` returns 0; TorchMetrics with default settings returns 0; with `zero_division="warn"` sklearn returns NaN. Match settings explicitly: `Precision(..., ..., zero_division=0)` (when supported) or post-process NaNs in your test.

---

## Q9. "I have probabilities from a softmax. What's the difference between argmax → accuracy vs probability → AUROC?"

**Answer.** Argmax accuracy throws away ranking — you only see whether the *largest* probability matches truth. AUROC uses the *full ranking*, so a model whose softmax barely separates classes can score lower AUROC than one with sharp ranks even if both have similar accuracy. Translation: AUROC penalizes flat softmaxes; accuracy doesn't.

> **F9.1** "If accuracy is 90% but AUROC is 0.7 multiclass, what's happening?"
>
> **Answer.** The argmax is right 90% of the time, but for non-top classes, the rank order is bad. This is fine if downstream only uses argmax, but if it uses top-k for retrieval/re-ranking, top-1 accuracy hides a real problem.

---

## Q10. "Walk me through how TorchMetrics handles DDP for AUROC."

**Answer.** AUROC's state is the full `(preds, target)` lists, not running counters. On `_sync_dist`, TorchMetrics gathers via `all_gather_object` (Python lists) or, with newer versions, `all_gather` on padded tensors with bookkeeping for ragged sizes. Once gathered on rank 0, AUROC is computed exactly. Cost: O(N) memory and bandwidth — much heavier than a metric like accuracy whose state is just `(correct, total)`.

> **F10.1** "How do you reduce that cost?"
>
> **Answer.** Three knobs: (1) `thresholds=k` argument on AUROC — bins predictions into k thresholds *before* sync, so state is k counts not N predictions. (2) Compute AUROC at epoch end on rank 0 only (`sync_dist=False` then `gather_for_metrics`). (3) Sample — AUROC on 1M samples ≈ AUROC on 50k samples ± ε; for tracking only.
>
> > **F10.1.1** "What's the trade-off of `thresholds=100`?"
> >
> > **Answer.** Quantizes the curve: AUROC is computed on a 100-step ROC instead of N-step. Bias is upper-bounded by `1/(2·thresholds)`. For tracking and dashboards, fine. For paper-quality numbers, use full thresholds on a single GPU at the end.

---

## Q11. "What's the right metric for medical-imaging cancer screening?"

**Answer.** Sensitivity at a fixed specificity (or specificity at a fixed sensitivity), with the floor set by regulator. The reason: regulators don't accept "we maximized AUROC" — they accept "we hit ≥ 95% sensitivity with ≥ 80% specificity." That's `SpecificityAtSensitivity(min_sensitivity=0.95)` directly. Co-report calibration (ECE) and per-subgroup numbers (Group Fairness) for review.

> **F11.1** "And the right *single* number for the leaderboard?"
>
> **Answer.** The constrained metric (specificity at the agreed sensitivity floor) — never raw AUROC. Raw AUROC is what gets you 0.99 while shipping a model that biopsies 30% of healthy patients to hit recall.

---

## Q12. "What is Cohen's kappa and when is it better than accuracy?"

**Answer.** Discounts agreement by *chance* agreement. On an imbalanced dataset two random raters might agree 80% of the time just from class-distribution overlap — accuracy 80% means nothing. κ subtracts that and rescales. For inter-rater agreement, model-vs-human comparisons, or imbalanced ordinal labels (5-star ratings, severity grades), κ is the right metric. Always use *quadratic* weighting on ordinal scales.

---

## Q13. "Group fairness — explain three definitions and why they're incompatible."

**Answer.**
- **Demographic parity:** `Pr(ŷ=1 | A=a) = Pr(ŷ=1 | A=b)` — same positive rate across groups.
- **Equalized odds:** `TPR(a)=TPR(b)` AND `FPR(a)=FPR(b)` — same error rates across groups conditional on truth.
- **Predictive parity:** `Precision(a) = Precision(b)` — when the model says positive, it's right at the same rate across groups.

The Chouldechova / Kleinberg impossibility result: if base rates differ across groups, you can have *at most* one of equalized odds and predictive parity. Demographic parity contradicts both whenever base rates differ. So fairness is a *choice*, not a measurement — pre-commit with stakeholders.

> **F13.1** "Which definition does TorchMetrics' `BinaryFairness` give you?"
>
> **Answer.** It exposes per-group statistics (TPR, FPR, precision, etc.) and demographic-parity / equalized-odds-style ratios. It doesn't pick a definition — you do, by reading the right ratio for your committed metric.

---

## Q14. "How do you debug a model whose accuracy regressed in production?"

**Answer.** Five-step ladder:
1. **Confusion matrix on production traffic** — which class lost the most? Which classes started colliding?
2. **Per-segment metrics** (timezone, device, language) — is it a slice that lost vs everywhere?
3. **Calibration drift** (ECE on recent vs training) — has the score distribution shifted?
4. **Input drift** — are features distributed the same? (Use a histogram-distance metric like JS divergence, not classification metrics.)
5. **Label drift** — is the underlying truth shifting (concept drift)? Compare base rates; a shift here means accuracy is non-comparable.

Top-line accuracy alone tells you nothing about *why*.

---

## Q15. "Why is `MetricTracker` better than logging the metric every epoch?"

**Answer.** `MetricTracker` keeps a list of epoch-end values and exposes `best(maximize=True)` and the epoch where the best happened. Three wins: (1) decouples "when did we peak" from logger UI; (2) `maximize=` makes the higher/lower-is-better contract explicit (catches Hamming distance bugs); (3) integrates with checkpointing/early-stopping without re-implementing argmax.

---

## Q16. "We have a 50-class classifier where 5 classes have 90% of the data. How do you report?"

**Answer.** Report three flavours:
1. **Macro F1** — every class equal, exposes the rare-class failures.
2. **Weighted F1** — sample-weighted, what end-users feel.
3. **Per-class F1 vector** (`average=None`) — for diagnostic dashboards.

Plus a confusion matrix with rare-class rows highlighted. Top-line "F1" alone is misleading either way.

---

## Q17. "When is precision-at-k better than precision?"

**Answer.** When the system surfaces a *fixed-size* result list to the user (search top-10, ad slate, content carousel). Precision-at-k matches the user-facing experience. Plain precision varies with the threshold, which the user never sees.

> **F17.1** "And recall-at-k?"
>
> **Answer.** "Of all relevant items, did the top-k contain them?" Useful when the relevant set is small (target retrieval, set-completion). For an open-ended search where many items are relevant, recall-at-k is dominated by k and uninformative.

---

## Q18. "EER — why is it the standard biometric metric?"

**Answer.** It's a single number that summarizes the ROC at the operationally meaningful "balanced" point (FPR = FNR). All vendors report it on the same scale, so cross-vendor comparison is direct. Real systems then move the threshold off the EER point: a phone unlock wants low FPR (don't let strangers in); an authentication challenge wants low FNR (don't lock real users out). The EER itself is a quality summary; the deployed threshold is task-specific.

---

## Q19. "If you had only one metric to gate model deployment, which would you pick?"

**Answer.** **Trick question.** No single metric is right. The honest answer: pick a *constrained* metric that matches the deployment contract — `RecallAtFixedPrecision` for spam, `SpecificityAtSensitivity` for medical, `Precision@k` for ranking. Beware interviewers who push for a single answer; the right move is to push back and ask about the deployment context. Then pick.

> **F19.1** "Okay, fine — pick something for a balanced multiclass problem with no clear cost."
>
> **Answer.** Macro F1 + a confusion matrix screenshot in the PR description. Macro F1 because it's class-balanced; confusion matrix because no scalar replaces seeing what's confused with what.

---

## Q20. "Walk me through what happens when I call `metric(preds, target)` in TorchMetrics."

**Answer.** Calling `__call__` is equivalent to `update(preds, target); val = compute(); reset_to_pre_call_state()`. So you get the metric value *for this batch only* without polluting the running state. If you want the running value (across batches), call `update` then `compute` separately and `reset` at epoch end. Most Lightning bugs come from mixing these patterns — `self.log("acc", accuracy(preds, target))` is per-batch; `self.log("acc", accuracy)` (no parens) is per-epoch with auto-reset.

---

## Q21. "What's the most common multiclass-vs-multilabel bug?"

**Answer.** Passing logits of shape `(B, C)` to a metric configured for `task="multilabel"`. The metric thresholds each class independently — for a single-label-multiclass problem, that means a row can sum to 0 labels (low confidence everywhere) or 5 labels (high confidence everywhere). The metric still computes — it just measures something nonsensical. Always set `task=` and `num_classes`/`num_labels` explicitly.

---

## Q22. "How do you measure model performance on rare classes when the test set has 3 examples of each?"

**Answer.** You don't, with point estimates. Use bootstrap CIs: `BootStrapper(F1Score(...), num_bootstraps=1000)` gives you mean and std. Or pool rare classes into "other" and compute on the head. Or accept that rare-class metrics are *qualitative*: "all three are correctly classified" is enough; don't claim per-class F1 = 1.0 ± 0.0 with N = 3.

---

## Cheat-sheet: "If they ask X, anchor with Y"

| If they ask… | Anchor with… |
|---|---|
| "What metric for X?" | "Depends on cost asymmetry and downstream consumer." Then pick. |
| "Accuracy bad — why?" | Imbalance → MCC. Cost asymmetry → constrained metric. |
| "AUROC vs AUPRC" | "AUPRC for rare-positive, AUROC for ranking generally." |
| "Calibration" | "ECE with 15 bins, post-hoc Platt or isotonic." |
| "Macro vs weighted" | "Class-equal vs sample-equal — pick the one matching your fairness goal." |
| "DDP" | "AUROC syncs full state O(N); accuracy syncs counters O(1)." |
| "Fairness" | "Per-group rates + a *committed* fairness definition (impossibility result if base rates differ)." |

[← Back to family page](./index.md)
