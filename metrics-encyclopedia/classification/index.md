---
title: Classification metrics — deep dive
---

# Classification metrics — deep dive

> The largest family. Every metric here is a different lens on a confusion matrix (or its rank/probability variants). The wrong choice silently flips your model ranking — that's why they all exist.

[← Back to home](../index.md) · [Interview deep-dive →](./interview-deep-dive.md)

## The four primitives all of these are built from

For binary, with a fixed threshold, every classification metric is a function of the four cells:

```
                 actual=1     actual=0
predicted=1        TP            FP
predicted=0        FN            TN
```

- **Sensitivity / Recall / TPR** = `TP / (TP + FN)`  — of the real positives, how many did we catch?
- **Specificity / TNR**          = `TN / (TN + FP)`  — of the real negatives, how many did we leave alone?
- **Precision / PPV**            = `TP / (TP + FP)`  — of our positive predictions, how many were right?
- **NPV**                        = `TN / (TN + FN)`  — of our negative predictions, how many were right?
- **FPR** = `1 - Specificity`,  **FNR** = `1 - Recall`.

Memorize these four denominators. Every metric on this page rebalances them.

## The `task=` argument

Most classes accept `task="binary" | "multiclass" | "multilabel"`. Pick wrong and the metric silently means something else:

- `binary`: 1 class, threshold-based. `preds` shape `(B,)` (probs) or `(B,)` (class IDs after argmax).
- `multiclass`: `C` mutually exclusive classes, argmax. `preds` shape `(B, C)` (probs/logits) or `(B,)` (class IDs).
- `multilabel`: `C` independent classes, threshold each. `preds` shape `(B, C)` (probs).

The `num_classes`/`num_labels` argument is required for multiclass/multilabel.

The `average=` argument controls how the per-class metric collapses across classes:

- `"micro"` — pool TP/FP/FN across classes first, then divide. Dominated by frequent classes.
- `"macro"` — compute per-class, then unweighted mean. Treats every class equally — good for class-imbalance fairness.
- `"weighted"` — per-class mean weighted by support (# real samples in that class). Same as micro for accuracy but not for F1.
- `"none"` — return the per-class vector. Use when you'll log per-class.

> **Macro vs weighted is where most bugs hide.** Macro = "every class is equally important." Weighted = "every sample is equally important." A 99/1 imbalanced dataset gives wildly different numbers under those two.

---

## Counts and totals

### StatScores (`tm.classification.StatScores`)

**What it computes.** The raw `(TP, FP, TN, FN, support)` tuple. Every other classification metric is a function of these.

**Intuition.** TorchMetrics builds StatScores on the GPU, syncs across DDP, then derives Precision/Recall/F1/etc. on top. If you implement a custom classification metric, *always* compose it on top of StatScores — never re-count.

**Range / direction.** Counts are non-negative integers; bigger TP/TN better, bigger FP/FN worse.

**When to use.** When you want to log many classification metrics and don't want to re-iterate the data each time. `StatScores` once → derive Precision/Recall/F1/Specificity downstream.

**When NOT to use.** When you need rank-based metrics (AUROC, AP) — those bypass thresholds and you can't get to them from `(TP, FP, TN, FN)`.

**Real-world scenario.** A churn model that's logged hourly under DDP across 8 GPUs. Compute `StatScores` once per hour, then derive the dashboard metrics in pure tensor math — much cheaper than calling 8 separate metrics each with `_sync_dist`.

**Code.**
```python
from torchmetrics.classification import StatScores
m = StatScores(task="multiclass", num_classes=3, average=None)
tp, fp, tn, fn, sup = m(preds, target).T  # per-class
```

**Pitfalls.**
- `average="micro"` collapses everything into one tuple — looks tempting but you've lost per-class.
- `support = TP + FN`, NOT `TP + FP`. Several Stack Overflow answers get this wrong.
- For `multilabel`, the `TN` cell is huge (most labels are negative) — TN-based metrics like specificity look ~1.0 trivially.

---

## Threshold-based scalars

### Accuracy (`tm.classification.Accuracy`)

**What it computes.** `(TP + TN) / (TP + FP + TN + FN)` — fraction of all predictions that are correct.

**Intuition.** A baseline that everyone wants. The number is intuitive: "we're right 87% of the time." It treats all errors as equal cost.

**Range / direction.** `[0, 1]`. Higher better.

**When to use.** Roughly balanced classes (within 30/70). Cost of FP ≈ cost of FN. Sanity check before you pick a real metric.

**When NOT to use.** Class imbalance (fraud detection at 0.1% positive: a model that always says "not fraud" gets 99.9% accuracy — and is useless). Asymmetric error costs (medical screening: missing a cancer ≠ scaring a healthy patient).

**Real-world scenario.** A 4-class image classifier on a balanced ImageNet subset. Accuracy is a fine top-level KPI. **Not** a fine top-level KPI for the spam detector where 0.5% of mail is spam — there, use precision-at-fixed-recall instead.

**Code.**
```python
from torchmetrics.classification import Accuracy
acc = Accuracy(task="multiclass", num_classes=10, average="micro")
acc(preds, target)
```

**Pitfalls.**
- `top_k=` only matters for multiclass with probability/logit inputs — irrelevant for binary class IDs.
- `average="macro"` *with multilabel* counts a label having 0 positives as accuracy = 1 (vacuously) — your "macro accuracy" is inflated. Use `subset_accuracy=True` for the strict "all labels match" definition.

---

### Precision (`tm.classification.Precision`)

**What it computes.** `TP / (TP + FP)` per class, then averaged. "Of what we said positive, what fraction were right?"

**Intuition.** Punishes FP. Maximizing precision = "be conservative."

**Range / direction.** `[0, 1]`. Higher better. Precision = 1 when the model only fires on certainty (or never fires — see pitfall).

**When to use.** When acting on a positive prediction is *expensive* (sending an email, freezing an account, ordering a biopsy).

**When NOT to use.** When missing a positive is the more expensive error (cancer screening, fraud, pump failure).

**Real-world scenario.** Spam filter promotion: a borderline email goes to inbox by default. We deploy a new model only if its precision on a labelled queue is ≥ 99% — false-positives push real mail to spam, which is a much worse user-facing failure than a piece of spam getting through.

**Code.**
```python
from torchmetrics.classification import Precision
p = Precision(task="multiclass", num_classes=5, average="macro")
p(preds, target)
```

**Pitfalls.**
- A model that predicts the positive class **only once** and gets it right has precision = 1.0. Always co-report recall.
- Precision over an unbalanced dataset can hide a model that's lazy — recall shows that.

---

### Recall / Sensitivity / TPR (`tm.classification.Recall`, `Sensitivity`)

**What it computes.** `TP / (TP + FN)` — "of the real positives, what fraction did we catch?"

**Intuition.** Punishes FN. Maximizing recall = "be aggressive."

**Range / direction.** `[0, 1]`. Higher better.

**When to use.** When *missing* a positive is expensive (cancer, fraud, suicide hotline routing).

**Real-world scenario.** A breast-cancer screening model: recall must be ≥ 95% under regulatory review, even at the cost of more biopsies. The decision threshold is moved to whatever value keeps recall above the floor; precision becomes the secondary metric.

**Code.**
```python
from torchmetrics.classification import Recall
r = Recall(task="binary", threshold=0.3)  # lower threshold ⇒ higher recall, lower precision
```

**Pitfalls.**
- Recall = 1.0 if you predict positive for everything. Co-report precision.
- For multilabel with sparse positives, per-label recall has tiny denominators and is noisy. Bootstrap CIs.

---

### Specificity / TNR (`tm.classification.Specificity`)

**What it computes.** `TN / (TN + FP)` — "of the real negatives, what fraction did we correctly leave alone?"

**Intuition.** Symmetric to recall but on the negative side. `1 − Specificity` is the false-positive rate, which appears on the x-axis of every ROC curve.

**Range / direction.** `[0, 1]`. Higher better.

**When to use.** Medical / safety contexts where the cost is to negatives (over-treating healthy people). Together with sensitivity, it's the canonical pair in epidemiology.

**Real-world scenario.** A COVID rapid antigen test: regulators specify a minimum *specificity* (≥ 99%) to keep the false-positive load manageable when prevalence is low. Sensitivity is reported separately.

**Pitfalls.**
- For multilabel with C labels, the TN cell is huge → specificity is trivially close to 1. Useless as a single number; use per-label.

---

### NegativePredictiveValue / NPV (`tm.classification.NegativePredictiveValue`)

**What it computes.** `TN / (TN + FN)` — "of what we called negative, what fraction were truly negative?"

**Intuition.** The mirror of precision. Important when a *negative* prediction triggers an action (e.g., releasing a patient).

**Range / direction.** `[0, 1]`. Higher better.

**Real-world scenario.** Hospital triage for low-acuity patients: if the model predicts "low acuity, can wait", you want the NPV high — almost no missed serious cases hide inside the "negative" bucket. Precision/recall don't capture this.

---

### F1 / F-beta (`tm.classification.F1Score`, `FBetaScore`)

**What it computes.** `F_β = (1 + β²) · (P · R) / (β² · P + R)`. F1 is β=1.

**Intuition.** Harmonic mean of precision and recall. The harmonic mean punishes the smaller term: F1 stays low if *either* P or R is low. β > 1 leans toward recall, β < 1 leans toward precision.

**Range / direction.** `[0, 1]`. Higher better.

**When to use.** Imbalanced classes, when you want a single number that punishes lopsided P-vs-R behaviour.

**When NOT to use.** When P and R are not equally important — pick a β that reflects cost. F1 is the lazy default.

**Real-world scenario.** Search-quality team uses F2 (β=2) for "did the result set contain the answer" — recall is twice as important as precision because we re-rank afterwards.

**Code.**
```python
from torchmetrics.classification import FBetaScore
f2 = FBetaScore(task="multiclass", num_classes=3, beta=2.0, average="macro")
```

**Pitfalls.**
- F1 macro vs micro vs weighted is *the* most common interview trap. See the macro/weighted note at the top of this page.
- Threshold sensitivity: F1 changes massively with threshold. Always co-report the threshold used.

---

### ExactMatch (`tm.classification.ExactMatch`)

**What it computes.** Multilabel: fraction of samples where *every* label matches (subset accuracy). Multiclass: same as accuracy.

**Intuition.** Strictest possible accuracy on multilabel. One label wrong → whole sample wrong.

**Range / direction.** `[0, 1]`. Higher better.

**Real-world scenario.** Document tag prediction where the downstream system requires the *exact* tag set (e.g., regulatory filings). One missing tag triggers a manual review, so per-label F1 isn't the right KPI; exact match is.

**Pitfalls.**
- On multilabel with many labels, ExactMatch is brutal: random-baseline ExactMatch ≈ 0.5^C. Always benchmark against per-label F1 too.

---

### HammingDistance (`tm.classification.HammingDistance`)

**What it computes.** Fraction of *individual labels* that are wrong, averaged over the matrix `(B × C)`.

**Intuition.** The opposite of subset accuracy — every label counts independently. Lower is better (it's a distance).

**Range / direction.** `[0, 1]`. **Lower better.**

**Real-world scenario.** Multilabel image tagger ("beach", "sunset", "people"): missing one of three tags should hurt 1/3 as much as missing all three. ExactMatch counts both as 0. Hamming counts them 0.33 vs 1.0 — that's what you want.

**Pitfalls.**
- "Lower better" trips up `MetricTracker(maximize=True)` and `early_stopping(mode="max")` — both default to higher = better.

---

### Jaccard / Intersection-over-Union for labels (`tm.classification.JaccardIndex`)

**What it computes.** `|y ∩ ŷ| / |y ∪ ŷ|`. Per-class for multiclass, per-label for multilabel.

**Intuition.** Set overlap. The same definition as IoU in detection / segmentation, just on label sets.

**Range / direction.** `[0, 1]`. Higher better.

**Real-world scenario.** Multilabel medical-imaging tagger ("nodule", "fluid", "fracture"): IoU on the predicted-tag set is more meaningful than per-label F1 averaged because it's invariant to the *number* of labels in the example.

---

### MatthewsCorrCoef / MCC (`tm.classification.MatthewsCorrCoef`)

**What it computes.** `(TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`. The Pearson correlation of the binary indicator vectors.

**Intuition.** A *single* number that uses all four confusion-matrix cells. Robust under class imbalance: a degenerate "always predict majority" scorer gets MCC = 0, whereas accuracy gets ≈ 1.0.

**Range / direction.** `[-1, 1]`. 0 = random, 1 = perfect, -1 = inverse-perfect. Higher better.

**When to use.** Heavy class imbalance, **single-number** model selection. Recommended by Chicco & Jurman (2020) over F1 for imbalanced binary.

**Real-world scenario.** Fraud team uses MCC as the *gating* metric for promotion — F1 was rewarding models that only predicted positive on outliers (high P, low R, F1 ≈ 0.4) versus one that caught more fraud (P slightly down, R way up, F1 ≈ 0.5 but MCC noticeably better).

**Pitfalls.**
- The denominator is zero when one row or column of the confusion matrix is zero. TorchMetrics returns 0 in that case; some implementations return NaN.
- Multiclass MCC exists (Gorodkin's formula) but is harder to interpret. Stick to binary or per-class.

---

### CohenKappa (`tm.classification.CohenKappa`)

**What it computes.** Inter-rater agreement: `(p_o − p_e) / (1 − p_e)`, where `p_o` is observed agreement and `p_e` is chance agreement.

**Intuition.** Like accuracy, but discounted by what two random scorers would agree on by luck.

**Range / direction.** `[-1, 1]`. 0 = chance, 1 = perfect. Higher better.

**When to use.** Two raters / two models comparing labels on the same dataset. Class imbalance.

**Real-world scenario.** Medical-imaging team has two radiologists *and* the model rate the same 1k cases. Cohen's κ between {model, radiologist1} is the publication-grade agreement metric — accuracy on this dataset would be inflated by the dominant "negative" class.

**Pitfalls.**
- Linear vs quadratic weighting (`weights="linear"|"quadratic"`) matters for ordered categories. Always pick `quadratic` for ordinal scales (e.g., 5-star ratings).

---

## Curve-based and rank-based

### ConfusionMatrix (`tm.classification.ConfusionMatrix`)

**What it computes.** The full matrix `C[i, j] = # samples with true class i predicted as j`.

**Intuition.** All other classification metrics are functions of this matrix. The *first* thing to look at when a model regresses; per-class scalars hide which classes confuse with which.

**Range / direction.** Counts. Diagonal big = good.

**Real-world scenario.** A speech-command model that's lost 2 points of accuracy in production: confusion matrix shows "left" → "right" jumped from 0.5% to 3% — the issue is acoustic, not the model. With only the scalar, you'd be running ablations for a week.

**Code.**
```python
from torchmetrics.classification import ConfusionMatrix
cm = ConfusionMatrix(task="multiclass", num_classes=10, normalize="true")
mat = cm(preds, target)  # shape (10, 10)
```

**Pitfalls.**
- `normalize="true"` (per-row), `"pred"` (per-column), `"all"` (overall). Mixing them across reports is a common chart-confusion bug.

---

### ROC (`tm.classification.ROC`) and AUROC (`tm.classification.AUROC`)

**What it computes.** ROC plots TPR vs FPR as the threshold sweeps. AUROC = area under that curve.

**Intuition.** AUROC = Pr(score(positive) > score(negative)). Threshold-free. Equal to Wilcoxon-Mann-Whitney U statistic / (n_pos · n_neg).

**Range / direction.** `[0, 1]`. 0.5 = random. Higher better.

**When to use.** Ranking quality at *any* threshold. Compare two models without committing to an operating point.

**When NOT to use.** Severe class imbalance. AUROC is dominated by the easy negatives — a model that bumps up a few mid-rank fraudsters from rank 50 to rank 5 barely moves AUROC, but moves the precision at top-100 a lot. **Use AUPRC instead.**

**Real-world scenario.** Two ad-CTR models compared offline. AUROC says model B is +0.001 better. AUPRC says model B is +5% better — because positive impressions are rare. Production wins are visible in AUPRC.

**Code.**
```python
from torchmetrics.classification import AUROC
aur = AUROC(task="multiclass", num_classes=5, average="macro")
```

**Pitfalls.**
- AUROC ignores calibration. Two models with identical AUROC can give wildly different probabilities; one needs calibration before deployment.
- For multiclass, `average="macro"` uses one-vs-rest. Watch out: this is *not* the same as the multinomial Hand-Till AUROC formula.

---

### AveragePrecision / AUPRC (`tm.classification.AveragePrecision`)

**What it computes.** Area under the precision-recall curve, summed as `Σ (R_n − R_{n−1}) · P_n`.

**Intuition.** "Across all thresholds, what's the average precision?" Sensitive to the rare-positive setting.

**Range / direction.** `[0, 1]`. The trivial baseline = base-rate (not 0.5 like AUROC). Higher better.

**When to use.** Imbalanced binary or multilabel. Information retrieval. Anomaly detection.

**Real-world scenario.** Fraud detection at 0.05% positive rate: AUPRC of 0.30 is a strong model (baseline = 0.0005). AUROC of 0.99 looks great but mostly measures the easy negatives — model B with AUROC 0.985 and AUPRC 0.45 ships.

**Pitfalls.**
- Multiple reasonable definitions: TorchMetrics uses the step-wise integral (no interpolation). Sklearn `average_precision_score` matches. PASCAL-VOC's "11-point interpolated" AP is *different* — be careful when copying numbers across ecosystems.

---

### PrecisionRecallCurve (`tm.classification.PrecisionRecallCurve`)

**What it computes.** The full `(P, R, thresholds)` arrays so you can plot the curve and pick an operating point.

**Real-world scenario.** Picking a deployment threshold: business says "we can manually review 50 alerts/day." Find the threshold where precision is highest *with recall ≥ what hits 50 alerts*. PR curve gives you the data; the scalar AUPRC doesn't.

---

### Hinge (`tm.classification.HingeLoss`)

**What it computes.** `max(0, 1 − y · f(x))` averaged over the batch. SVM loss.

**Intuition.** Penalizes margin violations. Zero loss when correctly classified with margin ≥ 1; linear penalty otherwise.

**Range / direction.** `[0, ∞)`. Lower better.

**Real-world scenario.** Tracking SVM-style learning during training; rarely a *deployment* metric, more a training diagnostic.

---

### CalibrationError / ECE (`tm.classification.CalibrationError`)

**What it computes.** Expected Calibration Error: `Σ_b (|B_b|/N) · |acc(B_b) − conf(B_b)|`. Bins predictions by confidence, measures gap between confidence and actual accuracy.

**Intuition.** "When the model says 90%, is it right 90% of the time?" Different from accuracy: a model can be accurate but overconfident, or right *and* well-calibrated.

**Range / direction.** `[0, 1]`. **Lower better.**

**When to use.** Anywhere downstream code uses *probabilities*, not class IDs: thresholds tied to expected cost, abstention, ensembling, A/B testing impressions.

**Real-world scenario.** A loan-default model with AUROC 0.86 but ECE 0.18 — the predicted "30% default risk" actually defaults 12% of the time. Under-prices the loan; the calibration step (Platt or isotonic) goes in *after* the model, and we monitor ECE in the dashboard.

**Code.**
```python
from torchmetrics.classification import CalibrationError
ece = CalibrationError(task="binary", n_bins=15, norm="l1")
```

**Pitfalls.**
- `n_bins` matters: too few hides miscalibration, too many is noisy. 15 is the standard.
- `norm="l1"` (ECE), `"l2"` (RMSCE), `"max"` (MCE) — don't mix across reports.

---

### LogAUC (`tm.classification.LogAUROC`)

**What it computes.** Area under ROC where the FPR axis is log-transformed, focusing on low-FPR regions.

**Intuition.** "How well does the model rank when we only care about the top of the list?" Standard in cheminformatics and large-scale screening.

**Real-world scenario.** Drug-target screening: only the top 0.1% candidates get assayed. Standard AUROC averages over all FPR — useless for our problem. LogAUC at the low-FPR region is the right metric.

---

### EER — Equal Error Rate (`tm.classification.EER`)

**What it computes.** The threshold where FPR == FNR; reports that error rate (= FPR = FNR at the crossing).

**Intuition.** A single number that summarizes ROC at a "balanced" operating point.

**Range / direction.** `[0, 0.5]`. **Lower better.**

**Real-world scenario.** Speaker verification: the conventional industry KPI. Used because biometric vendors all report it on the same scale, and authentication products want FPR = FNR.

---

### PrecisionAtFixedRecall, RecallAtFixedPrecision, SensitivityAtSpecificity, SpecificityAtSensitivity

**What they compute.** Pick the threshold that hits the fixed constraint, return the other side.

**Real-world scenario.** Regulators say "sensitivity ≥ 95%". You pick the threshold satisfying that and report the resulting specificity. `SpecificityAtSensitivity(min_sensitivity=0.95)` gives you the deployable number directly. **This is how you should report most binary medical/safety models** — not raw accuracy.

**Code.**
```python
from torchmetrics.classification import SpecificityAtSensitivity
m = SpecificityAtSensitivity(task="binary", min_sensitivity=0.95)
spec, threshold = m(preds, target)
```

**Pitfalls.**
- Returns a *tuple* `(metric_value, threshold)`. Easy to forget when logging.
- The threshold is the one hitting the constraint *on this dataset*. Re-fit on validation, hold out test for the final number.

---

### Ranking metrics (`tm.classification.MultilabelRankingAveragePrecision`, `MultilabelRankingLoss`, `MultilabelCoverageError`)

**What they compute.**
- **Ranking AP**: per-sample, the average precision of the predicted ranking restricted to the true labels. (Different from `AveragePrecision` which is curve-area.)
- **Ranking Loss**: average # of (label_i_positive, label_j_negative) pairs ranked in the wrong order.
- **Coverage Error**: how far down the ranked list you must go to recover all true labels.

**Real-world scenario.** Document-tag predictor used by an ops team that sees the **top-k tags ranked**. Ranking metrics directly model that workflow; per-label F1 doesn't.

---

### GroupFairness (`tm.classification.BinaryFairness`, `BinaryGroupStatRates`)

**What it computes.** Per-group versions of TPR/FPR/Precision/etc., plus disparate-impact-style ratios.

**Intuition.** Same model, sliced by sensitive attribute (race, sex, age bucket). Single overall accuracy can be 90% while one group sits at 60%.

**Range / direction.** Per-group rates `[0, 1]`; ratios target 1.0.

**Real-world scenario.** Loan-approval model: overall AUROC 0.84 but `FPR(group=A) / FPR(group=B) = 1.7` — the model rejects qualified applicants from group B 1.7× more often. This is the metric Compliance reads, not AUROC.

**Pitfalls.**
- "Fair" has no single definition. Demographic parity, equalized odds, predictive parity — these can be mutually incompatible. Always pre-commit to *one* definition with Compliance/Legal *before* training.
- Sample sizes per group can be tiny → noisy ratios. Bootstrap CIs.

---

## Quick-reference: which classification metric for which scenario?

| Scenario | Primary | Secondary |
|---|---|---|
| Balanced multiclass image classifier | Accuracy | Macro F1 |
| Imbalanced binary (fraud, churn) | AUPRC, MCC | Precision@k |
| Heavy positive cost (medical screening) | Recall (≥ floor) | SpecificityAtSensitivity |
| Heavy negative cost (spam filter precision) | PrecisionAtFixedRecall | F0.5 |
| Probabilities used downstream | Calibration Error (ECE) | Brier |
| Multiclass with confusion structure | Confusion Matrix | Per-class F1 |
| Multilabel tagging (loose) | Hamming, Macro F1 | Subset accuracy |
| Multilabel tagging (strict) | ExactMatch | Hamming |
| Top-k ranked outputs | Multilabel Ranking AP | Coverage Error |
| Two raters / model-vs-human | Cohen's κ | Confusion matrix |
| Single number for imbalanced binary | MCC | AUPRC |
| Top-of-list focus (rare positives) | LogAUC | AUPRC |
| Biometric / verification | EER | FPR/FNR curves |
| Fairness audit | BinaryFairness | per-group AUROC, AUPRC |

---

[Continue → interview deep-dive](./interview-deep-dive.md)
